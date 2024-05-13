# -*- coding: utf-8 -*-
"""Common routines for reading maximus files."""
__all__ = ["read_scan", "hdr_to_dict", "flatten_header", "process_key", "read_images", "read_pointscan"]
import json
from pathlib import Path
import re

import numpy as np

from ...tools.file import FileManager


def read_scan(file_root):
    """Find the .hdr and .xim/,xsp files for a scan load them into memory.

    Args:
        file_root (str): This is the part of the hdr filename beore the extension.

    Returns:
        (dict,ndarray, (n-1d arrays)):
            Returns the metadata and an ndarray of the scan data and then n 1D arrays of the axes.
    """
    hdr = Path(str(file_root) + ".hdr")
    header = hdr_to_dict(hdr, to_python=True)
    scan_type = header["ScanDefinition"]["Type"]

    if "Image Scan" in scan_type:
        data, dims = read_images(hdr.parent.glob(f"{hdr.stem}*.xim"), header)
    elif "Point Scan" in scan_type:
        data, dims = read_pointscan(hdr.parent.glob(f"{hdr.stem}*.xsp"), header)
    else:
        raise ValueError(f"Unrecognised scan type {scan_type}")

    return header, data, dims


def hdr_to_dict(filename, to_python=True):
    """Convert .hdr metadata file to  json or python dictionary.

    Args:
        filename (str):
            Name of file to read (can also be a pathlib.Path).

    Keyword Arguments:
        to_python (bool):
            If true, return a python dictionary, otherwise return a json text string.

    Returns:
        (dict or str):
            Either the header file as a python dictionary, or a json string.
    """
    bare = re.compile(r"([\s\{])([A-Za-z][A-Za-z0-9_]*)\s\:")  # Match for keys
    term = re.compile(r",\s*([\]\}])")  # match for extra , at the end of a dict or list
    nan = re.compile(r"([\-0-9\.]+\#QNAN)")  # Handle NaN values

    # Use oathlib to suck in the file
    with FileManager(filename, "r") as f:
        hdr = f.read()
    # Simple string replacements first
    stage1 = hdr.replace("=", ":").replace(";", ",").replace("(", "[").replace(")", "]")
    # Wrap in { }
    stage2 = f"{{{stage1}}}"
    # Regexp replacements next
    stage3 = bare.sub('\\1"\\2" :', stage2)
    stage4 = term.sub("\\n\\1", stage3)
    stage5 = nan.sub("NaN", stage4)

    if to_python:
        ret = process_key(json.loads(stage5))
    else:  # orettyify the json
        ret = json.dumps(json.loads(stage5), indent=4)

    return ret


def flatten_header(value):
    """Flatten nested dictionaries."""
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict):
        return value
    dels = []
    adds = {}
    for key, item in value.items():
        item = flatten_header(item)
        if isinstance(item, dict):
            for item_key, item_val in item.items():
                adds[f"{key}.{item_key}"] = item_val
            dels.append(key)
    for key in dels:
        del value[key]
    value.update(adds)
    return value


def process_key(value):
    """Carry out post loading processing of data structures."""
    if isinstance(value, dict):
        for key, val in value.items():
            value[key] = process_key(val)
        return value
    if isinstance(value, list):
        if len(value) > 0 and isinstance(value[0], int) and len(value) == value[0] + 1:  # Lists prepended with size
            del value[0]
        a = np.array(value)
        if a.dtype.kind in ["f", "i", "u", "U"]:  # convert arrays to arrays
            return a
        value = [process_key(v) for v in value]
    return value


def read_images(files, header):
    """Read one or more .xim files and construct the data array.

    Args:
        files (glob): glob pattern of xim files to read.
        header (dict): contents of the .jdr file.

    Returns:
        data (ndarray): 2D or 3D data.
        dims (tuple of 1D arrays): 2 or 3 1D arrays corresponding to the dimensions of data.
    """
    xims = list(files)
    scandef = header["ScanDefinition"]
    region = scandef["Regions"][0]  # FIXME assumes a single region in the data
    if len(xims) > 1:
        data = np.stack([np.genfromtxt(x)[::-1] for x in xims]).T
    elif len(xims) == 1:
        data = np.genfromtxt(xims[0])[::-1]
    else:  # no files !
        raise IOError("No Images located")
    xpts = region["PAxis"]["Points"]
    ypts = region["QAxis"]["Points"]
    if data.ndim == 3:
        zpts = scandef["StackAxis"]["Points"]
        dims = (xpts, ypts, zpts)
    else:
        dims = (xpts, ypts)
    return data, dims


def read_pointscan(files, header):
    """Read one or more .xsp files and construct the data array.

    Args:
        files (glob): glob pattern of xim files to read.
        header (dict): contents of the .jdr file.

    Returns:
        data (ndarray): 2D or 3D data.
        dims (tuple of 1D arrays): 2 or 3 1D arrays corresponding to the dimensions of data.
    """
    xsps = list(files)
    scandef = header["ScanDefinition"]
    region = scandef["Regions"][0]  # FIXME assumes a single region in the data
    if len(xsps) > 1:
        data = np.stack([np.genfromtxt(x)[:, 1] for x in xsps]).T
    elif len(xsps) == 1:
        data = np.genfromtxt(xsps[0])[:, 1]
    else:  # No files !
        raise IOError("No Spectra located")
    xpts = region["PAxis"]["Points"]
    if data.ndim == 2:
        zpts = scandef["StackAxis"]["Points"]
        dims = (xpts, zpts)
    else:
        dims = (xpts,)
    return data, dims
