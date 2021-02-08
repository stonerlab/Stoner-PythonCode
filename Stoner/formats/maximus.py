# -*- coding: utf-8 -*-
"""
Quick hacked  module to read STXM image files from the MAXIMUS STXM beamline.

(C) 2021 Gavin Burnell <g.burnell@leeds.ac.uk>

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__all__ = ["hdr_to_dict", "read_scan", "MaximusImage"]

import json
import re
from pathlib import Path

import numpy as np

from ..Image import ImageFile
from ..Core import DataFile

from Stoner.core.exceptions import StonerLoadError


class MaximusSpectra(DataFile):

    """Provides a DataFile subclass for loading Point spectra from Maximus."""

    _patterns = ["*.hdr"]
    mime_type = ["text/plain"]

    priority = 16

    def _load(self, filename=None, *args, **kargs):
        """Maximus xsp file loader routine.

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.
        """
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename
        # Open the file and read the main file header and unpack into a dict
        try:
            hdr = _flatten_header(hdr_to_dict(self.filename))
            if "Point Scan" not in hdr["ScanDefinition.Type"]:
                raise StonerLoadError("Not an Maximus Single Image File")
        except (StonerLoadError, ValueError, TypeError, IOError) as err:
            raise StonerLoadError("Error loading as Maximus File") from err
        stem = self.filename[:-4]
        header, data, dims = read_scan(stem)
        self.metadata.update(_flatten_header(header))
        self.data = np.column_stack((dims[0], data))
        headers = [self.metadata["ScanDefinition.Regions.PAxis.Name"]]
        if len(dims) == 2:
            headers.extend([str(x) for x in dims[1]])
        else:
            headers.append(self.metadata["ScanDefinition.Channels.Name"])
        self.column_headers = headers
        self.setas = "xy"
        return self


class MaximusImage(ImageFile):

    """Provide a STXMImage like class for the Maximus Beamline."""

    _patterns = ["*.hdr"]

    mime_type = ["text/plain"]

    priority = 16

    def _load(self, filename, *args, **kargs):
        """Load an ImageFile by calling the ImageArray method instead."""
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename
        try:
            hdr = _flatten_header(hdr_to_dict(self.filename))
            if hdr["ScanDefinition.Type"] != "Image Scan":
                raise StonerLoadError("Not an Maximus Single Image File")
        except (StonerLoadError, ValueError, TypeError, IOError) as err:
            raise StonerLoadError("Error loading as Maximus File") from err
        stem = str(self.filename)[:-4]
        data = read_scan(stem)[1]
        self.metadata.update(hdr)
        self.image = data
        return self


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
    bare = re.compile("([\s\{])([A-Za-z][A-Za-z0-9_]*)\s\:")  # Match for keys
    term = re.compile(r",\s*([\]\}])")  # match for extra , at the end of a dict or list
    nan = re.compile(r"([\-0-9\.]+\#QNAN)")  # Handle NaN values

    # Use oathlib to suck in the file
    hdr = Path(filename).read_text()
    # Simple string replacements first
    stage1 = hdr.replace("=", ":").replace(";", ",").replace("(", "[").replace(")", "]")
    # Wrap in { }
    stage2 = f"{{{stage1}}}"
    # Regexp replacements next
    stage3 = bare.sub('\\1"\\2" :', stage2)
    stage4 = term.sub("\\n\\1", stage3)
    stage5 = nan.sub("NaN", stage4)

    if to_python:
        ret = _process_key(json.loads(stage5))
    else:  # orettyify the json
        ret = json.dumps(json.loads(stage5), indent=4,)

    return ret


def read_scan(file_root):
    """Find the .hdr and .xim/,xsp files for a scan load them into memory.

    Args:
        file_root (str): This is the part of the hdr filename beore the extension.

    Returns:
        (dict,ndarray, (n-1d arrays)):
            Returns the metadata and an ndarray of the scan data and then n 1D arrays of the axes.
    """
    hdr = Path(file_root + ".hdr")
    header = hdr_to_dict(hdr, to_python=True)
    scan_type = header["ScanDefinition"]["Type"]

    if "Image Scan" in scan_type:
        data, dims = _read_images(hdr.parent.glob(f"{hdr.stem}*.xim"), header)
    elif "Point Scan" in scan_type:
        data, dims = _read_pointscan(hdr.parent.glob(f"{hdr.stem}*.xsp"), header)
    else:
        raise ValueError(f"Unrecognised scan type {scan_type}")

    return header, data, dims


def _flatten_header(value):
    """Flatten nested dictionaries."""
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict):
        return value
    dels = []
    adds = {}
    for key, item in value.items():
        item = _flatten_header(item)
        if isinstance(item, dict):
            for item_key, item_val in item.items():
                adds[f"{key}.{item_key}"] = item_val
            dels.append(key)
    for key in dels:
        del value[key]
    value.update(adds)
    return value


def _process_key(value):
    """Carry out post loading processing of data structures."""
    if isinstance(value, dict):
        for key, val in value.items():
            value[key] = _process_key(val)
        return value
    if isinstance(value, list):
        if len(value) > 0 and isinstance(value[0], int) and len(value) == value[0] + 1:  # Lists prepended with size
            del value[0]
        a = np.array(value)
        if a.dtype.kind in ["f", "i", "u", "U"]:  # convert arrays to arrays
            return a
        value = [_process_key(v) for v in value]
    return value


def _read_images(files, header):
    """Read one or more .xim files and construct the data array.

    Args:
        files (glob): glob pattern of xim files to read.
        header (dict): contents of the .jdr file.

    Returns:
        data (ndarray): 2D or 3D data.
        dims (tuple of 1D arays): 2 or 3 1D arrays corresponding to the dimensions of data.
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


def _read_pointscan(files, header):
    """Read one or more .xim files and construct the data array.

    Args:
        files (glob): glob pattern of xim files to read.
        header (dict): contents of the .jdr file.

    Returns:
        data (ndarray): 2D or 3D data.
        dims (tuple of 1D arays): 2 or 3 1D arrays corresponding to the dimensions of data.
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


if __name__ == "__main__":

    # Test by reading all files
    for infile in Path(".").glob("*.hdr"):
        hdr, data, dims = read_scan(infile.stem)
