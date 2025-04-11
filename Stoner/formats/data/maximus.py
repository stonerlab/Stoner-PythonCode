# -*- coding: utf-8 -*-
"""Loader for maximus spectra files."""
from pathlib import Path

import numpy as np

from ..decorators import register_loader
from ...core.exceptions import StonerLoadError
from ..utils.maximus import read_scan, flatten_header, hdr_to_dict
from ...tools.file import get_filename


@register_loader(
    patterns=[(".hdr", 16), (".xsp", 16)], mime_types=("text/plain", 16), name="MaximusSpectra", what="Data"
)
def load_maximus_spectra(new_data, *args, **kargs):
    """Maximus xsp file loader routine.

    Args:
        filename (string or bool): File to load. If None then the existing filename is used,
            if False, then a file dialog will be used.

    Returns:
        A copy of the itnew_data after loading the data.
    """
    filename, args, kargs = get_filename(args, kargs)
    if filename is None or not filename:
        new_data.get_filename("r")
    else:
        new_data.filename = filename
    # Open the file and read the main file header and unpack into a dict
    try:
        pth = Path(new_data.filename)
    except (TypeError, ValueError) as err:
        raise StonerLoadError("Can only open things that can be converted to paths!") from err
    if pth.suffix != ".hdr":  # Passed a .xim or .xsp file in instead of the hdr file.
        pth = Path("_".join(str(pth).split("_")[:-1]) + ".hdr")
    stem = pth.parent / pth.stem

    try:
        hdr = flatten_header(hdr_to_dict(pth))
        if "Point Scan" not in hdr["ScanDefinition.Type"]:
            raise StonerLoadError("Not an Maximus Single Image File")
    except (StonerLoadError, ValueError, TypeError, IOError) as err:
        raise StonerLoadError("Error loading as Maximus File") from err
    header, data, dims = read_scan(stem)
    new_data.metadata.update(flatten_header(header))
    new_data.data = np.column_stack((dims[0], data))
    headers = [new_data.metadata["ScanDefinition.Regions.PAxis.Name"]]
    if len(dims) == 2:
        headers.extend([str(x) for x in dims[1]])
    else:
        headers.append(new_data.metadata["ScanDefinition.Channels.Name"])
    new_data.column_headers = headers
    new_data.setas = "xy"
    return new_data


@register_loader(
    patterns=[(".hdr", 16), (".xim", 16)], mime_types=("text/plain", 16), name="MaximusImage", what="Data"
)
def load_maximus_data(new_data, *args, **kargs):
    """Load a maximus image, but to a Data object."""
    filename, args, kargs = get_filename(args, kargs)
    try:
        new_data.filename = filename
        pth = Path(new_data.filename)
    except TypeError as err:
        raise StonerLoadError(f"UUnable to interpret {filename} as a path like object") from err
    if pth.suffix != ".hdr":  # Passed a .xim or .xsp file in instead of the hdr file.
        pth = Path("_".join(str(pth).split("_")[:-1]) + ".hdr")
    stem = pth.parent / pth.stem

    try:
        hdr = flatten_header(hdr_to_dict(pth))
        if "Image Scan" not in hdr["ScanDefinition.Type"]:
            raise StonerLoadError("Not an Maximus Single Image File")
    except (StonerLoadError, ValueError, TypeError, IOError) as err:
        raise StonerLoadError("Error loading as Maximus File") from err
    data = read_scan(stem)[1]
    new_data.metadata.update(hdr)
    if data.ndim == 3:
        data = data[:, :, 0]
    new_data.data = data
    return new_data
