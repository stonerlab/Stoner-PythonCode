# -*- coding: utf-8 -*-
"""Loader for maximus scan image."""
from pathlib import Path


from ..decorators import register_loader
from ...core.exceptions import StonerLoadError
from ..utils.maximus import read_scan, flatten_header, hdr_to_dict
from ...tools.file import get_filename


@register_loader(
    patterns=[(".hdr", 16), (".xim", 16)], mime_types=("text/plain", 16), name="MaximusImage", what="Image"
)
def load_maximus_image(new_data, *args, **kargs):
    """Load an ImageFile by calling the ImageArray method instead."""
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
    new_data.metadata.update(hdr)
    new_data.image = read_scan(stem)[1]
    return new_data
