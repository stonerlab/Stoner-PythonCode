# -*- coding: utf-8 -*-
"""Image Loading routines for HDF5 format."""
from copy import deepcopy

import numpy as np

from ..decorators import register_loader
from ...tools import make_Data
from ...tools.file import get_filename


@register_loader(
    patterns=[(".hdf5", 16), (".hdf", 16)],
    mime_types=[("application/x-hdf", 16), ("application/x-hdf5", 16)],
    name="STXMImage",
    what="Image",
)
def load_stxm_image(new_data, *args, **kargs):
    """Initialise and load a STXM image produced by Pollux.

    Keyword Args:
        regrid (bool):
            If set True, the gridimage() method is automatically called to re-grid the image to known coordinates.
    """
    filename, args, kargs = get_filename(args, kargs)
    regrid = kargs.pop("regrid", False)
    kargs.setdefault("filetype", "SLS_STXMFile")
    bcn = kargs.pop("bcn", False)
    d = make_Data(filename, *args, **kargs)
    new_data.image = d.data
    new_data.metadata = deepcopy(d.metadata)
    new_data.filename = d.filename
    if isinstance(regrid, tuple):
        new_data.gridimage(*regrid)
    elif isinstance(regrid, dict):
        new_data.gridimage(**regrid)
    elif regrid:
        new_data.gridimage()
    if bcn:
        if regrid:
            new_data.metadata["beam current"] = new_data.metadata["beam current"].gridimage()
        new_data.image /= new_data["beam current"]
    new_data.polarization = np.sign(new_data.get("collection.polarization.value", 0))
    return new_data
