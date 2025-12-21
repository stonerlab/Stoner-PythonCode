"""Sub-package of routines to load Data objects in different formats.

This sub-package is lazy-loaded but then pulls in all of the modules in order to register the routines.
"""

__all__ = ["generic", "hdf5", "instruments", "facilities", "rigs", "simulations", "attocube", "maximus", "zipped"]

from . import (
    attocube,
    facilities,
    generic,
    hdf5,
    instruments,
    maximus,
    rigs,
    simulations,
)
from . import zip as zipped
