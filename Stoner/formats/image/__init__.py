"""Sub-package of routines to load ImageFile objects in different formats.

This sub-package is lazy-loaded but then pulls in all of the modules in order to register the routines.
"""

__all__ = ["generic", "hdf5", "facilities", "maximus"]

from . import generic, hdf5, facilities, maximus
