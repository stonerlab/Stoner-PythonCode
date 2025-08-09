"""Provides utility classes for writing simple data analysis scripts more easily.

It has been developed by members of the `Condensed Matter Group<http://www.stoner.leeds.ac.uk/>` at the
`University  of Leeds<http://www.leeds.ac.uk>`.
"""

# pylint: disable=import-error
__all__ = [
    "core",
    "analysis",
    "formats",
    "plot",
    "tools",
    "Image",
    "Data",
    "DataFolder",
    "ImageFile",
    "ImageFolder",
    "ImageStack",
    "set_option",
    "get_option",
    "Options",
    "HDF5",
    "Zip",
]

import pathlib

# These fake the old namespace if you do an import Stoner

from . import HDF5, Image, Zip, analysis, core, formats, plot, tools
from .core.data import Data
from .folders import DataFolder
from .Image import ImageFile, ImageFolder, ImageStack
from .tools import Options as _Options
from .tools import get_option, set_option

Options = _Options()


__version_info__ = ("0", "11", "0")
__version__ = ".".join(__version_info__)

__homepath__ = pathlib.Path(__file__).parent.resolve()
__datapath__ = (__homepath__ / ".." / "sample-data").resolve()
__home__ = str(__homepath__)
