"""The Stoner Python package provides utility classes for writing simple data analysis scripts more easily.  It has been developed by members
of the `Condensed Matter Group<http://www.stoner.leeds.ac.uk/>` at the `University of Leeds<http://www.leeds.ac.uk>`.
"""
# pylint: disable=import-error
__all__ = [
    "formats",
    "plot",
    "tools",
    "Data",
    "DataFolder",
    "ImageFile",
    "ImageFolder",
    "set_option",
    "get_option",
    "Options",
]

# These fake the old namespace if you do an import Stoner
from os import path as _path_

from . import formats, plot, tools
from .core.data import Data
from .Folders import DataFolder
from .Image import ImageFile, ImageFolder

from .tools import set_option, get_option, _Options

Options = _Options()


__version_info__ = ("0", "10", "0dev")
__version__ = ".".join(__version_info__)

__home__ = _path_.realpath(_path_.dirname(__file__))
