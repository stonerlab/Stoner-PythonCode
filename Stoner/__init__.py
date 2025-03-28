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
]

# These fake the old namespace if you do an import Stoner
from os import path as _path_
import pathlib

from . import core, analysis, formats, plot, tools, Image
from .core.data import Data
from .Folders import DataFolder
from .Image import ImageFile, ImageFolder, ImageStack

from .tools import set_option, get_option, Options as _Options

Options = _Options()


__version_info__ = ("0", "10", "12")
__version__ = ".".join(__version_info__)

__homepath__ = pathlib.Path(__file__).parent.resolve()
__datapath__ = (__homepath__ / ".." / "sample-data").resolve()
__home__ = str(__homepath__)
