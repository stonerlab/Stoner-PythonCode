"""Core support for wokring with collections of files in the :py:class:`Stoner.DataFolder`."""

__all__ = [
    "core",
    "each",
    "groups",
    "metadata",
    "mixins",
    "utils",
    "baseFolder",
    "DataFolder",
    "PlotFolder",
    "DiskBasedFolderMixin",
]

from .core import baseFolder
from .mixins import DataFolder, DiskBasedFolderMixin, PlotFolder
