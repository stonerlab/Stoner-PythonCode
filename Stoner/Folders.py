"""Stoner.Folders : Classes for working collections of :class:`.Data` files.

The core classes provides a means to access them as an ordered collection or as a mapping.
"""

__all__ = ["DataFolder", "PlotFolder"]

from Stoner.tools import make_Data
from .folders.core import baseFolder

from .folders.mixins import DiskBasedFolderMixin, DataMethodsMixin, PlotMethodsMixin


class DataFolder(DataMethodsMixin, DiskBasedFolderMixin, baseFolder):
    """Provide an interface to manipulating lots of data files stored within a directory structure on disc.

    By default, the members of the DataFolder are instances of :class:`Stoner.Data`. The DataFolder emplys a lazy
    open strategy, so that files are only read in from disc when actually needed.

    .. inheritance-diagram:: DataFolder

    """

    def __init__(self, *args, **kargs):
        """Set the default type before creating the DataFolder."""
        self.type = kargs.pop("type", make_Data(None))
        super().__init__(*args, **kargs)


class PlotFolder(PlotMethodsMixin, DataFolder):
    """A :py:class:`Stoner.folders.baseFolder` that knows how to ploth its underlying data files."""
