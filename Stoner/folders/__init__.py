"""Core support for working with collections of files in the :py:class:`Stoner.DataFolder`."""

__all__ = ["core", "each", "groups", "metadata", "mixins", "utils", "DataFolder", "PlotFolder"]

from . import mixins
from .mixins import DataFolder, PlotFolder
