"""Stoner.plot sub-package - contains classes and functions for visuallising data.

Most of the plotting functionailty is provided by the :class:`.PlotMixin` mixin class which is available through the :class:`.Data` classs.

The :mod:`.formats` module provides a set of template classes for producing different plot styles and formats. The :mod:#util# module provides
some handy utility functions.
"""

from .core import PlotMixin, hsl2rgb
from .core import PlotFile

__all__ = ["PlotMixin", "hsl2rgb", "formats", "utils", "PlotFile"]
