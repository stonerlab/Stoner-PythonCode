"""Stoner.plot sub-package - contains classes and functions for visuallising data.

Most of the plotting functionality is provided by the :class:`.PlotMixin` mixin class which is available through the
:py:class:`Stoner.Data` class.

The :mod:`.formats` module provides a set of template classes for producing different plot styles and formats. The
:py:mod:`Stoner.plot.util` module provides
some handy utility functions.
"""

from .core import PlotMixin

__all__ = ["PlotMixin", "formats", "utils"]
