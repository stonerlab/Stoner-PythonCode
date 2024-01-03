"""Stoner.plot sub-package - contains classes and functions for visuallising data.

Most of the plotting functionality is provided by the :class:`.PlotMixin` mixin class which is available through the
:py:class:`Stoner.Data` class.

The :mod:`.formats` module provides a set of template classes for producing different plot styles and formats. The
:py:mod:`Stoner.plot.util` module provides
some handy utility functions.
"""
import importlib

__all__ = ["PlotMixin", "formats", "utils"]
_sub_imports = {"PlotMixin": "core"}


def __getattr__(name):
    """Lazy import required module."""
    if name in __all__:
        if name in _sub_imports:
            ret = importlib.import_module(f".{_sub_imports[name]}", __name__)
            return getattr(ret, name)
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
