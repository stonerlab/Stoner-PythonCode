"""Core support for working with collections of files in the :py:class:`Stoner.DataFolder`."""
import importlib

__all__ = ["core", "each", "groups", "metadata", "mixins", "utils", "DataFolder", "PlotFolder"]
_mixins = ["DataFolder", "PlotFolder"]


def __getattr__(name):
    """Lazy import required module."""
    if name in __all__:
        if name in _mixins:
            ret = importlib.import_module(".mixins", __name__)
            return getattr(ret, name)
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
