"""Provides additional functionality for doing curve fitting to data."""
import importlib

__all__ = ["odr_Model", "FittingMixin", "models"]
_mixins = ["odr_Model", "FittingMixin"]


def __getattr__(name):
    """Lazy import required module."""
    if name in __all__:
        if name in _mixins:
            ret = importlib.import_module(".mixins", __name__)
            return getattr(ret, name)
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
