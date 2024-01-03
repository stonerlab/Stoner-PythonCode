"""Subpaclage to support the data analysis functions."""
import importlib

__all__ = ["fitting", "utils", "columns", "filtering", "features"]


def __getattr__(name):
    """Lazy import required module."""
    if name in __all__:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
