# -*- coding: utf-8 -*-
"""The :mod:`Stoner.Image` package provides a means to carry out image processing functions in a smilar way that
:mod:`Stoner.Core` and :class:`Stoner.Data` and :class:`Stoner.DataFolder` do. The :mod:`Stomner.Image.core` module
contains the key classes for achieving this.
"""
import importlib

__all__ = [
    "attrs",
    "core",
    "folders",
    "stack",
    "kerr",
    "widgets",
    "ImageArray",
    "ImageFile",
    "ImageFolder",
    "ImageStack",
    "KerrArray",
    "KerrStack",
    "MaskStack",
]
_sub_imports = {
    "ImageArray": "core",
    "ImageFile": "core",
    "ImageFolder": "folders",
    "ImageStack": "stack",
    "KerrArray": "kerr",
    "KerrStack": "kerr",
    "MaskStack": "kerr",
}


def __getattr__(name):
    """Lazy import required module."""
    if name in __all__:
        if name in _sub_imports:
            ret = importlib.import_module(f".{_sub_imports[name]}", __name__)
            return getattr(ret, name)
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
