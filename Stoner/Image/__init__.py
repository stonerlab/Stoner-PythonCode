# -*- coding: utf-8 -*-
"""Supackage to provide image processing capabilities.

The :mod:`Stoner.Image` package provides a means to carry out image processing functions in a smilar way that
:mod:`Stoner.core`, :class:`Stoner.core.data.Data` and :class:`Stoner.folders.mixins.DataFolder` do.
The :mod:`Stoner.Image.core` module
contains the key classes for achieving this.
"""

__all__ = [
    "attrs",
    "core",
    "folders",
    "stack",
    "kerr",
    "widgets",
    "ImageFile",
    "ImageFolder",
    "ImageStack",
    "KerrStack",
    "MaskStack",
]
from . import attrs, widgets
from .core import ImageFile
from .numerical import numerical_image
from .folders import ImageFolder
from .kerr import KerrImageFile, KerrStack, MaskStack
from .stack import ImageStack
