# -*- coding: utf-8 -*-
"""The :mod:`Stoner.Image` package provides a means to carry out image processing functions in a smilar way that :mod:`Stoner.Core` and :class:`Stoner.Data` and
:class:`Stoner.DataFolder` do. The :mod:`Stomner.Image.core` module contains the key classes for achieving this.
"""

__all__ = [
    "core",
    "folders",
    "stack",
    "kerr",
    "ImageArray",
    "ImageFile",
    "ImageFolder",
    "ImageStack",
    "KerrArray",
    "KerrStack",
    "MaskStack",
    "ImageStack2",
]
from .core import ImageArray, ImageFile
from .folders import ImageFolder
from .stack import ImageStack, ImageStack2
from .kerr import KerrArray, KerrStack, MaskStack
