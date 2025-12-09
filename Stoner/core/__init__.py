#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides core functionality for the :py:class:`Stoner.Data` class."""

__all__ = [
    "metadataObject",
    "TypeHintedDict",
    "RegexpDict",
    "_setas",
    "DataArray",
    "array",
    "base",
    "interfaces",
    "methods",
    "operators",
    "property",
    "setas",
    "string_to_type",
    "exceptions",
    "utils",
    "Typing",
]

from . import Typing, array, base, exceptions, utils
from .array import DataArray
from .base import RegexpDict, TypeHintedDict, metadataObject, string_to_type
from .setas import Setas as _setas
