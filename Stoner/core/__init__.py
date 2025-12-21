#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides core functionality for the :py:class:`Stoner.Data` class."""

__all__ = [
    "metadataObject",
    "typeHintedDict",
    "regexpDict",
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

from .base import regexpDict, typeHintedDict, metadataObject, string_to_type
from .setas import setas as _setas
from .array import DataArray
from . import utils, exceptions, base, array
from . import Typing
