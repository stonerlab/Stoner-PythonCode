#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Defines type hint types for various common types in the package."""
# pylint: disable-all
import typing

from ..compat import string_types, int_types, path_types, _pattern_type
from .setas import setas

# Column Indices
Single_Column_Index = typing.Union[str, bytes, int, _pattern_type]
Column_Index = typing.Union[typing.Sequence[Single_Column_Index], Single_Column_Index]

# Setas
Setas_Base = typing.Sequence[str]
Setas_Dict = typing.Mapping[str, Column_Index]
Setas = typing.Union[Setas_Base, Setas_Dict, str, setas]
