#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Defines type hint types for various common types in the package."""
# pylint: disable-all
import typing
from pathlib import PurePath

from numpy import ndarray

from ..compat import _pattern_type

# Types used in compat code
RegExp = _pattern_type
String_Types = typing.Union[str, bytes]
Int_Types = int
Path_Types = typing.Union[String_Types, PurePath]

# Column Indices
Single_Column_Index = typing.Union[str, bytes, int, RegExp]
Column_Index = typing.Union[typing.Sequence[Single_Column_Index], Single_Column_Index]

# Setas
Setas_Base = typing.Sequence[str]
Setas_Dict = typing.Mapping[str, Column_Index]
Setas = typing.Union[Setas_Base, Setas_Dict, str, "setas"]

# Other useful types
Filename = typing.Union[Path_Types, bool]
Numeric = typing.Union[float, int, complex]
NumericArray = typing.Union[Numeric, ndarray]
