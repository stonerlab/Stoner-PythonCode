#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Defines type hint types for various common types in the package."""
# pylint: disable-all
import typing

from Stoner.core.setas import setas

# Column Indices
Single_Column_Index = typing.Union[str, int]
Column_Index = typing.Union[typing.Sequence[Single_Column_Index], Single_Column_Index]

# Setas
Setas_Base = typing.Sequence[str]
Setas_Dict = typing.Mapping[str, Column_Index]
Setas = typing.Union[Setas_Base, Setas_Dict, str, setas]
