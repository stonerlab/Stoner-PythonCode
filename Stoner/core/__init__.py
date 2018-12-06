#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides core functionality for the :py:class:`Stoner.Data` class."""

__all__=["metadataObject","typeHintedDict","regexpDict","_setas","DataArray"]

from .base import regexpDict,typeHintedDict,metadataObject
from .setas import setas as _setas
from .array import DataArray
