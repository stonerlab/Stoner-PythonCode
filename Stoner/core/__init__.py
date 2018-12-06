#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 18:50:37 2018

@author: phygbu
"""
__all__=["metadataObject","typeHintedDict","regexpDict","_setas","DataArray"]
from .base import regexpDict,typeHintedDict,metadataObject
from .setas import setas as _setas
from .array import DataArray
