# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:14:25 2016

@author: phyrct
"""

__all__=['ImageArray','ImageFolder','ImageStack','core','folders','stack','MaskStack']
from .core import ImageArray
from .folders import ImageFolder
from .stack import ImageStack, KerrStack, MaskStack

KERR_IM=[0,512,0,672]