# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:14:25 2016

@author: phyrct
"""

__all__=['core']
from .core import ImageArray
from .Folders import ImageFolder, ImageStack, KerrStack, MaskStack

KERR_IM=[0,512,0,672]