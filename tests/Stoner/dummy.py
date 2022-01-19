# -*- coding: utf-8 -*-
"""Module to check late importing for DataFile sub ty[es in load"""

from Stoner.formats.generic import TDIFile
from Stoner.tools.decorators import register_loader

@register_loader(patterns=["dummy.ArbClass"])
def dummyLoader(filename,**kargs):
    return TDIFile(filename, **kargs)
