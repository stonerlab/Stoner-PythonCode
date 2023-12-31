# -*- coding: utf-8 -*-
"""Module to check late importing for DataFile sub ty[es in load"""

from Stoner.formats.generic import load_tdi_format
from Stoner.formats import register_loader


@register_loader(name="dummy.ArbClass", what="Data")
def arb_load(*args,**kargs):
    """Just a dummy loader routine that passes through to tdi_load."""
    return load_tdi_format(*args, **kargs)
