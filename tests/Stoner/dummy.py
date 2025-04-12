# -*- coding: utf-8 -*-
"""Module to check late importing for DataFile sub ty[es in load"""

from Stoner.formats.decorators import register_loader
from Stoner.formats.data.generic import load_tdi_format


@register_loader(
    patterns=[(".dat", 8), (".txt", 8), ("*", 8)],
    mime_types=[("application/tsv", 8), ("text/plain", 8), ("text/tab-separated-values", 8)],
    name="dummy.ArbClass",
    what="Data",
)
def load_arbclass_format(new_data, *args, **kargs):
    return load_tdi_format(new_data, *args, **kargs)
