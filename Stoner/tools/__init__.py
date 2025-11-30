#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Support functions for Stoner package.

These functions depend only on Stoner.compat which is used to ensure a consistent namespace between python 2.7 and 3.x.
"""
__all__ = [
    "AttributeStore",
    "all_size",
    "all_type",
    "fix_signature",
    "format_error",
    "format_val",
    "html_escape",
    "isanynone",
    "isComparable",
    "isnone",
    "isiterable",
    "isLikeList",
    "isproperty",
    "isTuple",
    "copy_into",
    "make_Data",
    "make_Class",
    "quantize",
    "tex_escape",
    "TypedList",
    "Options",
    "get_option",
    "set_option",
    "ordinal",
    "file",
    "decorators",
]
import numpy as np

from . import decorators
from .classes import Options, TypedList
from .classes import attributeStore as AttributeStore
from .classes import copy_into, get_option, set_option
from .decorators import fix_signature, make_Class, make_Data
from .formatting import (
    format_error,
    format_val,
    html_escape,
    ordinal,
    quantize,
    tex_escape,
)
from .tests import (
    all_size,
    all_type,
    isanynone,
    isComparable,
    isiterable,
    isLikeList,
    isnone,
    isproperty,
    isTuple,
)

operator = {
    "eq": lambda k, v: k == v,
    "ne": lambda k, v: k != v,
    "contains": lambda k, v: v in k,
    "in": lambda k, v: k in v,
    "icontains": lambda k, v: k.upper() in str(v).upper(),
    "iin": lambda k, v: str(v).upper() in k.upper(),
    "lt": lambda k, v: k < v,
    "le": lambda k, v: k <= v,
    "gt": lambda k, v: k > v,
    "ge": lambda k, v: k >= v,
    "between": lambda k, v: np.logical_and(min(v) < k, k < max(v)),
    "ibetween": lambda k, v: np.logical_and(min(v) <= k, k <= max(v)),
    "ilbetween": lambda k, v: np.logical_and(min(v) <= k, k < max(v)),
    "iubetween": lambda k, v: np.logical_and(min(v) < k, k <= max(v)),
    "startswith": lambda k, v: str(v).startswith(k),
    "istartswith": lambda k, v: str(v).upper().startswith(k.upper()),
    "endsswith": lambda k, v: str(v).endswith(k),
    "iendsswith": lambda k, v: str(v).upper().endswith(k.upper()),
}
