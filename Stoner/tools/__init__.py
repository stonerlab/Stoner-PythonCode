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
    "register_loader",
    "lookup_loaders",
    "make_Data",
    "quantize",
    "tex_escape",
    "typedList",
    "Options",
    "get_option",
    "set_option",
    "ordinal",
    "file",
    "decorators",
]
import inspect
from collections.abc import Iterable, MutableSequence
from copy import deepcopy
from importlib import import_module

from numpy import (  # pylint: disable=redefined-builtin
    dtype,
    floor,
    isnan,
    log10,
    logical_and,
    ndarray,
    round,
)

from ..compat import bytes2str
from . import decorators
from .classes import Options
from .classes import attributeStore as AttributeStore
from .classes import copy_into, get_option, set_option, typedList
from .decorators import fix_signature, lookup_loaders, make_Data, register_loader
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
    "between": lambda k, v: logical_and(min(v) < k, k < max(v)),
    "ibetween": lambda k, v: logical_and(min(v) <= k, k <= max(v)),
    "ilbetween": lambda k, v: logical_and(min(v) <= k, k < max(v)),
    "iubetween": lambda k, v: logical_and(min(v) < k, k <= max(v)),
    "startswith": lambda k, v: str(v).startswith(k),
    "istartswith": lambda k, v: str(v).upper().startswith(k.upper()),
    "endsswith": lambda k, v: str(v).endswith(k),
    "iendsswith": lambda k, v: str(v).upper().endswith(k.upper()),
}
