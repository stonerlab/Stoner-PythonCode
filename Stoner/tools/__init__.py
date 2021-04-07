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
from collections.abc import Iterable, MutableSequence
import inspect
from copy import deepcopy
from importlib import import_module

from numpy import log10, floor, logical_and, isnan, round, ndarray, dtype  # pylint: disable=redefined-builtin


from ..compat import bytes2str
from .classes import attributeStore as AttributeStore, typedList, Options, get_option, set_option, copy_into
from .tests import all_size, all_type, isanynone, isComparable, isiterable, isLikeList, isnone, isproperty, isTuple
from .formatting import format_error, format_val, quantize, html_escape, tex_escape, ordinal
from . import decorators
from .decorators import make_Data, fix_signature

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
