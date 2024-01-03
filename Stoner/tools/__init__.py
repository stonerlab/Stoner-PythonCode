#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Support functions for Stoner package.

Everything in the tools package now supports lazy imports
"""
import importlib

from numpy import logical_and

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
    "isclass",
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

_sub_imports = {
    "all_size": "tests",
    "all_type": "tests",
    "isanynone": "tests",
    "isComparable": "tests",
    "isiterable": "tests",
    "isLikeList": "tests",
    "isnone": "tests",
    "isproperty": "tests",
    "isTuple": "tests",
    "isclass": "tests",
    "bytes2str": ".compat",
    "AttributeStore": "classes",
    "typedList": "classes",
    "Options": "classes",
    "get_option": "classes",
    "set_option": "classes",
    "copy_into": "classes",
    "format_error": "formatting",
    "format_val": "formatting",
    "quantize": "formatting",
    "html_escape": "formatting",
    "tex_escape": "formatting",
    "ordinal": "formatting",
    "make_Data": "decorators",
    "fix_signature": "decorators",
}


def __getattr__(name):
    """Lazy import required module."""
    if name in __all__:
        if name in _sub_imports:
            ret = importlib.import_module(f".{_sub_imports[name]}", __name__)
            return getattr(ret, name)
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
