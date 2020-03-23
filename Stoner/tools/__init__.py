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
    "isAnyNone",
    "isComparable",
    "isNone",
    "isIterable",
    "isLikeList",
    "isProperty",
    "isTuple",
    "quantize",
    "tex_escape",
    "typedList",
    "get_option",
    "set_option",
]
from collections.abc import Iterable, MutableSequence
import re
import inspect
from copy import deepcopy
import copy
from importlib import import_module

from numpy import log10, floor, abs, logical_and, isnan, round, ndarray, dtype  # pylint: disable=redefined-builtin

try:
    from memoization import cached
except ImportError:

    def cached(func, *_):
        """Null dectorator."""
        return func


from ..compat import string_types, bytes2str
from .classes import attributeStore as AttributeStore, typedList, Options as _Options, get_option, set_option
from .tests import all_size, all_type, isAnyNone, isComparable, isIterable, isLikeList, isNone, isProperty, isTuple
from .formatting import format_error, format_val, quantize

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


###############################################################################################################
######################  Functions #############################################################################
###############################################################################################################


@cached
def make_Data(*args, **kargs):
    """Return an instance of Stoner.Data passig through constructor arguments

    Calling make_Data(None) is a speical case to return the Data class ratther than an instance
    """
    if len(args) == 1 and args[0] is None:
        return import_module("Stoner.core.data").Data
    return import_module("Stoner.core.data").Data(*args, **kargs)


def fix_signature(proxy_func, wrapped_func):
    """Tries to update proxy_func to have a signature that matches the wrapped func."""
    try:
        proxy_func.__wrapped__.__signature__ = inspect.signature(wrapped_func)
    except AttributeError:  # Non-critical error
        try:
            proxy_func.__signature__ = inspect.signature(wrapped_func)
        except AttributeError:
            pass
    if hasattr(wrapped_func, "changes_size"):
        proxy_func.changes_size = wrapped_func.changes_size
    return proxy_func
