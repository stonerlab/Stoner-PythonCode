#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions to support :py:mod:`Stoner.Core`."""

__all__ = ["copy_into", "itersubclasses", "tab_delimited", "decode_string"]

import copy
import csv
import re


def copy_into(source, dest):
    """Copies the data associated with source to dest.

    Args:
        source(DataFile): The DataFile object to be copied from
        dest (DataFile): The DataFile objrct to be changed by recieving the copiued data.

    Returns:
        The modified *dest* DataFile.

    Unlike copying or deepcopying a DataFile, this function preserves the class of the destination and just
    overwrites the attributes that represent the data in the DataFile.
    """
    dest.data = source.data.copy()
    dest.setas = source.setas
    for attr in source._public_attrs:
        if not hasattr(source, attr) or callable(getattr(source, attr)) or attr in ["data"]:
            continue
        try:
            setattr(dest, attr, copy.deepcopy(getattr(source, attr)))
        except NotImplementedError:  # Deepcopying failed, so just copy a reference instead
            setattr(dest, attr, getattr(source, attr))
    return dest


class tab_delimited(csv.Dialect):

    """A customised csv dialect class for reading tab delimited text files."""

    delimiter = "\t"
    quoting = csv.QUOTE_NONE
    doublequote = False
    lineterminator = "\r\n"


def itersubclasses(cls, _seen=None):
    """Iterate over subclasses of a given class

    Generator over all subclasses of a given class, in depth first order.

    >>> list(itersubclasses(int)) == [bool]
    True
    >>> class A(object): pass
    >>> class B(A): pass
    >>> class C(A): pass
    >>> class D(B,C): pass
    >>> class E(D): pass
    >>>
    >>> for cls in itersubclasses(A):
    ...     print(cls.__name__)
    B
    D
    E
    C
    >>> # get ALL (new-style) classes currently defined
    >>> [cls.__name__ for cls in itersubclasses(object)] #doctest: +ELLIPSIS
    ['type', ...'tuple', ...]
    """
    if not isinstance(cls, type):
        raise TypeError("itersubclasses must be called with " "new-style classes, not %.100r" % cls)
    if _seen is None:
        _seen = set()
    try:
        subs = cls.__subclasses__()
    except TypeError:  # fails only when cls is type
        subs = cls.__subclasses__(cls)
    for sub in subs:
        if sub not in _seen:
            _seen.add(sub)
            yield sub
            for sub in itersubclasses(sub, _seen):
                yield sub


def decode_string(value):
    """Expands a string of column assignments, replacing numbers with repeated characters."""
    pattern = re.compile(r"(([0-9]+)(x|y|z|d|e|f|u|v|w|\.|\-))")
    while True:
        res = pattern.search(value)
        if res is None:
            break
        (total, count, code) = res.groups()
        count = int(count)
        value = value.replace(total, code * count, 1)
    return value
