#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions to support :py:mod:`Stoner.Core`."""

__all__ = ["add_core", "and_core", "sub_core", "mod_core", "tab_delimited", "decode_string"]

import copy
import csv
import re
import sys
from collections.abc import Mapping
from typing import Callable, List
from typing import Mapping as MappingType
from typing import Union

import numpy as np
import pandas as pd

from ..compat import index_types, int_types
from ..tools.tests import all_type
from .Typing import Column_Index, Int_Types, Numeric


def add_core(other: Union["DataFile", np.ndarray, List[Numeric], MappingType], newdata: "DataFile") -> "DataFile":
    """Implement the core work of adding other to self and modifying newdata.

    Args:
        other (DataFile,array,list):
            The data to be added
        newdata(DataFile):
            The instance to be modified

    Returns:
        newdata:
            A modified newdata
    """
    if isinstance(other, (list, tuple)):  # Lists and tuples are converted to numpy arrays
        other = np.array(other)
    if isinstance(other, np.ndarray):  # Numpy array to DataFrtame
        if other.ndim == 1:  # 1D arrays are treated as rows
            other = pd.DataFrame(other).T
        elif other.ndim == 2:
            other = pd.DataFrame(other)
        else:
            raise ValueError(f"Cannot concatenat {other.ndim}D arrays to a DataFrame based DataFile")
        # Rename columns as best we can.
        other.rename(columns={c: k for c, k in zip(other.columns, newdata._data.columns)})
    if isinstance(other, pd.Series):
        other = pd.DataFrame(other).T

    if isinstance(other, getattr(sys.modules["Stoner.Core"], "DataFile")):
        other = other._data

    newdata.data = pd.concat([newdata, other], axis=0, ignore_index=False)
    return newdata


def and_core(other: Union["DataFile", np.ndarray], newdata: "DataFile") -> "DataFile":
    """Implement the core of the & operator, returning data in newdata.

    Args:
        other (array,DataFile):
            Data whose columns are to be added
        newdata (DataFile):
            instance of DataFile to be modified

    Returns:
        ():py:class:`DataFile`):
            new Data object with the columns of other concatenated as new columns at the end of the self object.
    """
    setas = None
    if isinstance(other, getattr(sys.modules["Stoner.Core"], "DataFile")):
        setas = [x for x in other.setas]
        other = other._data
    if isinstance(other, (np.ndarray, pd.Series)):
        other = pd.DataFrame(other)
    renames = dict()
    for x in other.columns:
        if x in newdata._data.columns:
            ix = 1
            trial = f"{x}_{ix}"
            while trial in newdata._data.columns:
                ix += 1
                trial = f"{x}_{ix}"
            renames[x] = trial
    if renames:
        other = other.rename(columns=renames)
    setas = ["."] * other.shape[1] if setas is None else setas
    setas = newdata.setas.to_list() + setas
    newdata.data = pd.concat([newdata._data, other], axis=1)
    newdata.setas = setas

    return newdata


def mod_core(other: Column_Index, newdata: "DataFile") -> "DataFile":
    """Implement the column deletion method."""
    if isinstance(other, index_types):
        newdata.del_column(other)
    else:
        newdata = NotImplemented
    return newdata


def sub_core(other: Union[Int_Types, slice, Callable], newdata: "DataFile") -> "DataFile":
    """Worker for the subtraction."""
    if isinstance(other, (slice, int_types)) or callable(other):
        newdata.del_rows(other)
    elif isinstance(other, list) and (all_type(other, int_types) or all_type(other, bool)):
        newdata.del_rows(other)
    else:
        newdata = NotImplemented
    newdata._data._setas.shape = newdata.shape
    return newdata


class tab_delimited(csv.Dialect):

    """A customised csv dialect class for reading tab delimited text files."""

    delimiter = "\t"
    quoting = csv.QUOTE_NONE
    doublequote = False
    lineterminator = "\r\n"


def decode_string(value: str) -> str:
    """Expand a string of column assignments, replacing numbers with repeated characters."""
    pattern = re.compile(r"(([0-9]+)(x|y|z|d|e|f|u|v|w|\.|\-))")
    while True:
        res = pattern.search(value)
        if res is None:
            break
        (total, count, code) = res.groups()
        count = int(count)
        value = value.replace(total, code * count, 1)
    return value


def unique_strings(strings: List[str]) -> List[str]:
    """Ensure that a list of string contains no duplicate values.

    Args:
        strings (list of str):
            The strings to check are unique

    Returns:
        (list of str):
            The same strings, but with a _# added to ensure each entry in sunuqye where # is a digit.
    """
    for ix, string in enumerate(strings):
        new = string
        i = 1
        while new in strings[:ix]:
            new = f"{string}_{i}"
            i += 1
        strings[ix] = new
    return strings
