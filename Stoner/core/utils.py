#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions to support :py:mod:`Stoner.Core`."""

__all__ = ["add_core", "and_core", "sub_core", "mod_core", "copy_into", "tab_delimited", "decode_string"]

import copy
import csv
import re
from collections.abc import Mapping
from typing import Union, List, Mapping as MappingType, Callable
import numpy as np
from ..compat import index_types, int_types
from ..tools import all_type
from .Typing import Numeric, Column_Index, Int_Types


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
    if isinstance(other, np.ndarray):
        if len(newdata) == 0:  # pylint: disable=len-as-condition
            ch = getattr(other, "column_headers", [])
            setas = getattr(other, "setas", "")
            t = np.atleast_2d(other)
            c = t.shape[1]
            if len(newdata.column_headers) < c:
                newdata.column_headers.extend([f"Column_{x}" for x in range(c - len(newdata.column_headers))])
            newdata.data = t
            newdata.setas = setas
            newdata.column_headers = ch
            ret = newdata
        elif len(np.shape(other)) == 1:
            # 1D array, so assume a single row of data
            if np.shape(other)[0] == np.shape(newdata.data)[1]:
                newdata.data = np.append(newdata.data, np.atleast_2d(other), 0)
                ret = newdata
            else:
                return NotImplemented
        elif len(np.shape(other)) == 2 and np.shape(other)[1] == np.shape(newdata.data)[1]:
            # DataFile + array with correct number of columns
            newdata.data = np.append(newdata.data, other, 0)
            ret = newdata
        else:
            return NotImplemented
    elif isinstance(other, type(newdata)):  # Appending another DataFile
        new_data = np.ones((other.shape[0], newdata.shape[1])) * np.nan
        for i in range(newdata.shape[1]):
            column = newdata.column_headers[i]
            try:
                new_data[:, i] = other.column(column)
            except KeyError:
                pass
        newdata.metadata.update(other.metadata)
        newdata.data = np.append(newdata.data, new_data, axis=0)
        ret = newdata
    elif isinstance(other, list):
        for o in other:
            newdata = newdata + o
        ret = newdata
    elif isinstance(other, Mapping):
        # First check keys all in newdata
        if len(newdata) == 0:
            newdata.data = np.atleast_2d(list(other.values()))
            newdata.column_headers = list(other.keys())
        else:
            order = dict()
            for k in other:
                try:
                    order[k] = newdata.find_col(k)
                except (KeyError, re.error):
                    mask = newdata.mask
                    newdata.add_column(np.ones(len(newdata)) * np.NaN, header=k)
                    newdata.mask[:, :-1] = mask
                    newdata.mask[:, -1] = np.ones(len(newdata), dtype=bool)
                    order[k] = newdata.shape[1] - 1
            row = np.ones(newdata.shape[1]) * np.NaN
            mask = np.ones_like(row, dtype=bool)
            for k in order:
                row[order[k]] = other[k]
                mask[order[k]] = False
            old_mask = newdata.mask
            newdata.data = np.ma.append(newdata.data, np.atleast_2d(row), axis=0)
            newdata.mask[:-1, :] = old_mask
            newdata.mask[-1] = mask
        ret = newdata
    else:
        return NotImplemented
    ret._data._setas.shape = ret.shape
    for attr in newdata.__dict__:
        if attr not in ("setas", "metadata", "data", "column_headers", "mask") and not attr.startswith("_"):
            ret.__dict__[attr] = newdata.__dict__[attr]
    return ret


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
    if len(newdata.data.shape) < 2:
        newdata.data = np.atleast_2d(newdata.data)

    # Get other to be a numpy masked array of data
    # Get other_headers to be a suitable length list of strings
    if isinstance(other, type(newdata)):
        newdata.metadata.update(other.metadata)
        other_headers = other.column_headers
        other = copy.copy(other.data)
    elif isinstance(other, type(newdata.data)):
        other = copy.copy(other)
        if len(other.shape) < 2:  # 1D array, make it 2D column
            other = np.atleast_2d(other)
            other = other.T
        other_headers = [f"Column {i + newdata.shape[1]}" for i in range(other.shape[1])]
    elif isinstance(other, np.ndarray):
        other = type(newdata.data)(copy.copy(other))
        if len(other.shape) < 2:  # 1D array, make it 2D column
            other = np.atleast_2d(other)
            other = other.T
        other_headers = [f"Column {i + newdata.shape[1]}" for i in range(other.shape[1])]
    else:
        return NotImplemented

    newdata_headers = newdata.column_headers + other_headers
    setas = newdata.setas.clone

    # Workout whether to extend rows on one side or the other
    if np.product(newdata.data.shape) == 0:  # Special case no data yet
        newdata.data = other
    elif newdata.data.shape[0] == other.shape[0]:
        newdata.data = np.append(newdata.data, other, 1)
    elif newdata.data.shape[0] < other.shape[0]:  # Need to extend self.data
        extra_rows = other.shape[0] - newdata.data.shape[0]
        newdata.data = np.append(newdata.data, np.zeros((extra_rows, newdata.data.shape[1])), 0)
        new_mask = newdata.mask
        new_mask[-extra_rows:, :] = True
        newdata.data = np.append(newdata.data, other, 1)
        other_mask = np.ma.getmaskarray(other)
        new_mask = np.append(new_mask, other_mask, 1)
        newdata.mask = new_mask
    elif other.shape[0] < newdata.data.shape[0]:
        # too few rows we can extend with zeros
        extra_rows = newdata.data.shape[0] - other.shape[0]
        other = np.append(other, np.zeros((extra_rows, other.shape[1])), 0)
        other_mask = np.ma.getmaskarray(other)
        other_mask[-extra_rows:, :] = True
        new_mask = newdata.mask
        new_mask = np.append(new_mask, other_mask, 1)
        newdata.data = np.append(newdata.data, other, 1)
        newdata.mask = new_mask

    setas.column_headers = newdata_headers
    newdata._data._setas = setas
    newdata._data._setas.shape = newdata.shape
    for attr in newdata.__dict__:
        if attr not in ("setas", "metadata", "data", "column_headers", "mask") and not attr.startswith("_"):
            newdata.__dict__[attr] = newdata.__dict__[attr]
    return newdata


def mod_core(other: Column_Index, newdata: "DataFile") -> "DataFile":
    """Implement the column deletion method."""
    if isinstance(other, index_types):
        newdata.del_column(other)
    else:
        newdata = NotImplemented
    newdata._data._setas.shape = newdata.shape
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


def copy_into(source: "DataFile", dest: "DataFile") -> "DataFile":
    """Copy the data associated with source to dest.

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
        except (NotImplementedError, TypeError):  # Deepcopying failed, so just copy a reference instead
            setattr(dest, attr, getattr(source, attr))
    return dest


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
