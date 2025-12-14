#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions to support :py:mod:`Stoner.Core`."""

__all__ = ["add_core", "and_core", "sub_core", "mod_core", "copy_into", "Tab_Delimited", "decode_string"]

import copy
import csv
from dataclasses import dataclass
import re
from collections.abc import Mapping
from typing import Callable
from typing import Mapping as MappingType, Union, Optional

import numpy as np
from numpy.typing import NDArray
from numpy import ma

from ..compat import index_types, int_types
from ..tools import all_type, copy_into, isiterable
from ..tools.typing import Data, Index, NumericArray


@dataclass
class _WorkingData:

    x: Optional[NDArray] = None
    y: Optional[NDArray] = None
    z: Optional[NDArray] = None
    d: Optional[NDArray] = None
    e: Optional[NDArray] = None
    f: Optional[NDArray] = None
    u: Optional[NDArray] = None
    v: Optional[NDArray] = None
    w: Optional[NDArray] = None

    def __getitem__(self, index):
        """Use either integer or string mapping to get item."""
        mapping = "xyedzfuvw"
        match index:
            case int() if -len(mapping) <= index < len(mapping):
                return getattr(self, mapping[index])
            case str() if len(index) == 1 and index in "xyzdefuvw":
                return getattr(self, index)
            case slice():
                return [self[ix] for ix in range(*index.indices(len(mapping)))]
            case _:
                raise IndexError(f"{index} is out of range for WorkingData.")

    def __setitem__(self, index, value):
        """Use either integer or string to set items."""
        mapping = "xyedzfuvw"
        match index:
            case int() if -len(mapping) <= index < len(mapping):
                setattr(self, mapping[index], value)
            case str() if len(index) == 1 and index in "xyzdefuvw":
                setattr(self, index, value)
            case _:
                raise IndexError(f"{index} is out of range for WorkingData.")


def assemnle_data(datafile, xcol, ycol, sigma, sigma_x=None, **kwargs):
    """Marshall the data for doing a curve_fit or equivalent.

    Args:
        datafile (Data):
            Data object to work with if not being used as a bound method.
        xcol (index):
            Column with xdata in it
        ycol(index):
            Column with ydata in it
        sigma (index or array-like):
            column of y-errors or uncertainty values.
        bounds (callable):
            Used to select the data rows to fit

    Keyword Args:
        sigma_x (index or array-like):
            column of x-errors or uncertainty values.
        kwargs:
            Other keyword arguments to set the scale_covar/absolute_sigma.

    Returns:
        (data,kwargs,col_assignments):
            data is a tuple of (x,y,sigma). scale_covar is False if sigma is real errors.
    """
    # Special case for doing a fit where we're matching a function to a value not in data.
    return_data = _WorkingData()
    if ycol is not None and isinstance(ycol[0], np.ndarray) and len(ycol[0]) == len(datafile):
        return_data.y = ycol[0]
        _ = datafile._col_args(scalar=False, xcol=xcol, ycol=None, yerr=sigma, xerr=sigma_x)
        xcol, ycol, sigma, sigma_x = _.xcol, [], _.yerr, _.xerr
    else:
        _ = datafile._col_args(scalar=False, xcol=xcol, ycol=ycol, yerr=sigma, xerr=sigma_x)
        xcol, ycol, sigma, sigma_x = _.xcol, _.ycol, _.yerr, _.xerr

    bounds = kwargs.pop("bounds", lambda x, y: True)
    working = datafile.search(_.xcol, bounds)
    working = ma.mask_rowcols(working, axis=0)
    working = working[~working.mask[:, 0]]
    # Now check for sigma_y and sigma_x and have them default to sigma (which in turn defaults to None)
    match xcol:
        case _ if isinstance(xcol, index_types):
            return_data.x = working[:, datafile.find_col(xcol)]
        case np.ndarray(ndim=1, size=len(working)):
            return_data.x = xcol
        case _ if isiterable(xcol):
            for ix, c in enumerate(xcol):
                if ix == 0:
                    return_data.x = working[:, datafile.find_col(c)]
                else:
                    return_data.x = np.column_stack((return_data.x, working[:, datafile.find_col(c)]))
        case _:
            raise TypeError("Unable to idneify x-data for fitting.")
    for i, yc in enumerate(ycol):
        match yc:
            case _ if isinstance(yc, index_types):
                ydat = working[:, datafile.find_col(yc)]
            case np.ndarray() if yc.ndim == 1 and yc.size == len(working):
                ydat = yc
            case _:
                raise TypeError(
                    """Y-data for fitting not defined - should either be an index or a 1D numpy array of the same
                    length as the dataset"""
                )
        if i == 0:
            return_data.y = np.atleast_2d(ydat)
        else:
            return_data.y = np.vstack([return_data.y, ydat])
        for isigma, sigma_n in enumerate([sigma, sigma_x]):
            match sigma_n:
                case None:
                    sdat = np.ones_like(ydat)
                    kwargs["scale_covar"] = True
                case list() if len(sigma_n) == 0:
                    sdat = np.ones_like(ydat)
                    kwargs["scale_covar"] = True
                case list() if all(isinstance(s, index_types) for s in sigma_n) and len(sigma_n) == len(ycol):
                    sdat = working[:, datafile.find_col(sigma_n[i])]
                case _ if isinstance(sigma_n, index_types):
                    sdat = working[:, datafile.find_col(sigma_n)]
                case float():
                    sdat = np.ones_like(ydata) * sigma_n
                case np.ndarray() if sigma_n.size == ydat:
                    sdat = sigma_n
                case np.ndarray() if sigma_n.ndims == 2 and sigma_n.shape[1] == len(ycol):
                    sdat = sigma_n[:, i]
                case _:
                    raise TypeError("Unable to recognise the y-error data.")
            match (i, isigma):
                case (0, 0):
                    return_data.e = np.atleast_2d(sdat)
                case (0, 1):
                    return_data.d = np.atleast_2d(sdat)
                case (_, 0):
                    return_data.e = np.vstack([return_data.e, sdat])
                case (_, 1):
                    return_data.d = np.vstack([return_data.d, sdat])

    kwargs.setdefault("absolute_sigma", not kwargs.pop("scale_covar", sigma is not None))
    return return_data, kwargs, _


def add_core(other: Union[Data, NumericArray, MappingType], newdata: Data) -> Data:
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
    match other:
        case np.ndarray() if len(newdata) == 0:  # pylint: disable=len-as-condition
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
        case np.ndarray() if len(np.shape(other)) == 1:
            # 1D array, so assume a single row of data
            if np.shape(other)[0] == np.shape(newdata.data)[1]:
                newdata.data = np.append(newdata.data, np.atleast_2d(other), 0)
                ret = newdata
            else:
                return NotImplemented
        case np.ndarray() if len(np.shape(other)) == 2 and np.shape(other)[1] == np.shape(newdata.data)[1]:
            # DataFile + array with correct number of columns
            newdata.data = np.append(newdata.data, other, 0)
            ret = newdata
        case _ if isinstance(other, type(newdata)):  # Appending another DataFile
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
        case list():
            for o in other:
                newdata = newdata + o
            ret = newdata
        case _ if isinstance(other, Mapping):
            # First check keys all in newdata
            if len(newdata) == 0:
                newdata.data = np.atleast_2d(list(other.values()))
                newdata.column_headers = list(other.keys())
            else:
                order = {}
                for k in other:
                    try:
                        order[k] = newdata.find_col(k)
                    except (KeyError, re.error):
                        mask = newdata.mask
                        newdata.add_column(np.ones(len(newdata)) * np.nan, header=k)
                        newdata.mask[:, :-1] = mask
                        newdata.mask[:, -1] = np.ones(len(newdata), dtype=bool)
                        order[k] = newdata.shape[1] - 1
                row = np.ones(newdata.shape[1]) * np.nan
                mask = np.ones_like(row, dtype=bool)
                for k, val in order.items():
                    row[order[k]] = other[k]
                    mask[val] = False
                old_mask = newdata.mask
                newdata.data = np.ma.append(newdata.data, np.atleast_2d(row), axis=0)
                newdata.mask[:-1, :] = old_mask
                newdata.mask[-1] = mask
            ret = newdata
        case _:
            return NotImplemented
    ret._data._setas.shape = ret.shape
    for attr in newdata.__dict__:
        if attr not in ("setas", "metadata", "data", "column_headers", "mask") and not attr.startswith("_"):
            ret.__dict__[attr] = newdata.__dict__[attr]
    return ret


def and_core(other: Union[Data, NumericArray], newdata: Data) -> Data:
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
        if other.ndim < 2:  # 1D array, make it 2D column
            other = np.atleast_2d(other)
            other = other.T
        other_headers = [f"Column {i + newdata.shape[1]}" for i in range(other.shape[1])]
    elif isinstance(other, np.ndarray):
        other = type(newdata.data)(copy.copy(other))
        if other.ndim < 2:  # 1D array, make it 2D column
            other = np.atleast_2d(other)
            other = other.T
        other_headers = [f"Column {i + newdata.shape[1]}" for i in range(other.shape[1])]
    else:
        return NotImplemented

    newdata_headers = newdata.column_headers + other_headers
    setas = newdata.setas.clone

    # Workout whether to extend rows on one side or the other
    if np.prod(newdata.data.shape) == 0:  # Special case no data yet
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


def mod_core(other: Index, newdata: Data) -> Data:
    """Implement the column deletion method."""
    if isinstance(other, index_types):
        newdata.del_column(other)
    else:
        newdata = NotImplemented
    newdata._data._setas.shape = newdata.shape
    return newdata


def sub_core(other: Union[int, slice, Callable], newdata: Data) -> Data:
    """Worker for the subtraction."""
    if isinstance(other, (slice, int_types)) or callable(other):
        newdata.del_rows(other)
    elif isinstance(other, list) and (all_type(other, int_types) or all_type(other, bool)):
        newdata.del_rows(other)
    else:
        newdata = NotImplemented
    newdata._data._setas.shape = newdata.shape
    return newdata


class Tab_Delimited(csv.Dialect):
    """A customised csv dialect class for reading tab delimited text files."""

    delimiter = "\t"
    quoting = csv.QUOTE_NONE
    doublequote = False
    lineterminator = "\r\n"


def decode_string(value: str) -> str:
    """Expand a string of column assignments, replacing numbers with repeated characters."""
    pattern = re.compile(r"(([0-9]+)(x|y|z|d|e|f|u|v|w|\.|\-))")
    while res := pattern.search(value):
        (total, count, code) = res.groups()
        count = int(count)
        value = value.replace(total, code * count, 1)
    return value
