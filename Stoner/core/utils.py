#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions supporting :py:mod:`Stoner.core`."""

__all__ = ["add_core", "and_core", "sub_core", "mod_core", "copy_into", "Tab_Delimited", "decode_string"]

import csv
import re
from dataclasses import dataclass
from typing import Callable
from typing import Mapping as MappingType
from typing import Optional, Union

import numpy as np
from numpy import ma
from numpy.typing import NDArray

from ..compat import index_types
from ..tools import copy_into, isiterable
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
    working: Optional[NDArray] = None

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


def _assemble_x_data(xcol, datafile, return_data):
    """Asse,ble x data from a working set and datafile settings."""
    match xcol:
        case _ if isinstance(xcol, index_types):
            return_data.x = return_data.working[:, datafile.find_col(xcol)]
        case np.ndarray(ndim=1, size=len(return_data.working)):
            return_data.x = xcol
        case _ if isiterable(xcol):
            for ix, c in enumerate(xcol):
                if ix == 0:
                    return_data.x = return_data.working[:, datafile.find_col(c)]
                else:
                    return_data.x = np.column_stack((return_data.x, return_data.working[:, datafile.find_col(c)]))
        case _:
            raise TypeError("Unable to idneify x-data for fitting.")


def _assemble_col_data(col, datafile, return_data, attr):
    """Add one column to the y data."""
    match col:
        case _ if isinstance(col, index_types):
            ydat = return_data.working[:, datafile.find_col(col)]
        case np.ndarray() if col.ndim == 1 and col.size == len(return_data.working):
            ydat = col
        case _:
            raise TypeError(
                """Y-data for fitting not defined - should either be an index or a 1D numpy array of the same
                length as the dataset"""
            )
    if not isinstance(getattr(return_data, attr), np.ndarray):
        setattr(return_data, attr, np.atleast_2d(ydat))
    else:
        setattr(return_data, attr, np.vstack([getattr(return_data, attr), ydat]))


def _assemble_sigma(sigma_n, datafile, return_data, col, ix, kwargs):
    """Assemble the sigma data."""
    match sigma_n:
        case None:
            sdat = np.ones_like(return_data.y)
            kwargs["scale_covar"] = True
        case list() if len(sigma_n) == 0:
            sdat = np.ones_like(return_data.y)
            kwargs["scale_covar"] = True
        case list() if all(isinstance(s, index_types) for s in sigma_n) and len(sigma_n) == len(col):
            sdat = return_data.working[:, datafile.find_col(sigma_n[ix])]
        case _ if isinstance(sigma_n, index_types):
            sdat = return_data.working[:, datafile.find_col(sigma_n)]
        case float():
            sdat = np.ones_like(return_data.y) * sigma_n
        case np.ndarray() if sigma_n.size == return_data.y.size:
            sdat = sigma_n
        case np.ndarray() if sigma_n.ndims == 2 and sigma_n.shape[1] == len(col):
            sdat = sigma_n[:, ix]
        case _:
            raise TypeError("Unable to recognise the y-error data.")
    return sdat


def assemnle_data(datafile, **kwargs):
    """Marshall the data for doing a curve_fit or equivalent.

    Args:
        datafile (Data):
            Data object to work with if not being used as a bound method.
        xcol (index):
            Column with xdata in it
        col(index):
            Column with ydata in it
        sigma (index or array-like):
            column of y-errors or uncertainty values.
        bounds (callable):
            Used to select the data rows to fit

    Keyword Arguments:
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
    ycol = kwargs.get("ycol", None)
    kwargs.setdefault("xerr", kwargs.get("sigma_x"))
    kwargs.setdefault("yerr", kwargs.get("sigma"))
    kwargs.setdefault("zerr", kwargs.get("sigma_z"))
    if ycol is not None and isinstance(ycol[0], np.ndarray) and len(ycol[0]) == len(datafile):
        return_data.y = ycol[0]
        kwargs["ycol"] = None
        _ = datafile._col_args(scalar=False, **kwargs)
        ycol = []
    else:
        _ = datafile._col_args(scalar=False, **kwargs)
        ycol = _.ycol

    bounds = kwargs.pop("bounds", lambda x, y: True)
    return_data.working = datafile.search(_.xcol, bounds)
    return_data.working = ma.mask_rowcols(return_data.working, axis=0)
    return_data.working = return_data.working[~return_data.working.mask[:, 0]]
    # Now check for sigma_y and sigma_x and have them default to sigma (which in turn defaults to None)
    _assemble_x_data(_.xcol, datafile, return_data)
    for i in range(len(ycol)):
        for col, attr in zip(["ycol", "zcol", "ucol", "vcol", "wcol"], "yzuvw"):
            col = getattr(_, col)
            if isinstance(col, list) and len(col) > i:
                _assemble_col_data(col[i], datafile, return_data, attr)

        for isigma, sigma_n in enumerate([_.xerr, _.yerr, _.zerr]):
            sdat = _assemble_sigma(sigma_n, datafile, return_data, ycol, i, kwargs)
            match (i, isigma):
                case (0, 0):
                    return_data.d = np.atleast_2d(sdat)
                case (0, 1):
                    return_data.e = np.atleast_2d(sdat)
                case (0, 2):
                    return_data.f = np.atleast_2d(sdat)
                case (_, 0):
                    return_data.d = np.vstack([return_data.d, sdat])
                case (_, 1):
                    return_data.e = np.vstack([return_data.e, sdat])
                case (_, 2):
                    return_data.f = np.vstack([return_data.f, sdat])

    kwargs.setdefault("absolute_sigma", not kwargs.pop("scale_covar", return_data.e is not None))
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
    from . import storage_operations
    return storage_operations.append(newdata, other)


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
    from . import storage_operations
    return storage_operations.concatenate_columns(newdata, other)


def mod_core(other: Index, newdata: Data) -> Data:
    """Implement the column deletion method."""
    from . import storage_operations
    return storage_operations.del_column(newdata, other)


def sub_core(other: Union[int, slice, Callable], newdata: Data) -> Data:
    """Worker for the subtraction."""
    from . import storage_operations
    return storage_operations.del_rows(newdata, other)


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
        total, count, code = res.groups()
        count = int(count)
        value = value.replace(total, code * count, 1)
    return value
