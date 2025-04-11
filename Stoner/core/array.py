#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides the DataArray class.

A subclass of :py:class:`numpy.ma.MaskedArray` that knows that columns have  names.
"""
__all__ = ["DataArray"]

import copy

import numpy as np
import numpy.ma as ma
from Stoner.compat import int_types
from Stoner.tools import AttributeStore, all_size, all_type, isiterable, isnone

from .exceptions import StonerSetasError
from .setas import setas as _setas


class DataArray(ma.MaskedArray):
    r"""A sub class of :py:class:`numpy.ma.MaskedArray` with a copy of the setas attribute to allow indexing by name.

    Attributes:
        column_headers (list):
            of strings of the column names of the data.
        i (array of integers):
            When read, returns the row  umbers of the data. When written to, sets the base row index. The base row
            index is preserved when a DataArray is indexed.
        x,y,z (1D DataArray):
            When a column is declared to contain *x*, *y*, or *z* data, then these attributes access the corresponding
            columns. When written to, the attributes overwrite the existing column's data.
        d,e,f (1D DataArray):
            Where a column is identified as containing uncertainties for *x*, *y* or *z* data, then these attributes
            provide a quick access to them. When written to, the attributes overwrite the existing column's data.
        u,v,w (1D DataArray):
            Columns may be identieid as containing vectgor field information. These attributes provide quick
            access to them, assuming that they are defined as cartesian coordinates. When written to, the attributes
            overwrite the existing column's data.
        p,q,r (1D DataArray):
            These attributes access calculated columns that convert :math:`(x,y,z)` data or :math:`(u,v,w)`
            into :math:`(\phi,\theta,r)` polar coordinates. If on *x* and *y* columns are defined, then 2D polar
            coordinates are returned for *q* and *r*.
        setas (list or string):
            Actually a proxy to a magic class that handles the assignment of columns to different axes and
            also tracks the names of columns (so that columns may be accessed as named items).

    This array type is used to represent numeric data in the Stoner Package - primarily as a 2D
    matrix in :py:class:`Stoner.Core.DataFile` but also when a 1D row is required. In con trast to
    the parent class, DataArray understands that it came from a DataFile which has a setas attribute and column
    assignments. This allows the row to be indexed by column name, and also for quick
    attribute access to work. This makes writing functions to work with a single row of data
    more attractive.
    """

    # ==============================================================================================================
    ############################           Object Construction                       ###############################
    # ==============================================================================================================

    def __new__(cls, input_array, *args, **kargs):
        """Create the new instance of the DataArray."""
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        setas = kargs.pop("setas", _setas())
        if isinstance(input_array, ma.MaskedArray):
            default_mask = input_array.mask
        else:
            default_mask = None
        mask = np.copy(kargs.pop("mask", default_mask))
        column_headers = kargs.pop("column_headers", [])
        _row = kargs.pop("isrow", False)
        if isinstance(input_array, DataArray):
            i = input_array.i
        else:
            i = 0
        obj = ma.asarray(input_array, *args, **kargs).view(cls)
        # add the new attribute to the created instance
        setas.shape = obj.shape
        obj._setas = setas
        if mask is not None:
            obj.mask = mask
        else:
            obj.mask = False
        # Finally, we must return the newly created object:
        obj.i = i
        obj.setas._row = _row and obj.ndim == 1
        # Set shared mask - stops some deprication warnings
        obj.unshare_mask()
        if np.issubdtype(obj.dtype, np.floating):
            obj.fill_value = np.nan
        obj.column_headers = column_headers
        return obj

    def __array_finalize__(self, obj):
        """Numpy ndarray magic method."""
        # see InfoArray.__array_finalize__ for comments
        super().__array_finalize__(obj)
        if obj is None:
            self._setas = _setas()
            self.i = 0
            self.mask = False
            if np.issubdtype(self.dtype, np.floating):
                self.fill_value = np.nan
            self._setas._row = False
            self._setas.shape = (0,)
        else:
            self._setas = getattr(obj, "_setas", _setas())
            if isinstance(obj, DataArray):
                self.i = obj.i
                self.mask = obj.mask
                self.fill_value = obj.fill_value
                self._setas._row = getattr(obj._setas, "_row", False)
            else:
                self.i = 0
                self.mask = False
                self._setas._row = False
                if np.issubdtype(self.dtype, np.floating):
                    self.fill_value = np.nan
            self._setas.shape = getattr(self, "shape", (0,))

    def __array_wrap__(self, out_arr, context=None, return_scalar=None):
        """Make sure ufuncs do the right thing with DataArrays."""
        ret = ma.MaskedArray.__array_wrap__(self, out_arr, context=context, return_scalar=return_scalar)
        return ret

    def _prepare_index(self, ix):
        """Mangle an indexing argument into a standard tuple form."""
        match ix:
            case str():  # Index by string - look for columns
                if self.ndim > 1:
                    ix = (slice(None, None, None), self._setas.find_col(ix))
                else:
                    ix = (self._setas.find_col(ix),)
            case int() | slice():  # index by integer or slice on rows
                ix = (ix,)
            case (*i, str()):  # index by something and then string
                ix = (*i, self._setas.find_col(ix[-1]))
            case (*i, np.ndarray()) if self.ndim == 1:
                if len(ix) == 1:
                    ix = ix[0]
            case (*i, np.ndarray()) if len(ix[1]) == self.shape[1] and ix[1].dtype.kind == "b":
                pass
            case (*i, s) if isiterable(s):  # index by whatever and thena string
                ix = (*i, [self._setas.find_col(c) for c in s])
            case (str(), *i):
                ix = (*i, self._setas.find_col(ix[0]))
        return ix

    # ==============================================================================================================
    ############################          Property Accessor Functions                ###############################
    # ==============================================================================================================

    @property
    def _(self):
        """Return the DataArray as a normal numpy array for those operations that need this."""
        return ma.getdata(self)

    @property
    def isrow(self):
        """Define whether this is a single row or a column if 1D."""
        return self._setas._row

    @isrow.setter
    def isrow(self, value):
        """Set whether this object is a single row or not."""
        self._setas._row = self.ndim == 1 and value

    @property
    def r(self):
        r"""Calculate the radius :math:`\rho` coordinate if using spherical or polar coordinate systems."""
        axes = int(self._setas.cols["axes"])
        m = [
            lambda d: None,
            lambda d: None,
            lambda d: np.sqrt(d.x**2 + d.y**2),
            lambda d: np.sqrt(d.x**2 + d.y**2 + d.z**2),
            lambda d: np.sqrt(d.x**2 + d.y**2 + d.z**2),
            lambda d: np.sqrt(d.u**2 + d.v**2),
            lambda d: np.sqrt(d.u**2 + d.v**2 + d.w**2),
        ]
        ret = m[axes](self)
        if ret is None:
            raise StonerSetasError(
                f"Insufficient axes defined in setas to calculate the r component. need 2 not {axes}"
            )
        else:
            return ret

    @property
    def q(self):
        r"""Calculate the azimuthal :math:`\theta` coordinate if using spherical or polar coordinates."""
        axes = int(self._setas.cols["axes"])
        m = [
            lambda d: None,
            lambda d: None,
            lambda d: np.arctan2(d.x, d.y),
            lambda d: np.arctan2(d.x, d.y),
            lambda d: np.arctan2(d.x, d.y),
            lambda d: np.arctan2(d.u, d.v),
            lambda d: np.arctan2(d.u, d.v),
        ]
        ret = m[axes](self)
        if ret is None:
            raise StonerSetasError(
                f"Insufficient axes defined in setas to calculate the theta component. need 2 not {axes}"
            )
        else:
            return ret

    @property
    def p(self):
        r"""Calculate the inclination :math:`\phi` coordinate for spherical coordinate systems."""
        axes = int(self._setas.cols["axes"])
        m = [
            lambda d: None,
            lambda d: None,
            lambda d: None,
            lambda d: np.arcsin(d.z),
            lambda d: np.arsin(d.z),
            lambda d: np.arcsin(d.w),
            lambda d: np.arcsin(d.w),
        ]
        ret = m[axes](self)
        if ret is None:
            raise StonerSetasError(
                f"Insufficient axes defined in setas to calculate the phi component. need 3 not {axes}"
            )
        return ret

    @property
    def i(self):
        """Return the row indices of the DataArray or sets the base index - the row number of the first row."""
        if not hasattr(self, "_ibase"):
            self._ibase = []
        if len(self._ibase) == 1 and self.isrow:
            ret = min(self._ibase)
        else:
            ret = self._ibase
        return ret

    @i.setter
    def i(self, value):
        if self.ndim == 0:
            pass
        elif self.ndim == 1 and self.isrow:
            if isiterable(value) and value:
                self._ibase = np.array([min(value)])
            else:
                self._ibase = np.array([value])
        elif self.ndim >= 1:
            r = self.shape[0]
            if isiterable(value) and len(value) == r:  # Iterable and the correct length - assign straight
                self._ibase = np.array(value)
            elif isiterable(value) and len(value) > 0:  # Iterable but not the correct length - count from min of value
                self._ibase = np.arange(min(value), min(value) + r)
            elif (
                isiterable(value) and len(value) == 0
            ):  # Iterable but not the correct length - count from min of value
                self._ibase = np.arange(0, r, r)
            else:  # No iterable
                self._ibase = np.arange(value, value + r)

    @property
    def column_headers(self):
        """Pass through to the setas attribute."""
        return self._setas.column_headers

    @column_headers.setter
    def column_headers(self, value):
        """Write the column_headers attribute (delagated to the setas object)."""
        self._setas.column_headers = value

    @property
    def setas(self):
        """Return an object for setting column assignments."""
        if self._setas is None:
            self._setas = _setas()
        if self._setas.shape != self.shape:
            self._setas.shape = self.shape
        return self._setas

    @setas.setter
    def setas(self, value):
        """Set the object for setting column assignments."""
        if isinstance(value, _setas):
            value = value.clone
        setas = self.setas
        setas(value)

    # ==============================================================================================================
    ############################        Special Methods         ####################################################
    # ==============================================================================================================

    def __reduce__(self):
        """Implement hooks for pickling."""
        # Get the parent's __reduce__ tuple
        pickled_state = super().__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self._setas, self.i)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        """Implement hooks for unpickling."""
        self._setas = state[-2]  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super().__setstate__(state[0:-2])
        self.i = state[-1]

    def __getattr__(self, name):
        """Get a column using the setas attribute."""
        # Overrides __getattr__ to allow access as row.x etc.
        col_check = {
            "x": "xcol",
            "d": "xerr",
            "y": "ycol",
            "e": "yerr",
            "z": "zcol",
            "f": "zerr",
            "u": "ucol",
            "v": "vcol",
            "w": "wcol",
        }
        if name in self.setas.cols:
            return self.setas.__getattr__(name)
        if name not in col_check:
            return super().__getattribute__(name)
        indexer = [slice(0, dim, 1) for ix, dim in enumerate(self.shape)]
        col = col_check[name]
        if col.startswith("x"):
            if self._setas.cols[col] is not None:
                indexer[-1] = self._setas.cols[col]
                ret = self[tuple(indexer)]
                if ret.ndim > 0:
                    ret.column_headers = self.column_headers[self._setas.cols[col]]
            else:
                ret = None
        else:
            if isiterable(self._setas.cols[col]) and len(self._setas.cols[col]) > 0:
                indexer[-1] = self._setas.cols[col][0]
            elif isiterable(self._setas.cols[col]):
                indexer[-1] = self._setas.cols[col]
            else:
                return None
            ret = self[tuple(indexer)]
            if ret.ndim > 0:
                ret.column_headers = self.column_headers[indexer[-1]]
        if ret is None:
            raise StonerSetasError(f"Tried accessing a {name} column, but setas is not defined.")
        return ret

    def __getitem__(self, ix):
        """Indexing function for DataArray.

        Args:
            ix (various):
                Index to find.

        Returns:
            An indexed part of the DataArray object with extra attributes.

        Notes:
            This tries to support all of the indexing operations of a regular numpy array,
            plus the special operations where one columns are named.

        Warning:
            The code almost certainly makes some assumptiuons that DataArray is one or 2D and
            may blow up with 3D arrays ! On the other hand it has a special case exception for where
            you give a string as the first index element and assumes that you've forgotten that we're
            row major and tries to do the right thing.
        """
        # Is this going to be a single row ?
        single_row = isinstance(ix, int_types) or (
            isinstance(ix, tuple) and len(ix) > 0 and isinstance(ix[0], int_types)
        )
        idx = self._prepare_index(ix)
        ret = super().__getitem__(idx)
        match ret:
            case np.ndarray(size=1) if ret.ndim > 0:
                return ret.ravel()[0]
            case ma.core.MaskedConstant() | np.ndarray(ndim=0):
                if ret.mask:
                    return self.fill_value
                return ret.dtype.type(ret)
            case ma.MaskedArray() if ret.ndim == 0:
                return ret.dtype.type(ma.filled(ret))
            case np.ndarray() if (
                isinstance(idx, tuple) and len(idx) == 0
            ):  # special case for indexing with empty tuple.
                return ret
            case np.ndarray(ndim=1) if single_row:
                ret.isrow = single_row
                ret.setas = self.setas.clone
                ret.column_headers = copy.copy(self.column_headers)
                if isinstance(self.i, np.ndarray):
                    ret.i = self.i[idx[0]]
                else:
                    ret.i = self.i
                return ret
            case np.ndarray(ndim=1):
                ret.isrow = single_row
                if isinstance(idx, tuple) and len(idx) >= 2:
                    tmp = np.array(self.setas)[idx[-1]].ravel()
                    ret.setas(tmp)
                    tmpcol = np.array(self.column_headers)[idx[-1]]
                    ret.column_headers = tmpcol
                ret.name = self.column_headers
                return ret
            case np.ndarray() if ret.ndim == 2:

                if idx[-1] is None:  # Special case for increasing an array dimension
                    if self.ndim == 1:  # Going from 1 D to 2D
                        ret.setas = self.setas.clone
                        ret.i = self.i
                        ret.name = getattr(self, "name", "Column")
                    return ret
                ret.isrow = single_row
                ret.setas = self.setas.clone
                ret.column_headers = copy.copy(self.column_headers)
                if len(idx) > 0 and isiterable(idx[-1]):  # pylint: disable=len-as-condition
                    ret.column_headers = list(np.array(ret.column_headers)[idx[-1]])
                # Sort out whether we need an array of row labels
                if isinstance(self.i, np.ndarray) and len(idx) > 0:  # pylint: disable=len-as-condition
                    if isiterable(idx[0]) or isinstance(idx[0], int_types):
                        ret.i = self.i[idx[0]]
                    else:
                        ret.i = 0
                else:
                    ret.i = self.i
                return ret
            case _:
                return ret

    def __setitem__(self, ix, val):
        """Override __setitem__ to handle string indexing."""
        match ix:
            case str():
                ix = self._setas.find_col(ix)
            case (*i, str()):
                ix = (*i, self._setas.find_col(ix[-1]))
            case (str(), *i):
                ix = (*i, self._setas.find_col(ix[0]))
            case _:
                pass

        if self.sharedmask:  # We do not want to share a mask when we're about to change soimething here...
            self.unshare_mask()

        super().__setitem__(ix, val)

    # ==============================================================================================================
    ############################              Private Methods                #######################################
    # ==============================================================================================================

    def _col_args(
        self,
        scalar=True,
        xcol=None,
        ycol=None,
        zcol=None,
        ucol=None,
        vcol=None,
        wcol=None,
        xerr=None,
        yerr=None,
        zerr=None,
        **kargs,
    ):  # pylint: disable=unused-argument
        """Create an object which has keys  based either on arguments or setas attribute."""
        cols = {
            "xcol": xcol,
            "ycol": ycol,
            "zcol": zcol,
            "ucol": ucol,
            "vcol": vcol,
            "wcol": wcol,
            "xerr": xerr,
            "yerr": yerr,
            "zerr": zerr,
        }
        no_guess = kargs.get("no_guess", True)
        for i in cols.values():
            if i is not None:  # User specification wins out
                break
        else:  # User didn't set any values, setas will win
            no_guess = kargs.get("no_guess", False)
        ret = AttributeStore(self.setas._get_cols(no_guess=no_guess))
        force_list = kargs.get("force_list", not scalar)
        for c in list(cols.keys()):
            if isnone(cols[c]):  # Not defined, fallback on setas
                del cols[c]
                continue
            if isinstance(cols[c], bool) and not cols[c]:  # False, delete column altogether
                del cols[c]
                if c in ret:
                    del ret[c]
                continue
            if c in ret and isinstance(ret[c], list):
                if isinstance(cols[c], float) or (isinstance(cols[c], np.ndarray) and cols[c].size == len(self)):
                    continue
            if isinstance(cols[c], float):
                continue
            cols[c] = self.setas.find_col(cols[c], force_list=force_list)
        ret.update(cols)
        if scalar:
            for c in ret:
                if isinstance(ret[c], list):
                    if ret[c]:
                        ret[c] = ret[c][0]
                    else:
                        ret[c] = None
        elif isinstance(scalar, bool) and not scalar:
            for c in ret:
                if c.startswith("x") or c.startswith("has_"):
                    continue
                if not isiterable(ret[c]) and ret[c] is not None:
                    ret[c] = list([ret[c]])
                elif ret[c] is None:
                    ret[c] = []
        for n in ["xcol", "xerr", "ycol", "yerr", "zcol", "zerr", "ucol", "vcol", "wcol", "axes"]:
            ret[f"has_{n}"] = n in ret and not (ret[n] is None or (isinstance(ret[n], list) and not ret[n]))

        return ret

    # ==============================================================================================================
    ############################              Public Methods                ########################################
    # ==============================================================================================================

    def keys(self):
        """Return a list of column headers."""
        return self._setas.column_headers

    def swap_column(self, *swp, **kargs):
        """Swap pairs of columns in the data.

        Useful for reordering data for idiot programs that expect columns in a fixed order.

        Args:
            swp  (tuple of list of tuples of two elements):
                Each element will be iused as a column index (using the normal rules
                for matching columns).  The two elements represent the two
                columns that are to be swapped.
            headers_too (bool):
                Indicates the column headers are swapped as well

        Returns:
            self:
                A copy of the modified :py:class:`DataFile` objects

        Note:
            If swp is a list, then the function is called recursively on each
            element of the list. Thus in principle the @swp could contain
            lists of lists of tuples
        """
        headers_too = kargs.pop("headers_too", True)
        setas_too = kargs.pop("setas_too", True)

        if len(swp) == 1:
            swp = swp[0]
        if isinstance(swp, list) and all_type(swp, tuple) and all_size(swp, 2):
            for item in swp:
                self.swap_column(item, headers_too=headers_too)
        elif isinstance(swp, tuple):
            col1 = self._setas.find_col(swp[0])
            col2 = self._setas.find_col(swp[1])
            self[:, [col1, col2]] = self[:, [col2, col1]]
            if headers_too:
                self._setas.column_headers[col1], self._setas.column_headers[col2] = (
                    self._setas.column_headers[col2],
                    self._setas.column_headers[col1],
                )
            if setas_too:
                self._setas[col1], self._setas[col2] = self._setas[col2], self._setas[col1]
        else:
            raise TypeError(
                "Swap parameter must be either a tuple or a \
            list of tuples"
            )
