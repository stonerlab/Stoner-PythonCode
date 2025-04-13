# -*- coding: utf-8 -*-
"""Provide Mxin classes for Properties for DataFile Objects."""

__all__ = ["DataFilePropertyMixin"]

import copy
import os
import pathlib
import urllib

import numpy as np
from numpy import ma

from ..compat import path_types
from ..tools import get_option
from ..tools.classes import copy_into
from ..tools.file import URL_SCHEMES
from .array import DataArray

try:
    from tabulate import tabulate

    tabulate.PRESERVE_WHITESPACE = True
except ImportError:
    tabulate = None


class DataFilePropertyMixin:
    """Provide the properties for DataFile Like Objects."""

    _subclasses = None

    @property
    def _repr_html_(self):
        """Generate an html representation of the DataFile.

        Raises:
            AttributeError:
                If short representation options are selected, raise an AttributeError.

        Returns:
            str:
                Produce an HTML table from the Data object.
        """
        if get_option("short_repr") or get_option("short_data_repr"):
            raise AttributeError("Rich html output suppressed")
        return self._repr_html_private

    @property
    def basename(self):
        """Return the basename of the current filename."""
        try:
            return os.path.basename(self.filename)
        except TypeError:
            return ""

    @property
    def clone(self):
        """Get a deep copy of the current DataFile."""
        c = type(self)()
        if self.debug:
            print("Cloning in DataFile")
        return copy_into(self, c)

    @property
    def column_headers(self):
        """Pass through to the setas attribute."""
        return self.data._setas.column_headers

    @column_headers.setter
    def column_headers(self, value):
        """Write the column_headers attribute (delagated to the setas object)."""
        self.data._setas.column_headers = value

    @property
    def data(self):
        """Property Accessors for the main numerical data."""
        return np.atleast_2d(self._data)

    @data.setter
    def data(self, value):
        """Set the data attribute, but force it through numpy.ma.masked_array first."""
        nv = value
        if not nv.shape:  # nv is a scalar - make it a 2D array
            nv = ma.atleast_2d(nv)
        elif nv.ndim == 1:  # nv is a vector - make it a 2D array
            nv = ma.atleast_2d(nv).T
        elif nv.ndim > 2:  # nv has more than 2D - raise an error # TODO 0.9? Support 3D arrays in DataFile?
            raise ValueError(f"DataFile.data should be no more than 2 dimensional not shape {nv.shape}")
        if not isinstance(
            nv, DataArray
        ):  # nv isn't a DataArray, so preserve setas (does this preserve column_headers too?)
            nv = DataArray(nv)
            nv._setas = getattr(self, "_data")._setas.clone
        elif (
            nv.shape[1] == self.shape[1]
        ):  # nv is a DataArray with the same number of columns - preserve column_headers and setas
            ch = getattr(self, "_data").column_headers
            nv._setas = getattr(self, "_data")._setas.clone
            nv.column_headers = ch
        nv._setas.shape = nv.shape
        self._data = nv

    @property
    def dict_records(self):
        """Return the data as a dictionary of single columns with column headers for the keys."""
        return np.array([dict(zip(self.column_headers, r)) for r in self.rows()])

    @property
    def dims(self):
        """Alias for self.data.axes."""
        return self.data.axes

    @property
    def dtype(self):
        """Return the np dtype attribute of the data."""
        return self.data.dtype

    @property
    def filename(self):
        """Return DataFile filename, or make one up."""
        if self._filename is None:
            self.filename = "Untitled"
        if isinstance(self._filename, path_types):
            return str(self._filename)
        else:
            return self._filename

    @filename.setter
    def filename(self, filename):
        """Store the DataFile filename."""
        if isinstance(filename, path_types) and urllib.parse.urlparse(str(filename)) in URL_SCHEMES:
            self._filename = pathlib.Path(filename)
        else:
            self._filename = filename

    @property
    def filepath(self):
        """Return DataFile filename, or make one up, returning as a pathlib.Path."""
        if self._filename is None:
            self.filename = "Untitled"
        return pathlib.Path(self._filename)

    @filepath.setter
    def filepath(self, filename):
        """Store the DataFile filename."""
        self._filename = pathlib.Path(filename)

    @property
    def header(self):
        """Make a pretty header string that looks like the tabular representation."""
        if tabulate is None:
            raise ImportError("No tabulate.")
        fmt = "rst"
        lb = "<br/>" if fmt == "html" else "\n"
        rows, cols = self._repr_limits
        r, c = self.shape
        interesting, col_assignments, cols = self._interesting_cols(cols)
        c = min(c, cols)
        if r > rows:
            shorten = [True, False]
            r = rows + rows % 2
        else:
            shorten = [False, False]

        shorten[1] = c > cols
        r = max(len(self.metadata), r)

        outp = np.zeros((1, c + 1), dtype=object)
        outp[:, :] = "..."
        ch = [self.column_headers[ix] if ix >= 0 else "...." for ix in interesting]

        for ix, (h, i) in enumerate(zip(ch, col_assignments)):
            ch[ix] = f"{h}{lb}{i}"
        outp[0, 1:] = ch
        # outp[1,1:]=col_assignments
        outp[0, 0] = f"TDI Format 1.5{lb}index"
        ret = tabulate(outp, tablefmt=fmt, numalign="decimal", stralign="center")
        return ret

    @property
    def mask(self):
        """Return the mask of the data array."""
        self.data.mask = ma.getmaskarray(self.data)
        return self.data.mask

    @mask.setter
    def mask(self, value):
        """Set the mask attribute by setting the data.mask."""
        if callable(value):
            self._set_mask(value, invert=False)
        else:
            self.data.mask = value

    @property
    def records(self):
        """Return the data as a np structured data array.

        If columns names are duplicated then they are made unique."""
        ch = copy.copy(self.column_headers)  # renoved duplicated column headers for structured record
        ch_bak = copy.copy(ch)
        setas = self.setas.clone  # We'll need these later !
        f = self.data.flags
        if (
            not f["C_CONTIGUOUS"] and not f["F_CONTIGUOUS"]
        ):  # We need our data to be contiguous before we try a records view
            self.data = self.data.copy()
        for i, header in enumerate(ch):
            j = 0
            while ch[i] in ch[i + 1 :] or ch[i] in ch[0:i]:
                j = j + 1
                ch[i] = f"{header}_{j}"
        dtype = [(str(x), self.dtype) for x in ch]
        self.setas = setas
        self.column_headers = ch_bak
        try:
            return self.data.view(dtype=dtype).reshape(len(self))
        except TypeError as err:
            raise TypeError(f"Failed to get record view. Dtype was {dtype}") from err

    @property
    def shape(self):
        """Pass through the numpy shape attribute of the data."""
        return self.data.shape

    @property
    def setas(self):
        """Get the list of column assignments."""
        setas = self._data._setas
        return setas

    @setas.setter
    def setas(self, value):
        """Set a new setas assignment by calling the setas object."""
        self._data._setas(value)

    @property
    def T(self):
        """Get the current data transposed."""
        return self.data.T

    @T.setter
    def T(self, value):
        """Write directly to the transposed data."""
        self.data = value.T
