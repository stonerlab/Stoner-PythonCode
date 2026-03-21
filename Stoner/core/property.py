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
from .setas import ColumnHeadersDescriptor, Setas

try:
    from tabulate import tabulate

    tabulate.PRESERVE_WHITESPACE = True
except ImportError:
    tabulate = None


class DataFilePropertyMixin:
    """Provide the properties for DataFile Like Objects."""

    _subclasses = None

    #: setas (:py:class:`Setas`): Descriptor that delegates column-type assignments to the
    #:   internal :py:class:`DataArray` (``_data``).  Getting or setting ``obj.setas`` is
    #:   equivalent to ``obj._data.setas``.
    setas = Setas(source="_data")

    #: column_headers (list): Descriptor that forwards to ``setas.column_headers`` via the
    #:   delegating :py:class:`Setas` descriptor above.
    column_headers = ColumnHeadersDescriptor()

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

    data = DataArray([])
    """DataArray descriptor that enforces the data attribute is always a :class:`DataArray` instance."""

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
    def T(self):
        """Get the current data transposed."""
        return self.data.T

    @T.setter
    def T(self, value):
        """Write directly to the transposed data."""
        self.data = value.T
