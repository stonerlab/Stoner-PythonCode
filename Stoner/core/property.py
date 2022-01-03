# -*- coding: utf-8 -*-
"""Provide Mxin classes for Properties for DataFile Objects."""

__all__ = ["DataFilePropertyMixin"]

import os
import copy
import pathlib
import urllib

import numpy as np
from numpy import ma
import pandas as pd

from ..tools import get_option, isiterable, isLikeList
from ..compat import classproperty, path_types, string_types

from .utils import copy_into
from ..tools.classes import subclasses
from ..tools.file import URL_SCHEMES
from .setas import Setas
from .columns import Column_Headers

try:
    from tabulate import tabulate

    tabulate.PRESERVE_WHITESPACE = True
except ImportError:
    tabulate = None


class DataFilePropertyMixin:

    """Provide the proerties for DataFile Like Objects."""

    _subclasses = None

    def __init__(self, *args, **kargs):
        """Set the attributes that nack the properties defined here."""
        self._data = pd.DataFrame()
        self._mask = pd.DataFrame()
        self._masks = [False]
        self._mask_value = np.NaN
        self._setas = Setas(self)
        self._column_headers = Column_Headers(self)
        super().__init__(*args, **kargs)

    @property
    def _repr_html_(self):
        """Generate an html representation of the DataFile.

        Raises:
            AttributeError:
                If short representation options are selcted, raise an AttributeError.

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
        return self._column_headers

    @column_headers.setter
    def column_headers(self, value):
        """Write the column_headers attribute (delagated to the setas object)."""
        if not isLikeList(value):
            raise ValueError(f"Value assigned to column headers must be a sequence not a {type(value)}")
        self._column_headers.set_all(value)

    @property
    def data(self):
        """Property Accessors for the main numerical data."""
        return self._data.mask(self._mask, self._mask_value)

    @data.setter
    def data(self, value):
        """Set the data attribute, but force it through numpy.ma.masked_array first."""
        if hasattr(value, "mask"):
            mask = value.mask
            value.mask = False
        else:
            mask = False
        if not isinstance(value, (pd.DataFrame, pd.Series)):
            column_headers = self.column_headers
        elif isinstance(value, pd.DataFrame):
            column_headers = [str(x) for x in value.columns]
        else:
            column_headers = [str(x) for x in value.index]
        data = pd.DataFrame(value)
        if isinstance(value, pd.Series) or (data.shape[1] == 1 and data.shape[0] > 1):
            data = data.T
        # Ensure column names are all strings.
        data.rename(columns={x: str(x) for x in data.columns}, inplace=True)
        if np.shape(mask) != data.shape:
            mask = np.zeros_like(data, dtype=bool)
        if list(data.columns) != list(column_headers):
            renames = {old: new for old, new in zip(data.columns, column_headers)}
            data = data.rename(columns=renames)
        self._data = data
        self._mask = pd.DataFrame(mask, columns=self._data.columns)
        newsetas = {ch: self._setas._index.get(ch, ".") for ch in self._data.columns}
        self._setas._index = pd.Series(newsetas)

    @property
    def dict_records(self):
        """Return the data as a dictionary of single columns with column headers for the keys."""
        return self.data.to_records()

    @property
    def dims(self):
        """Alias for self.data.axes."""
        return self.data.axes

    @property
    def dtypes(self):
        """Return the np dtype attribute of the data."""
        return self.data.dtypes

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
        return self._mask

    @mask.setter
    def mask(self, value):
        """Set the mask attribute by setting the data.mask."""
        if callable(value):
            return self._set_mask(value, invert=False)
        elif np.ndim(value) == 0:
            value = np.ones_like(self._data, dtype=bool) * value
        elif np.shape(value) == self._data.shape:
            value = value.astype(bool)
        elif np.ndim(value) == 2:
            value = np.array(value)
            r, c = value.shape
            x, y = self._data.shape
            r = min(r, x)
            c = min(c, y)
            mask = np.zeros_like(self._data)
            mask[:r, :c] = value
            value = mask
        else:
            raise ValueError(f"Unable to set mask from {value}")
        self._mask = pd.DataFrame(value, columns=self._data.columns)

    @classproperty
    def patterns(cls):  # pylint: disable=no-self-argument
        """Return the possible filename patterns for use in dialog boxes."""
        patterns = cls._patterns
        for cls_name, klass in subclasses().items():  # pylint: disable=not-an-iterable
            if cls_name == "DataFile" or "patterns" not in klass.__dict__:
                continue
            patterns.extend([p for p in klass.patterns if p not in patterns])
        return patterns

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
        return self._setas

    @setas.setter
    def setas(self, value):
        """Set a new setas assignment by calling the setas object."""
        self._setas(value)

    @property
    def T(self):
        """Get the current data transposed."""
        return self.data.T

    @T.setter
    def T(self, value):
        """Write directly to the transposed data."""
        self.data = value.T

    #################################################################################
    ############## Methods for manipulating properties

    def _set_mask(self, func, invert=False, cumulative=False, col=0):
        """Apply func to each row in self.data and uses the result to set the mask for the row.

        Args:
            func (callable):
                A Callable object of the form lambda x:True where x is a row of data (numpy
            invert (bool):
                Optionally invert te reult of the func test so that it unmasks data instead
            cumulative (bool):
                if tru, then an unmask value doesn't unmask the data, it just leaves it as it is.
        """
        args = len(_inspect_.getargs(func.__code__)[0])
        for i, r in self.data.iterrows():
            if args == 2:
                t = func(r[self.setas.x], r)
            else:
                t = func(r)
            if isinstance(t, (bool, np.bool_)) or (isiterable(t) and len(t) == self._data.shape[1]):
                t = t ^ invert
                if cummulative:
                    self._mask.iloc[i] = t
                else:
                    self.mask.iloc[i, t] = ~invert
            else:
                raise ValueError(
                    f"Mask function {func} should return either a single boolean value or an aray of the same length as the input not {t}"
                )

    def _push_mask(self, mask=None):
        """Copy the current data mask to a temporary store and replace it with a new mask if supplied.

        Args:
            mask (:py:class:numpy.array of bool or bool or None):
                The new data mask to apply (defaults to None = unmask the data

        Returns:
            Nothing
        """
        self._masks.append(self.mask)
        if mask is None:
            self.mask = False
        else:
            self.mask = mask

    def _pop_mask(self):
        """Replace the mask on the data with the last one stored by _push_mask().

        Returns:
            The previous mask
        """
        ret = self.mask
        self.mask = self._masks.pop()
        if not self._masks:
            self._masks = [False]
