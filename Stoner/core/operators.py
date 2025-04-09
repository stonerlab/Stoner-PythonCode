# -*- coding: utf-8 -*-
"""Provide the mixin classes for the DataFile Operators."""

__all__ = ["DataFileOperatorsMixin"]
import numpy as np

from ..compat import index_types, string_types
from ..tools import isiterable
from . import DataArray, _setas
from .utils import add_core, and_core, mod_core, sub_core


class DataFileOperatorsMixin:
    """Provides the operator mixins for DataFile like objects."""

    def __add__(self, other):
        """Implement a + operator to concatenate rows of data.

        Args:
            other (numpy arra `Stoner.Core.DataFile` or a dictionary or a list):
                The object to be added to the DataFile

        Note:
            -   if other is a dictionary then the keys of the dictionary are passed to
                :py:meth:`find_col` to see if they match a column, in which case the
                corresponding value will be used for theat column in the new row.
                Columns which do not have a matching key will be set to NaN. If other has keys
                that are not found as columns in self, additional columns are added.
            -   If other is a list, then the add method is called recursively for each element
                of the list.
            -   Returns: A Datafile object with the rows of @a other appended
                to the rows of the current object.
            -   If other is a 1D numopy array with the same number of
                elements as their are columns in @a self.data then the numpy
                array is treated as a new row of data If @a ither is a 2D numpy
                array then it is appended if it has the same number of
                columns and @a self.data.
        """
        newdata = self.clone
        return add_core(other, newdata)

    def __iadd__(self, other):
        """Implement a += operator to concatenate rows of data inplace.

        Args:
            other (numpy arra `Stoner.Core.DataFile` or a dictionary or a list):
                The object to be added to the DataFile

        Note:
            -   if other is a dictionary then the keys of the dictionary are passed to
                :py:meth:`find_col` to see if they match a column, in which case the
                corresponding value will be used for theat column in the new row.
                Columns which do not have a matching key will be set to NaN. If other has keys
                that are not found as columns in self, additional columns are added.
            -   If other is a list, then the add method is called recursively for each element
                of the list.
            -   Returns: A Datafile object with the rows of @a other appended
                to the rows of the current object.
            -   If other is a 1D numopy array with the same number of
                elements as their are columns in @a self.data then the numpy
                array is treated as a new row of data If @a ither is a 2D numpy
                array then it is appended if it has the same number of
                columns and @a self.data.
        """
        newdata = self
        return add_core(other, newdata)

    def __and__(self, other):
        """Implement the & operator to concatenate columns of data in a :py:class:`DataFile` object.

        Args:
            other  (numpy array or :py:class:`DataFile`):
                Data to be added to this DataFile instance

        Returns:
            ():py:class:`DataFile`):
                new Data object with the columns of other concatenated as new columns at the end of the self object.

        Note:
            Whether other is a numopy array of :py:class:`DataFile`, it must
            have the same or fewer rows than the self object.
            The size of @a other is increased with zeros for the extra rows.
            If other is a 1D numpy array it is treated as a column vector.
            The new columns are given blank column headers, but the
            length of the :py:meth:`column_headers` is
            increased to match the actual number of columns.
        """
        # Prep the final DataFile
        newdata = self.clone
        return and_core(other, newdata)

    def __iand__(self, other):
        """Implement the &= operator to concatenate columns of data in a :py:class:`DataFile` object.

        Args:
            other  (numpy array or :py:class:`DataFile`):
                Data to be added to this DataFile instance

        Returns:
            ():py:class:`DataFile`):
                new Data object with the columns of other concatenated as new columns at the end of the self object.

        Note:
            Whether other is a numopy array of :py:class:`DataFile`, it must
            have the same or fewer rows than the self object.
            The size of @a other is increased with zeros for the extra rows.
            If other is a 1D numpy array it is treated as a column vector.
            The new columns are given blank column headers, but the
            length of the :py:meth:`column_headers` is
            increased to match the actual number of columns.
        """
        newdata = self
        return and_core(other, newdata)

    def __eq__(self, other):
        """Equality operator.

        Args:
            other (DataFile):
                The object to test for equality against.

        Returns:
            bool:
                True if data, column headers and metadata are equal.
        """
        if not isinstance(other, DataFileOperatorsMixin):  # Start checking we're all DataFile's
            return False
        if self.data.shape != other.data.shape:  # Check we have the same data
            return False
        if not np.all(self.data[~np.isnan(self.data)] == other.data[~np.isnan(other.data)]):  # Don't compare nan!
            return False
        if len(self.column_headers) != len(other.column_headers) or np.any(
            [c1 != c2 for c1, c2 in zip(self.column_headers, other.column_headers)]
        ):  # And the same headers
            return False
        if not super().__eq__(other):  # Check metadata
            return False
        # Ok give up, we are the same!
        return True

    def __floordiv__(self, other):
        """Just aslias for self.column(other)."""
        if not isinstance(other, index_types):
            return NotImplemented
        return self.column(other)

    def __lshift__(self, other):
        """Convert a string or iterable to a new DataFile like object.

        Args:
            other (string or iterable object):
                Used to source the DataFile object

        Returns:
            (DataFile):
                A new :py:class:`DataFile` object

        Todo:
            Make code work better with streams

        Overird the left shift << operator for a string or an iterable object to import using the :py:meth:`__
        read_iterable` function.
        """
        newdata = type(self)()
        if isinstance(other, string_types):
            lines = map(lambda x: x, other.splitlines())
            newdata.__read_iterable(lines)
        elif isiterable(other):
            newdata.__read_iterable(other)
        else:
            return NotImplemented
        return type(self)(newdata)

    def __mod__(self, other):
        """Overload the % operator to mean column deletion.

        Args:
            Other (column index):
                column(s) to delete.

        Return:
            (self):
                A copy of self with a column deleted.
        """
        newdata = self.clone
        return mod_core(other, newdata)

    def __imod__(self, other):
        """Overload the % operator to mean in-place column deletion.

        Args:
            Other (column index):
                column(s) to delete.

        Return:
            (self):
                A copy of self with a column deleted.
        """
        newdata = self
        return mod_core(other, newdata)

    def __sub__(self, other):
        """Implement what to do when subtraction operator is used.

        Args:
            other (int,list of integers):
                Delete row(s) from data.

        Returns:
            (DataFile):
                A :py:data:`DataFile` with rows removed.
        """
        newdata = self.clone
        return sub_core(other, newdata)

    def __isub__(self, other):
        """Implement what to do when subtraction operator is used.

        Args:
            other (int,list of integers):
                Delete row(s) from data.

        Returns:
            (self):
                The :py:data:`DataFile` with rows removed.
        """
        newdata = self
        return sub_core(other, newdata)

    def __invert__(self):
        """Swap x and y column assignments around."""
        ret = self.clone
        setas = list(self.setas)
        cols = self.setas._cols
        if cols["axes"] == 2:
            swaps = zip(["ycol", "yerr"], ["x", "d"])
        elif cols["axes"] >= 3:
            swaps = zip(["ycol", "zcol", "yerr", "zerr"], ["z", "x", "f", "d"])
        else:
            raise ValueError("Cannot invert unless at least two columns are identified in setas")
        setas[cols["xcol"]] = "y"
        if cols["has_xerr"]:
            setas[cols["xerr"]] = "e"
        for cname, nlet in swaps:
            for c in cols[cname]:
                setas[c] = nlet
        ret.setas = setas
        return ret  #

    def __read_iterable(self, reader):
        """Read a string representation of py:class:`DataFile` in line by line."""
        if isiterable(reader):
            reader = iter(reader)
        if "readline" in dir(reader):  # Filelike iterator
            readline = reader.readline
        elif "__next__" in dir(reader):  # Python v3 iterator
            readline = reader.__next__

        else:
            return NotImplemented
        row = readline().split("\t")
        if row[0].strip() == "TDI Format 1.5":
            fmt = 1.5
        elif row[0].strip() == "TDI Format=Text 1.0":
            fmt = 1.0
        else:
            raise RuntimeError("Not a TDI File")
        col_headers_tmp = [x.strip() for x in row[1:]]
        cols = len(col_headers_tmp)
        self._data._setas = _setas("." * cols)
        self.data = DataArray([], setas=self._data._setas)
        for r in reader:
            if r.strip() == "":  # Blank line
                continue
            row = r.rstrip().split("\t")
            cols = max(cols, len(row) - 1)
            if row[0].strip() != "":
                md = row[0].split("=")
                if len(md) >= 2:
                    md[1] = "=".join(md[1:])
                elif len(md) <= 1:
                    md.extend(["", ""])

                if fmt == 1.5:
                    self.metadata[md[0].strip()] = md[1].strip()
                elif fmt == 1.0:
                    self.metadata[md[0].strip()] = self.metadata.string_to_type(md[1].strip())
            if len(row) < 2:
                continue
            self.data = np.append(self.data, self._conv_float(row[1:]))
        self.data = np.reshape(self.data, (-1, cols))
        self.column_headers = [f"Column {i}" for i in range(cols)]
        for i, head_temp in enumerate(col_headers_tmp):
            self.column_headers[i] = head_temp
        self["TDI Format"] = fmt
