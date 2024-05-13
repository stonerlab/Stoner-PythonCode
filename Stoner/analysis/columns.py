#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Channel column operations functions for analysis code."""

__all__ = ["ColumnOpsMixin"]

import numpy as np

from Stoner.tools import isiterable, all_type
from Stoner.compat import index_types


class ColumnOpsMixin:
    """A mixin calss designed to work with :py:class:`Stoner.Core.DataFile` to provide additional stats methods."""

    def _do_error_calc(self, col_a, col_b, error_type="relative"):
        """Do an error calculation."""
        col_a = self.find_col(col_a)
        error_calc = None
        if (
            isinstance(col_a, (list, tuple))
            and isinstance(col_b, (list, tuple))
            and len(col_a) == 2
            and len(col_b) == 2
        ):  # Error columns on
            (col_a, e1) = col_a
            (col_b, e2) = col_b
            e1data = self.__get_math_val(e1)[0]
            e2data = self.__get_math_val(e2)[0]
            if error_type == "relative":

                def error_calc(adata, bdata):  # pylint: disable=function-redefined
                    """Relative error summation."""
                    return np.sqrt((e1data / adata) ** 2 + (e2data / bdata) ** 2)

            elif error_type == "absolute":

                def error_calc(adata, bdata):  # pylint: disable=function-redefined, unused-argument
                    """Sum absolute errors."""
                    return np.sqrt(e1data**2 + e2data**2)

            elif error_type == "diffsum":

                def error_calc(adata, bdata):  # pylint: disable=function-redefined
                    """Calculate error for difference over sum."""
                    return np.sqrt(
                        (1.0 / (adata + bdata) - (adata - bdata) / (adata + bdata) ** 2) ** 2 * e1data**2
                        + (-1.0 / (adata + bdata) - (adata - bdata) / (adata + bdata) ** 2) ** 2 * e2data**2
                    )

            else:
                raise ValueError(f"Unknown error calculation mode {error_type}")

        adata, aname = self.__get_math_val(col_a)
        bdata, bname = self.__get_math_val(col_b)
        return adata, bdata, error_calc, aname, bname

    def __get_math_val(self, col):
        """Interpret col as either col_a column index or value or an array of values.

        Args:
            col (various):
                If col can be interpreted as col_a column index then return the first matching column.
                If col is col_a 1D array of the same length as the data then just return the data. If col is col_a
                float then just return it as col_a float.

        Returns:
            (tuple of (:py:class:`Stoner.cpre.DataArray`,str)):
                The matching data.
        """
        if isinstance(col, index_types):
            col = self.find_col(col)
            if isinstance(col, list):
                col = col[0]
            data = self.column(col)
            name = self.column_headers[col]
        elif isinstance(col, np.ndarray) and col.ndim == 1 and len(col) == len(self):
            data = col
            name = "data"
        elif isinstance(col, float):
            data = col * np.ones(len(self))
            name = str(col)
        else:
            raise RuntimeError(f"Bad column index: {col}")
        return data, name

    def add(self, col_a, col_b, replace=False, header=None, index=None):
        """Add one column, number or array (col_b) to another column (col_a).

        Args:
            col_a (index):
                First column to work with
            col_b (index, float or 1D array):
                Second column to work with.

        Keyword Arguments:
            header (string or None):
                new column header  (defaults to a-b
            replace (bool):
                Replace the col_a column with the new data
            index (column index or None):
                Column to insert new data at.

        Returns:
            (:py:class:`Stoner.Data`):
                The newly modified Data object.

        If col_a and col_b are tuples of length two, then the firstelement is assumed to be the value and
        the second element an uncertainty in the value. The uncertainties will then be propagated and an
        additional column with the uncertainites will be added to the data.
        """
        adata, bdata, err_calc, aname, bname = self._do_error_calc(col_a, col_b, error_type="absolute")
        err_header = None
        if isinstance(header, tuple) and len(header) == 2:
            header, err_header = header
        if header is None:
            header = f"{aname}+{bname}"
        if err_calc is not None and err_header is None:
            err_header = "Error in " + header
        index = self.shape[1] if index is None else self.find_col(index)
        if err_calc is not None:
            err_data = err_calc(adata, bdata)
        self.add_column((adata + bdata), header=header, index=index, replace=replace)
        if err_calc is not None:
            self.add_column(err_data, header=err_header, index=index + 1, replace=False)
        return self

    def diffsum(self, col_a, col_b, replace=False, header=None, index=None):
        r"""Calculate :math:`\frac{a-b}{a+b}` for the two columns *a* and *b*.

        Args:
            col_a (index):
                First column to work with
            col_b (index, float or 1D array):
                Second column to work with.

        Keyword Arguments:
            header (string or None):
                new column header  (defaults to a-b
            replace (bool):
                Replace the col_a column with the new data
            index (column index or None):
                Column to insert new data at.

        Returns:
            (:py:class:`Stoner.Data`):
                The newly modified Data object.

        If col_a and col_b are tuples of length two, then the firstelement is assumed to be the value and
        the second element an uncertainty in the value. The uncertainties will then be propagated and an
        additional column with the uncertainites will be added to the data.
        """
        adata, bdata, err_calc, aname, bname = self._do_error_calc(col_a, col_b, error_type="diffsum")
        err_header = None
        if isinstance(header, tuple) and len(header) == 2:
            header, err_header = header
        if header is None:
            header = f"({aname}-{bname})/({aname}+{bname})"
        if err_calc is not None and err_header is None:
            err_header = "Error in " + header
        if err_calc is not None:
            err_data = err_calc(adata, bdata)
        index = self.shape[1] if index is None else self.find_col(index)
        self.add_column((adata - bdata) / (adata + bdata), header=header, index=index, replace=replace)
        if err_calc is not None:
            self.add_column(err_data, header=err_header, index=index + 1, replace=False)
        return self

    def divide(self, col_a, col_b, replace=False, header=None, index=None):
        """Divide one column (col_a) by  another column, number or array (col_b).

        Args:
            col_a (index):
                First column to work with
            col_b (index, float or 1D array):
                Second column to work with.

        Keyword Arguments:
            header (string or None):
                new column header  (defaults to a-b
            replace (bool):
                Replace the col_a column with the new data
            index (column index or None):
                Column to insert new data at.

        Returns:
            (:py:class:`Stoner.Data`):
                The newly modified Data object.

        If col_a and col_b are tuples of length two, then the firstelement is assumed to be the value and
        the second element an uncertainty in the value. The uncertainties will then be propagated and an
        additional column with the uncertainites will be added to the data.
        """
        adata, bdata, err_calc, aname, bname = self._do_error_calc(col_a, col_b, error_type="relative")
        err_header = None
        if isinstance(header, tuple) and len(header) == 2:
            header, err_header = header
        if header is None:
            header = f"{aname}/{bname}"
        if err_calc is not None and err_header is None:
            err_header = "Error in " + header
        index = self.shape[1] if index is None else self.find_col(index)
        self.add_column((adata / bdata), header=header, index=index, replace=replace)
        if err_calc is not None:
            err_data = err_calc(adata, bdata) * np.abs(adata / bdata)
            self.add_column(err_data, header=err_header, index=index + 1, replace=False)
        return self

    def max(self, column=None, bounds=None):
        """Find maximum value and index in col_a column of data.

        Args:
            column (index):
                Column to look for the maximum in

        Keyword Arguments:
            bounds (callable):
                col_a callable function that takes col_a single argument list of
                numbers representing one row, and returns True for all rows to search in.

        Returns:
            (float,int):
                (maximum value,row index of max value)

        Note:
            If column is not defined (or is None) the :py:attr:`DataFile.setas` column
            assignments are used.
        """
        if column is None:
            col = self.setas._get_cols("ycol")
        else:
            col = self.find_col(column)
        if bounds is not None:
            self._push_mask()
            self._set_mask(bounds, True, col)
        result = self.data[:, col].max(), self.data[:, col].argmax()
        if bounds is not None:
            self._pop_mask()
        return result

    def mean(self, column=None, sigma=None, bounds=None):
        """Find mean value of col_a data column.

        Args:
            column (index):
                Column to look for the maximum in

        Keyword Arguments:
            sigma (column index or array):
                The uncertainty noted for each value in the mean
            bounds (callable):
                col_a callable function that takes col_a single argument list of
                numbers representing one row, and returns True for all rows to search in.

        Returns:
            (float):
                The mean of the data.

        Note:
            If column is not defined (or is None) the :py:attr:`DataFile.setas` column
            assignments are used.

        .. todo::
            Fix the row index when the bounds function is used - see note of :py:meth:`AnalysisMixin.max`
        """
        _ = self._col_args(scalar=True, ycol=column, yerr=sigma)

        if bounds is not None:
            self._push_mask()
            self._set_mask(bounds, True, _.ycol)

        if isiterable(sigma) and len(sigma) == len(self) and all_type(sigma, float):
            sigma = np.array(sigma)
            _["has_yerr"] = True
        elif _.has_yerr:
            sigma = self.data[:, _.yerr]

        if not _.has_yerr:
            result = self.data[:, _.ycol].mean()
        else:
            ydata = self.data[:, _.ycol]
            w = 1 / (sigma**2 + 1e-8)
            norm = w.sum(axis=0)
            error = np.sqrt((sigma**2).sum(axis=0)) / len(sigma)
            result = (ydata * w).mean(axis=0) / norm, error
        if bounds is not None:
            self._pop_mask()
        return result

    def min(self, column=None, bounds=None):
        """Find minimum value and index in col_a column of data.

        Args:
            column (index):
                Column to look for the maximum in

        Keyword Arguments:
            bounds (callable):
                col_a callable function that takes col_a single argument list of
                numbers representing one row, and returns True for all rows to search in.

        Returns:
            (float,int):
                (minimum value,row index of min value)

        Note:
            If column is not defined (or is None) the :py:attr:`DataFile.setas` column
            assignments are used.
        """
        if column is None:
            col = self.setas._get_cols("ycol")
        else:
            col = self.find_col(column)
        if bounds is not None:
            self._push_mask()
            self._set_mask(bounds, True, col)
        result = self.data[:, col].min(), self.data[:, col].argmin()
        if bounds is not None:
            self._pop_mask()
        return result

    def multiply(self, col_a, col_b, replace=False, header=None, index=None):
        """Multiply one column (col_a) by  another column, number or array (col_b).

        Args:
            col_a (index):
                First column to work with
            col_b (index, float or 1D array):
                Second column to work with.

        Keyword Arguments:
            header (string or None):
                new column header  (defaults to a-b
            replace (bool):
                Replace the col_a column with the new data
            index (column index or None):
                Column to insert new data at.

        Returns:
            (:py:class:`Stoner.Data`):
                The newly modified Data object.

        If col_a and col_b are tuples of length two, then the firstelement is assumed to be the value and
        the second element an uncertainty in the value. The uncertainties will then be propagated and an
        additional column with the uncertainites will be added to the data.
        """
        adata, bdata, err_calc, aname, bname = self._do_error_calc(col_a, col_b, error_type="relative")
        err_header = None
        if isinstance(header, tuple) and len(header) == 2:
            header, err_header = header
        if header is None:
            header = f"{aname}*{bname}"
        if err_calc is not None and err_header is None:
            err_header = "Error in " + header
        index = self.shape[1] if index is None else self.find_col(index)
        if err_calc is not None:
            err_data = err_calc(adata, bdata) * np.abs(adata * bdata)
        self.add_column((adata * bdata), header=header, index=index, replace=replace)
        if err_calc is not None:
            self.add_column(err_data, header=err_header, index=index + 1, replace=False)
        return self

    def span(self, column=None, bounds=None):
        """Return a tuple of the maximum and minimum values within the given column and bounds.

        Args:
            column (index):
                Column to look for the maximum in

        Keyword Arguments:
            bounds (callable):
                col_a callable function that takes col_a single argument list of
                numbers representing one row, and returns True for all rows to search in.

        Returns:
            (float,float):
                col_a tuple of (min value, max value)

        Note:
            This works by calling into :py:meth:`Data.max` and :py:meth:`Data.min`.

            If column is not defined (or is None) the :py:attr:`DataFile.setas` column
            assignments are used.

        """
        return (self.min(column, bounds)[0], self.max(column, bounds)[0])

    def std(self, column=None, sigma=None, bounds=None):
        """Find standard deviation value of col_a data column.

        Args:
            column (index):
                Column to look for the maximum in

        Keyword Arguments:
            sigma (column index or array):
                The uncertainty noted for each value in the mean
            bounds (callable):
                col_a callable function that takes col_a single argument list of
                numbers representing one row, and returns True for all rows to search in.

        Returns:
            (float):
                The standard deviation of the data.

        Note:
            If column is not defined (or is None) the :py:attr:`DataFile.setas` column
            assignments are used.

        .. todo::
            Fix the row index when the bounds function is used - see note of :py:meth:`AnalysisMixin.max`
        """
        _ = self._col_args(scalar=True, ycol=column, yerr=sigma)

        if bounds is not None:
            self._push_mask()
            self._set_mask(bounds, True, _.ycol)

        if isiterable(sigma) and len(sigma) == len(self) and all_type(sigma, float):
            sigma = np.array(sigma)
        elif _.yerr:
            sigma = self.data[:, _.yerr]
        else:
            sigma = np.ones(len(self))

        ydata = self.data[:, _.ycol]

        sigma = np.abs(sigma) / np.nanmax(np.abs(sigma))
        sigma = np.where(sigma < 1e-8, 1e-8, sigma)
        weights = 1 / sigma**2
        weights[np.isnan(weights)] = 0.0

        result = np.sqrt(np.cov(ydata, aweights=weights))

        if bounds is not None:
            self._pop_mask()
        return result

    def subtract(self, col_a, col_b, replace=False, header=None, index=None):
        """Subtract one column, number or array (col_b) from another column (col_a).

        Args:
            col_a (index):
                First column to work with
            col_b (index, float or 1D array):
                Second column to work with.

        Keyword Arguments:
            header (string or None):
                new column header  (defaults to a-b
            replace (bool):
                Replace the col_a column with the new data
            index (column index or None):
                Column to insert new data at.

        Returns:
            (:py:class:`Stoner.Data`):
                The newly modified Data object.

        If col_a and col_b are tuples of length two, then the firstelement is assumed to be the value and
        the second element an uncertainty in the value. The uncertainties will then be propagated and an
        additional column with the uncertainites will be added to the data.
        """
        adata, bdata, err_calc, aname, bname = self._do_error_calc(col_a, col_b, error_type="absolute")
        err_header = None
        if isinstance(header, tuple) and len(header) == 2:
            header, err_header = header
        if header is None:
            header = f"{aname}-{bname}"
        if err_calc is not None and err_header is None:
            err_header = "Error in " + header
        index = self.shape[1] if index is None else self.find_col(index)
        if err_calc is not None:
            err_data = err_calc(adata, bdata)
        self.add_column((adata - bdata), header=header, index=index, replace=replace)
        if err_calc is not None:
            col_a = self.find_col(col_a)
            self.add_column(err_data, header=err_header, index=index + 1, replace=False)
        return self
