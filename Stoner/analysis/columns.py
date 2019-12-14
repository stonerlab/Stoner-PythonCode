#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Channel column operations functions for analysis code.
"""

__all__ = ["ColumnOps"]

import numpy as np

from Stoner.tools import isiterable, all_type
from Stoner.compat import index_types


class ColumnOps(object):

    """A mixin calss designed to work with :py:class:`Stoner.Core.DataFile` to provide additional stats methods."""

    def __get_math_val(self, col):
        """Utility routine to interpret col as either a column index or value or an array of values.

        Args:
            col (various):
                If col can be interpreted as a column index then return the first matching column.
                If col is a 1D array of the same length as the data then just return the data. If col is a
                float then just return it as a float.

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
        elif isinstance(col, np.ndarray) and len(col.shape) == 1 and len(col) == len(self):
            data = col
            name = "data"
        elif isinstance(col, float):
            data = col * np.ones(len(self))
            name = str(col)
        else:
            raise RuntimeError("Bad column index: {}".format(col))
        return data, name

    def add(self, a, b, replace=False, header=None, index=None):
        """Add one column, number or array (b) to another column (a).

        Args:
            a (index):
                First column to work with
            b (index, float or 1D array):
                Second column to work with.

        Keyword Arguments:
            header (string or None):
                new column header  (defaults to a-b
            replace (bool):
                Replace the a column with the new data
            index (column index or None):
                Column to insert new data at.

        Returns:
            (:py:class:`Stoner.Data`):
                The newly modified Data object.

        If a and b are tuples of length two, then the firstelement is assumed to be the value and
        the second element an uncertainty in the value. The uncertainties will then be propagated and an
        additional column with the uncertainites will be added to the data.
        """
        a = self.find_col(a)
        if (
            isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)) and len(a) == 2 and len(b) == 2
        ):  # Error columns on
            (a, e1) = a
            (b, e2) = b
            e1data = self.__get_math_val(e1)[0]
            e2data = self.__get_math_val(e2)[0]
            err_header = None
            err_calc = lambda adata, bdata, e1data, e2data: np.sqrt(e1data ** 2 + e2data ** 2)
        else:
            err_calc = None
        adata, aname = self.__get_math_val(a)
        bdata, bname = self.__get_math_val(b)
        if isinstance(header, tuple) and len(header) == 2:
            header, err_header = header
        if header is None:
            header = "{}+{}".format(aname, bname)
        if err_calc is not None and err_header is None:
            err_header = "Error in " + header
        index = self.shape[1] if index is None else self.find_col(index)
        if err_calc is not None:
            err_data = err_calc(adata, bdata, e1data, e2data)
        self.add_column((adata + bdata), header=header, index=index, replace=replace)
        if err_calc is not None:
            self.add_column(err_data, header=err_header, index=index + 1, replace=False)
        return self

    def diffsum(self, a, b, replace=False, header=None, index=None):
        r"""Calculate :math:`\frac{a-b}{a+b}` for the two columns *a* and *b*.

        Args:
            a (index):
                First column to work with
            b (index, float or 1D array):
                Second column to work with.

        Keyword Arguments:
            header (string or None):
                new column header  (defaults to a-b
            replace (bool):
                Replace the a column with the new data
            index (column index or None):
                Column to insert new data at.
        Returns:
            (:py:class:`Stoner.Data`):
                The newly modified Data object.

        If a and b are tuples of length two, then the firstelement is assumed to be the value and
        the second element an uncertainty in the value. The uncertainties will then be propagated and an
        additional column with the uncertainites will be added to the data.
        """
        a = self.find_col(a)
        if (
            isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == 2 and len(b) == 2
        ):  # Error columns on
            (a, e1) = a
            (b, e2) = b
            e1data = self.__get_math_val(e1)[0]
            e2data = self.__get_math_val(e2)[0]
            err_header = None
            err_calc = lambda adata, bdata, e1data, e2data: np.sqrt(
                (1.0 / (adata + bdata) - (adata - bdata) / (adata + bdata) ** 2) ** 2 * e1data ** 2
                + (-1.0 / (adata + bdata) - (adata - bdata) / (adata + bdata) ** 2) ** 2 * e2data ** 2
            )
        else:
            err_calc = None
        adata, aname = self.__get_math_val(a)
        bdata, bname = self.__get_math_val(b)
        if isinstance(header, tuple) and len(header) == 2:
            header, err_header = header
        if header is None:
            header = "({}-{})/({}+{})".format(aname, bname, aname, bname)
        if err_calc is not None and err_header is None:
            err_header = "Error in " + header
        if err_calc is not None:
            err_data = err_calc(adata, bdata, e1data, e2data)
        index = self.shape[1] if index is None else self.find_col(index)
        self.add_column((adata - bdata) / (adata + bdata), header=header, index=index, replace=replace)
        if err_calc is not None:
            self.add_column(err_data, header=err_header, index=index + 1, replace=False)
        return self

    def divide(self, a, b, replace=False, header=None, index=None):
        """Divide one column (a) by  another column, number or array (b).

        Args:
            a (index):
                First column to work with
            b (index, float or 1D array):
                Second column to work with.

        Keyword Arguments:
            header (string or None):
                new column header  (defaults to a-b
            replace (bool):
                Replace the a column with the new data
            index (column index or None):
                Column to insert new data at.
        Returns:
            (:py:class:`Stoner.Data`):
                The newly modified Data object.

        If a and b are tuples of length two, then the firstelement is assumed to be the value and
        the second element an uncertainty in the value. The uncertainties will then be propagated and an
        additional column with the uncertainites will be added to the data.
        """
        a = self.find_col(a)
        if (
            isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == 2 and len(b) == 2
        ):  # Error columns on
            (a, e1) = a
            (b, e2) = b
            e1data = self.__get_math_val(e1)[0]
            e2data = self.__get_math_val(e2)[0]
            err_header = None
            err_calc = lambda adata, bdata, e1data, e2data: np.sqrt(
                (e1data / adata) ** 2 + (e2data / bdata) ** 2
            ) * np.abs(adata / bdata)
        else:
            err_calc = None
        adata, aname = self.__get_math_val(a)
        bdata, bname = self.__get_math_val(b)
        if isinstance(header, tuple) and len(header) == 2:
            header, err_header = header
        if header is None:
            header = "{}/{}".format(aname, bname)
        if err_calc is not None and err_header is None:
            err_header = "Error in " + header
        index = self.shape[1] if index is None else self.find_col(index)
        if err_calc is not None:
            err_data = err_calc(adata, bdata, e1data, e2data)
        self.add_column((adata / bdata), header=header, index=index, replace=replace)
        if err_calc is not None:
            self.add_column(err_data, header=err_header, index=index + 1, replace=False)
        return self

    def max(self, column=None, bounds=None):
        """Find maximum value and index in a column of data.

        Args:
            column (index):
                Column to look for the maximum in

        Keyword Arguments:
            bounds (callable):
                A callable function that takes a single argument list of
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
        """Find mean value of a data column.

        Args:
            column (index):
                Column to look for the maximum in

        Keyword Arguments:
            sigma (column index or array):
                The uncertainity noted for each value in the mean
            bounds (callable):
                A callable function that takes a single argument list of
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
            w = 1 / (sigma ** 2 + 1e-8)
            norm = w.sum(axis=0)
            error = np.sqrt((sigma ** 2).sum(axis=0)) / len(sigma)
            result = (ydata * w).mean(axis=0) / norm, error
        if bounds is not None:
            self._pop_mask()
        return result

    def min(self, column=None, bounds=None):
        """Find minimum value and index in a column of data.

        Args:
            column (index):
                Column to look for the maximum in

        Keyword Arguments:
            bounds (callable):
                A callable function that takes a single argument list of
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

    def multiply(self, a, b, replace=False, header=None, index=None):
        """Multiply one column (a) by  another column, number or array (b).

        Args:
            a (index):
                First column to work with
            b (index, float or 1D array):
                Second column to work with.

        Keyword Arguments:
            header (string or None):
                new column header  (defaults to a-b
            replace (bool):
                Replace the a column with the new data
            index (column index or None):
                Column to insert new data at.

        Returns:
            (:py:class:`Stoner.Data`):
                The newly modified Data object.

        If a and b are tuples of length two, then the firstelement is assumed to be the value and
        the second element an uncertainty in the value. The uncertainties will then be propagated and an
        additional column with the uncertainites will be added to the data.
        """
        a = self.find_col(a)
        if (
            isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == 2 and len(b) == 2
        ):  # Error columns on
            (a, e1) = a
            (b, e2) = b
            e1data = self.__get_math_val(e1)[0]
            e2data = self.__get_math_val(e2)[0]
            err_header = None
            err_calc = lambda adata, bdata, e1data, e2data: np.sqrt(
                (e1data / adata) ** 2 + (e2data / bdata) ** 2
            ) * np.abs(adata * bdata)
        else:
            err_calc = None
        adata, aname = self.__get_math_val(a)
        bdata, bname = self.__get_math_val(b)
        if isinstance(header, tuple) and len(header) == 2:
            header, err_header = header
        if header is None:
            header = "{}*{}".format(aname, bname)
        if err_calc is not None and err_header is None:
            err_header = "Error in " + header
        index = self.shape[1] if index is None else self.find_col(index)
        if err_calc is not None:
            err_data = err_calc(adata, bdata, e1data, e2data)
        self.add_column((adata * bdata), header=header, index=index, replace=replace)
        if err_calc is not None:
            self.add_column(err_data, header=err_header, index=index + 1, replace=False)
        return self

    def span(self, column=None, bounds=None):
        """Returns a tuple of the maximum and minumum values within the given column and bounds by calling into
        :py:meth:`AnalysisMixin.max` and :py:meth:`AnalysisMixin.min`.

        Args:
            column (index):
                Column to look for the maximum in

        Keyword Arguments:
            bounds (callable):
                A callable function that takes a single argument list of
                numbers representing one row, and returns True for all rows to search in.

        Returns:
            (float,float):
                A tuple of (min value, max value)

        Note:
            If column is not defined (or is None) the :py:attr:`DataFile.setas` column
            assignments are used.

        """
        return (self.min(column, bounds)[0], self.max(column, bounds)[0])

    def std(self, column=None, sigma=None, bounds=None):
        """Find standard deviation value of a data column.

        Args:
            column (index):
                Column to look for the maximum in

        Keyword Arguments:
            sigma (column index or array):
                The uncertainity noted for each value in the mean
            bounds (callable):
                A callable function that takes a single argument list of
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
        weights = 1 / sigma ** 2
        weights[np.isnan(weights)] = 0.0

        result = np.sqrt(np.cov(ydata, aweights=weights))

        if bounds is not None:
            self._pop_mask()
        return result

    def subtract(self, a, b, replace=False, header=None, index=None):
        """Subtract one column, number or array (b) from another column (a).

        Args:
            a (index):
                First column to work with
            b (index, float or 1D array):
                Second column to work with.

        Keyword Arguments:
            header (string or None):
                new column header  (defaults to a-b
            replace (bool):
                Replace the a column with the new data
            index (column index or None):
                Column to insert new data at.

        Returns:
            (:py:class:`Stoner.Data`):
                The newly modified Data object.

        If a and b are tuples of length two, then the firstelement is assumed to be the value and
        the second element an uncertainty in the value. The uncertainties will then be propagated and an
        additional column with the uncertainites will be added to the data.
        """
        a = self.find_col(a)
        if (
            isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == 2 and len(b) == 2
        ):  # Error columns on
            (a, e1) = a
            (b, e2) = b
            e1data = self.__get_math_val(e1)[0]
            e2data = self.__get_math_val(e2)[0]
            err_header = None
            err_calc = lambda adata, bdata, e1data, e2data: np.sqrt(e1data ** 2 + e2data ** 2)
        else:
            err_calc = None
        adata, aname = self.__get_math_val(a)
        bdata, bname = self.__get_math_val(b)
        if isinstance(header, tuple) and len(header) == 2:
            header, err_header = header
        if header is None:
            header = "{}-{}".format(aname, bname)
        if err_calc is not None and err_header is None:
            err_header = "Error in " + header
        index = self.shape[1] if index is None else self.find_col(index)
        if err_calc is not None:
            err_data = err_calc(adata, bdata, e1data, e2data)
        self.add_column((adata - bdata), header=header, index=index, replace=replace)
        if err_calc is not None:
            a = self.find_col(a)
            self.add_column(err_data, header=err_header, index=index + 1, replace=False)
        return self
