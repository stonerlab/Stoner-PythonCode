# -*- coding: utf-8 -*-
"""Mxinin classes for DataFile objects."""

__all__ = ["DataFileSearchMixin"]

import copy
from collections.abc import Iterable

import numpy as np
from statsmodels.stats.weightstats import DescrStatsW

from ..compat import _pattern_type, int_types
from ..tools import all_type, isiterable, operator
from ..tools.widgets import RangeSelect


class DataFileSearchMixin:
    """Mixin class that provides the search, selecting and sorting methods for a DataFile."""

    def _search_index(self, xcol=None, value=None, accuracy=0.0, invert=False):
        """Return an array of booleans for indexing matching rows for use with search method."""
        _ = self._col_args(scalar=False, xcol=xcol)
        x = self.find_col(_.xcol, force_list=True)[0]  # Workaround in newer numpy
        match value:
            case int() | float():
                ix = np.isclose(self.data[:, x], value, atol=accuracy)
            case (_, _):
                (low, u) = (min(value), max(value))
                low -= accuracy
                u += accuracy
                v = self.data[:, x]
                low = np.ones_like(v) * low
                u = np.ones_like(v) * u
                ix = np.logical_and(v > low, v <= u)
            case list() | np.ndarray() if all_type(value, bool) and len(value) <= len(self):
                if len(value) < len(self):  # Expand array if necessary
                    ix = np.append(value, [False] * len(self) - len(value))
                else:
                    ix = np.array(value)
            case list() | np.ndarray():
                ix = np.zeros(len(self), dtype=bool)
                for v in value:
                    ix = np.logical_or(ix, self._search_index(xcol, v, accuracy))
            case _ if callable(value):
                ix = np.array([value(r[x], r) for r in self], dtype=bool)
            case _ if value is None:
                selector = RangeSelect()
                ix = selector(self, x, accuracy)
            case _:
                raise RuntimeError(f"Unknown search value type {value}")

        ix = np.logical_xor(invert, ix)
        if ix.ndim > 1:
            ix = ix[:, 0]
        return ix

    def asarray(self):
        """Provide a consistent way to get at the underlying array data."""
        return self.data

    def closest(self, value, xcol=None):
        """Return the row in a data file which has an x-column value closest to the given value.

        Args:
            value (float):
                Value to search for.

        Keyword Arguments:
            xcol (index or None):
                Column in which to look for value, or None to use setas.

        Returns:
            ndarray:
                A single row of data as a :py:class:`Stoner.Core.DataArray`.

        Notes:
            To find which row it is that has been returned, use the :py:attr:`Stoner.Core.DataArray.i`
            index attribute.
        """
        _ = self._col_args(xcol=xcol)
        xdata = np.abs(self // _.xcol - value)
        i = int(xdata.argmin())
        return self[i]

    def column(self, col):
        """Extract one or more columns of data from the datafile.

        Args:
            col (int, string, list or re):
                is the column index as defined for :py:meth:`DataFile.find_col`

        Returns:
            (ndarray):
                One or more columns of data as a :py:class:`numpy.ndarray`.
        """
        return self.data[:, self.find_col(col)]

    def find_col(self, col, force_list=False):
        """Indexes the column headers in order to locate a column of data.shape.

        Indexing can be by supplying an integer, a string, a regular expression, a slice or a list of any of the above.

        -   Integer indices are simply checked to ensure that they are in range
        -   String indices are first checked for an exact match against a column header
            if that fails they are then compiled to a regular expression and the first
            match to a column header is taken.
        -   A regular expression index is simply matched against the column headers and the
            first match found is taken. This allows additional regular expression options
            such as case insensitivity.
        -   A slice index is converted to a list of integers and processed as below
        -   A list index returns the results of feading each item in the list at :py:meth:`find_col`
            in turn.

        Args:
            col (int, a string, a re, a slice or a list):
                Which column(s) to retuirn indices for.

        Keyword Arguments:
            force_list (bool):
                Force the output always to be a list. Mainly for internal use only

        Returns:
            int, list of ints:
                The matching column index as an integer or a KeyError
        """
        return self.data._setas.find_col(col, force_list)

    def find_duplicates(self, xcol=None, delta=1e-8):
        """Find rows with duplicated values of the search column(s).

        Keyword Arguments:
            xcol (index types):
                The column)s) to search for duplicates in.
            delta (float or array):
                The absolute difference(s) to consider equal when comparing floats.

        Returns:
            (dictionary of value:[list of row indices]):
                The unique value and the associated rows that go with it.

        Notes:
            If *xcol* is not specified, then the :py:attr:`Data.setas` attribute is used. If this is also
            not set, then all columns are considered.
        """
        _ = self._col_args(xcol=xcol)
        if not _.has_xcol:
            _.xcol = list(range(self.shape[1]))
        search_data = self.data[:, _.xcol]
        if search_data.ndim == 1:
            search_data = np.atleast_2d(search_data).T

        delta = np.atleast_1d(np.array(delta))
        if delta.size != search_data.shape[1]:
            delta = np.append(delta, np.ones(search_data.shape[1]) * delta[0])[: search_data.shape[1]]
        results = dict()
        for ix in range(search_data.shape[0]):
            row = np.atleast_1d(search_data[ix])
            if tuple(row) in results:
                continue
            for iy, (value, dealt) in enumerate(zip(row, delta)):
                # Modify all search data that is close to the current row
                search_data[np.isclose(search_data[:, iy], value, atol=dealt), iy] = value
            matches = np.arange(search_data.shape[0])[np.all(search_data == row, axis=1)]
            results[tuple(row)] = matches.tolist()
        return results

    def remove_duplicates(self, xcol=None, delta=1e-8, strategy="keep first", ycol=None, yerr=None):
        """Find and remove rows with duplicated values of the search column(s).

        Keyword Arguments:
            xcol (index types):
                The column)s) to search for duplicates in.
            delta (float or array):
                The absolute difference(s) to consider equal when comparing floats.
            strategy (str, default *keep first*):
                What to do with duplicated rows. Options are:
                    - *keep first* - the first row is kept, others are discarded
                    - *average* - the duplicate rows are average together.
            ycol, yerr (index types):
                When using an average strategey identifies columns that represent values and uncertainties where
                the proper weighted standard error should be done.

        Returns:
            (dictionary of value:[list of row indices]):
                The unique value and the associated rows that go with it.

        Notes:
            If *ycol* is not specified, then the :py:attr:`Data.setas` attribute is used. If this is also
            not set, then all columns are considered.
        """
        _ = self._col_args(xcol=xcol, ycol=ycol, yerr=yerr, scalar=False)
        dups = self.find_duplicates(xcol=xcol, delta=delta)
        tmp = self.clone
        tmp.data = np.ma.empty((0, self.data.shape[1]))
        for indices in dups.values():
            section = self[indices, :]
            if strategy == "keep first":
                section = section[0, :]
            elif strategy == "average":
                tmp_sec = np.mean(section, axis=0)
                if _.has_ycol and _.has_yerr:  # reclaculate the ycolumns
                    ycol = _.ycol
                    yerr = _.yerr
                    if len(yerr) < len(ycol):
                        yerr += [yerr[0]] * (len(ycol) - len(yerr))
                    for yy, ye in zip(ycol, yerr):
                        stats = DescrStatsW(section[:, yy], weights=1 / (section[:, ye]) ** 2)
                        tmp_sec[yy] = stats.mean
                        tmp_sec[ye] = stats.std_mean
                section = tmp_sec
            else:
                raise RuntimeError(f"Unknown duplicate removal strategy {strategy}")
            tmp += section
        setas = self.setas
        self.data = tmp.data
        self.setas = setas
        return self

    def rolling_window(self, window=7, wrap=True, exclude_centre=False):
        """Iterate with a rolling window section of the data.

        Keyword Arguments:
            window (int):
                Size of the rolling window (must be odd and >= 3)
            wrap (bool):
                Whether to use data from the other end of the array when at one end or the other.
            exclude_centre (odd int or bool):
                Exclude the ciurrent row from the rolling window (defaults to False)

        Yields:
            ndarray:
                Yields with a section of data that is window rows long, each iteration moves the marker
                one row further on.
        """
        if isinstance(exclude_centre, bool) and exclude_centre:
            exclude_centre = 1
        if isinstance(exclude_centre, int_types) and not isinstance(exclude_centre, bool):
            if exclude_centre % 2 == 0:
                raise ValueError("If excluding the centre of the window, this must be an odd number of rows.")
            if window - exclude_centre < 2 or window < 3 or window % 2 == 0:
                raise ValueError(
                    """Window must be at least two bigger than the number of rows exluded from the centre, bigger than
                    3 and odd"""
                )

        hw = int((window - 1) / 2)
        if exclude_centre:
            hc = int((exclude_centre - 1) / 2)

        for i in range(len(self)):
            if i < hw:
                pre_data = self.data[i - hw :]
            else:
                pre_data = np.zeros((0, self.shape[1]))
            if i + 1 > len(self) - hw:
                post_data = self.data[0 : hw - (len(self) - i - 1)]
            else:
                post_data = np.zeros((0, self.shape[1]))
            starti = max(i - hw, 0)
            stopi = min(len(self), i + hw + 1)
            if exclude_centre:  # hacked to stop problems with DataArray concatenation
                tmp = self.clone  # copy all properties
                data = np.vstack((self.data[starti : i - hc], self.data[i + 1 + hc : stopi]))
                tmp.data = np.array(data)  # guarantee an ndarray
                data = tmp.data  # get the DataArray
            else:
                data = self.data[starti:stopi]
            if wrap:
                tmp = self.clone  # copy all properties
                ret = np.vstack((pre_data, data, post_data))
                tmp.data = np.array(ret)  # guarantee an ndarray
                ret = tmp.data  # get the DataArray
            else:
                ret = data
            yield ret

    def search(self, xcol=None, value=None, columns=None, accuracy=0.0):
        """Search the numerica data part of the file for lines that match and returns  the corresponding rows.

        Keyword Arguments:
            xcol (index types, None):
                a Search Column Index. If None (default), use the current setas.x
            value (float, tuple, list or callable, None):
                Value to look for
            columns (index or array of indices or None (default)):
                columns of data to return - none represents all columns.
            accuracy (float):
                Uncertainty to accept when testing equalities

        Returns:
            ndarray: numpy array of matching rows or column values depending on the arguments.

        Note:
            The value is interpreted as follows:

            - a float looks for an exact match
            - a list is a list of exact matches
            - an array or list of booleans (index like Numpy does)
            - a tuple should contain a (min,max) value.
            - A callable object should have accept a float and an array representing the value of
              the search col for the the current row and the entire row.
            - None opens an interactive span selector in a plot window.
        """
        ix = self._search_index(xcol, value, accuracy)
        if columns is None:  # Get the whole slice
            data = self.data[ix, :]
        else:
            columns = self.find_col(columns)
            if not isinstance(columns, list):
                data = self.data[ix, columns]
            else:
                data = self.data[ix, columns[0]]
                for c in columns[1:]:
                    data = np.column_stack((data, self.data[ix, c]))
        return data

    def section(self, **kargs):
        """Assuming data has x,y or x,y,z coordinates, return data from a section of the parameter space.

        Keyword Arguments:
            x (float, tuple, list or callable):
                x values ,atch this condition are included inth e section
            y (float, tuple, list  or callable):
                y values ,atch this condition are included inth e section
            z (float, tuple,list  or callable):
                z values ,atch this condition are included inth e section
            r (callable): a
            function that takes a tuple (x,y,z) and returns True if the line is to be included in section

        Returns:
            (DataFile):
                A :py:class:`DataFile` like object that includes only those lines from the original that match the
                section specification

        Internally this function is calling :py:meth:`DataFile.search` to pull out matching sections of the data array.
        To extract a 2D section of the parameter space orthogonal to one axis you just specify a condition on that
        axis. Specifying conditions on two axes will return a line of points along the third axis. The final
        keyword parameter allows you to select data points that lie in an arbitrary plane or line. eg::

            d.section(r=lambda x,y,z:abs(2+3*x-2*y)<0.1 and z==2)

        would extract points along the line 2y=3x+2 (note the use of an < operator to avoid floating point rounding
        errors) where the z-co-ordinate is 2.
        """
        cols = self.setas._get_cols()
        tmp = self.clone
        xcol = cols["xcol"] if cols.has_xcol else None
        ycol = cols["ycol"][0] if cols.has_ycol else None
        zcol = cols["zcol"][0] if cols.has_zcol else None

        accuracy = kargs.pop("accuracy", 0.0)

        if "x" in kargs:
            tmp.data = tmp.search(xcol, kargs.pop("x"), accuracy=accuracy)
        if "y" in kargs:
            tmp.data = tmp.search(ycol, kargs.pop("y"), accuracy=accuracy)
        if "z" in kargs:
            tmp.data = tmp.search(zcol, kargs.pop("z"), accuracy=accuracy)
        if "r" in kargs:
            func = lambda x, r: kargs.pop("r")(r[xcol], r[ycol], r[zcol])
            tmp.data = tmp.search(0, func, accuracy=accuracy)

        if kargs:  # Fallback to working with select if nothing else.
            tmp.select(**kargs)
        return tmp

    def select(self, *args, **kargs):
        """Produce a copy of the DataFile with only data rows that match a criteria.

        Args:
            args (various):
                A single positional argument if present is interpreted as follows:

                -   If a callable function is given, the entire row is presented to it. If it evaluates True then that
                    row is selected. This allows arbitrary select operations
                -   If a dict is given, then it and the kargs dictionary are merged and used to select the rows

        Keyword Arguments:
            kargs (various):
                Arbitrary keyword arguments are interpreted as requestion matches against the corresponding
                columns. The keyword argument may have an additional *__operator** appended to it which is interpreted
                as follows:

                -   *eq*  value equals argument value (this is the default test for scalar argument)
                -   *ne*  value doe not equal argument value
                -   *gt*  value doe greater than argument value
                -   *lt*  value doe less than argument value
                -   *ge*  value doe greater than or equal to argument value
                -   *le*  value doe less than or equal to argument value
                -   *between*  value lies between the minimum and maximum values of the argument (the default test
                    for 2-length tuple arguments)
                -   *ibetween*,*ilbetween*,*iubetween* as above but include both,lower or upper values

        Returns:
            (DatFile): a copy the DataFile instance that contains just the matching rows.

        Note:
            if the operator is preceeded by *__not__* then the sense of the test is negated.

            If any of the tests is True, then the row will be selected, so the effect is a logical OR. To
            achieve a logical AND, you can chain two selects together::

                d.select(temp__le=4.2,vti_temp__lt=4.2).select(field_gt=3.0)

            will select rows that have either temp or vti_temp metadata values below 4.2 AND field metadata values
            greater than 3.

            If you need to select on a row value that ends in an operator word, then append
            *__eq* in the keyword name to force the equality test. If the metadata keys to select on are not valid
            python identifiers, then pass them via the first positional dictionary value.

            There is a "magic" column name "_i" which is interpreted as the row numbers of the data.

        Example
            .. plot:: samples/select_example.py
                :include-source:
                :outname: select
        """
        match args:
            case (arg,) if callable(arg):
                kargs["__"] = arg
            case (dict() as arg,):
                kargs.update(arg)
            case _:
                pass

        result = self.clone
        res = np.zeros(len(self), dtype=bool)
        for arg in kargs:
            parts = arg.split("__")
            if parts == ["", ""]:
                func = kargs[arg]
                res = np.logical_or(res, np.array([func(r) for r in self.data]))
                continue
            if len(parts) == 1 or parts[-1] not in operator:
                parts.append("eq")
            if len(parts) > 2 and parts[-2] == "not":
                end = -2
                negate = True
            else:
                end = -1
                negate = False
            if parts[0] == "_i":
                res = np.logical_or(res, np.logical_xor(negate, operator[parts[-1]](self.data.i, kargs[arg])))
            else:
                col = "__".join(parts[:end])
                res = np.logical_or(res, np.logical_xor(negate, operator[parts[-1]](self.column(col), kargs[arg])))
        result.data = self.data[res, :]
        return result

    def sort(self, *order, **kargs):
        """Sort the data by column name.

        Arguments:
            order (column index or list of indices or callable function):
                One or more sort order keys.

        Keyword Arguments:
            reverse (boolean):
                If true, the sorted array isreversed.

        Returns:
            (self):
                A copy of the :py:class:`DataFile` sorted object

        Notes:
            Sorts in place and returns a copy of the sorted data object fo chaining methods.

            If the argument is a callable function then it should take a two tuple arguments and
            return +1,0,-1 depending on whether the first argument is bigger, equal or smaller. Otherwise
            if the argument is interpreted as a column index. If a single argument is supplied, then it may be
            a list of column indices. If no sort orders are supplied then the data is sorted by the
            :py:attr:`DataFile.setas` attribute or if that is not set, then order of the columns in the data.
        """
        reverse = kargs.pop("reverse", False)
        order = list(order)
        setas = self.setas.clone
        ch = copy.copy(self.column_headers)
        if not order:
            if self.setas.cols["xcol"] is not None:
                order = [self.setas.cols["xcol"]]
            order.extend(self.setas.cols["ycol"])
            order.extend(self.setas.cols["zcol"])
        if not order:  # Ok, no setas here then
            order = None
        elif len(order) == 1:
            order = order[0]

        if order is None:
            order = list(range(len(self.column_headers)))
        recs = self.records
        match order:
            case _ if callable(order):
                d = sorted(recs, cmp=order)
            case int() | str() | _pattern_type():
                order = [recs.dtype.names[self.find_col(order)]]
                d = np.sort(recs, order=order)
            case Iterable():
                order = [recs.dtype.names[self.find_col(x)] for x in order]
                d = np.sort(recs, order=order)
            case _:
                raise KeyError(f"Unable to work out how to sort by a {type(order)}")

        self.data = d.view(dtype=self.dtype).reshape(len(self), len(self.column_headers))
        if reverse:
            self.data = self.data[::-1]
        self.data._setas = setas
        self.column_headers = ch
        return self

    def split(self, *args, final="files"):
        """Recursively splits the current DataFile into a :py:class:`Stoner.Folders.DataFolder`.

        Args:
            *args (column index or function):
                Each argument is used in turn to find key values for the files in the DataFolder

        Keyword Arguments:
            final (str):
                Controls whether the final argument plaes the files in the DataFolder (default: "files") or in
                groups ("groups")

        Returns:
            Stoner.Folders.DataFolder:
                A :py:class:`Stoner.Folders.DataFolder` object containing the individual
                :py:class:`AnalysisMixin` objects

        Note:
            Creates a DataFolder of  DataFiles where each one contains the rows from the original object which
            had the same value of a given column(s) or function.


            On each iteration the first argument is called. If it is a column type then rows which amtch each unique
            value are collated together and made into a separate file. If the argument is a callable, then it is
            called for each row, passing the row as a single 1D array and the return result is used to group lines
            together. The return value should be hashable.

            Once this is done and the :py:class:`Stoner.Folders.DataFolder` exists, if there are remaining argument,
            then the method is called recusivelyt for each file and the resulting DataFolder added into the root
            DataFolder and the file is removed.

            Thus, when all of the arguments are evaluated, the resulting DataFolder is a multi-level tree.

            .. warning::

                There has been a change in the arguments for the split function  from version 0.8 of the Stoner
                Package.
        """
        from Stoner import DataFolder

        if not args:
            xcol = self.setas._get_cols("xcol")
        else:
            args = list(args)
            xcol = args.pop(0)
        data = dict()

        match xcol:
            case int() | str() | _pattern_type():
                for val in np.unique(self.column(xcol)):
                    newfile = self.clone
                    newfile.filename = f"{self.column_headers[self.find_col(xcol)]}={val} {self.filename}"
                    newfile.data = self.search(xcol, val)
                    data[val] = newfile
            case _ if callable(xcol):
                try:  # Try to call function with all data in one go
                    keys = xcol(self.data)
                    if not isiterable(keys):
                        keys = [keys] * len(self)
                except Exception:  # pylint: disable=W0703  # Ok try instead to do it row by row
                    keys = [xcol(r) for r in self]
                if not isiterable(keys) or len(keys) != len(self):
                    raise RuntimeError("Not returning an index of keys")
                keys = np.array(keys)
                for key in np.unique(keys):
                    data[key] = self.clone
                    data[key].data = self.data[keys == key, :]
                    data[key].filename = f"{xcol.__name__}={key} {self.filename}"
                    data[key].setas = self.setas
            case _:
                raise NotImplementedError(f"Unable to split a file with an argument of type {type(xcol)}")
        out = DataFolder(nolist=True, setas=self.setas)
        for k, f in data.items():
            if args:
                out.add_group(k)
                out.groups[k] = f.split(*args)
            else:
                if final == "files":
                    out += f
                elif final == "groups":
                    out.add_group(k)
                    f.filename = self.filename
                    out.groups[k] += f
                else:
                    raise ValueError(f"{final} not recognised as a valid value for final")
        return out

    def unique(self, col, return_index=False, return_inverse=False):
        """Return the unique values from the specified column - pass through for numpy.unique.

        Args:
            col (index):
                Column to look for unique values in

        Keyword Arguments:
            return_index (bool):
                Pass through to :py:func:`np.unique`
            reverse (bool):
                Pass through to :py:func:`np.unique`

        Returns:
            (1D array):
                Array of unique values from the column.
        """
        return np.unique(self.column(col), return_index, return_inverse)
