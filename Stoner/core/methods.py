# -*- coding: utf-8 -*-
"""Mxinin classes for DataFile objects."""

import copy
from collections.abc import Iterable
import io
import pathlib
import re
from sys import float_info


import numpy as np
from numpy import ma
from statsmodels.stats.weightstats import DescrStatsW

from .array import DataArray
from ..compat import _pattern_type, int_types
from ..tools import all_type, isiterable, operator, format_error
from ..tools.widgets import RangeSelect
from ..tools.file import file_dialog, best_saver

try:
    import pandas as pd
except ImportError:
    pd = None


def search_index(datafile, xcol=None, value=None, accuracy=0.0, invert=False):
    """Return an array of booleans for indexing matching rows for use with search method."""
    _ = datafile._col_args(scalar=False, xcol=xcol)
    x = datafile.find_col(_.xcol, force_list=True)[0]  # Workaround in newer numpy
    match value:
        case int() | float():
            ix = np.isclose(datafile.data[:, x], value, atol=accuracy)
        case (_, _):
            (low, u) = (min(value), max(value))
            low -= accuracy
            u += accuracy
            v = datafile.data[:, x]
            low = np.ones_like(v) * low
            u = np.ones_like(v) * u
            ix = np.logical_and(v > low, v <= u)
        case list() | np.ndarray() if all_type(value, bool) and len(value) <= len(datafile):
            if len(value) < len(datafile):  # Expand array if necessary
                ix = np.append(value, [False] * len(datafile) - len(value))
            else:
                ix = np.array(value)
        case list() | np.ndarray():
            ix = np.zeros(len(datafile), dtype=bool)
            for v in value:
                ix = np.logical_or(ix, search_index(datafile, xcol, v, accuracy))
        case _ if callable(value):
            ix = np.array([value(r[x], r) for r in datafile], dtype=bool)
        case _ if value is None:
            selector = RangeSelect()
            ix = selector(datafile, x, accuracy)
        case _:
            raise RuntimeError(f"Unknown search value type {value}")

    ix = np.logical_xor(invert, ix)
    if ix.ndim > 1:
        ix = ix[:, 0]
    return ix


def asarray(datafile):
    """Provide a consistent way to get at the underlying array data."""
    return datafile.data


def closest(datafile, value, xcol=None):
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
    _ = datafile._col_args(xcol=xcol)
    xdata = np.abs(datafile // _.xcol - value)
    i = int(xdata.argmin())
    return datafile[i]


def column(datafile, col):
    """Extract one or more columns of data from the datafile.

    Args:
        col (int, string, list or re):
            is the column index as defined for :py:meth:`DataFile.find_col`

    Returns:
        (ndarray):
            One or more columns of data as a :py:class:`numpy.ndarray`.
    """
    return datafile.data[:, datafile.find_col(col)]


def find_col(datafile, col, force_list=False):
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
    return datafile.data._setas.find_col(col, force_list)


def find_duplicates(datafile, xcol=None, delta=1e-8):
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
    _ = datafile._col_args(xcol=xcol)
    if not _.has_xcol:
        _.xcol = list(range(datafile.shape[1]))
    search_data = datafile.data[:, _.xcol]
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


def remove_duplicates(datafile, xcol=None, delta=1e-8, strategy="keep first", ycol=None, yerr=None):
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
    _ = datafile._col_args(xcol=xcol, ycol=ycol, yerr=yerr, scalar=False)
    dups = datafile.find_duplicates(xcol=xcol, delta=delta)
    tmp = datafile.clone
    tmp.data = np.ma.empty((0, datafile.data.shape[1]))
    for indices in dups.values():
        section = datafile[indices, :]
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
    setas = datafile.setas
    datafile.data = tmp.data
    datafile.setas = setas
    return datafile


def rolling_window(datafile, window=7, wrap=True, exclude_centre=False):
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

    for i in range(len(datafile)):
        if i < hw:
            pre_data = datafile.data[i - hw :]
        else:
            pre_data = np.zeros((0, datafile.shape[1]))
        if i + 1 > len(datafile) - hw:
            post_data = datafile.data[0 : hw - (len(datafile) - i - 1)]
        else:
            post_data = np.zeros((0, datafile.shape[1]))
        starti = max(i - hw, 0)
        stopi = min(len(datafile), i + hw + 1)
        if exclude_centre:  # hacked to stop problems with DataArray concatenation
            tmp = datafile.clone  # copy all properties
            data = np.vstack((datafile.data[starti : i - hc], datafile.data[i + 1 + hc : stopi]))
            tmp.data = np.array(data)  # guarantee an ndarray
            data = tmp.data  # get the DataArray
        else:
            data = datafile.data[starti:stopi]
        if wrap:
            tmp = datafile.clone  # copy all properties
            ret = np.vstack((pre_data, data, post_data))
            tmp.data = np.array(ret)  # guarantee an ndarray
            ret = tmp.data  # get the DataArray
        else:
            ret = data
        yield ret


def search(datafile, xcol=None, value=None, columns=None, accuracy=0.0):
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
    ix = search_index(datafile, xcol, value, accuracy)
    if columns is None:  # Get the whole slice
        data = datafile.data[ix, :]
    else:
        columns = datafile.find_col(columns)
        if not isinstance(columns, list):
            data = datafile.data[ix, columns]
        else:
            data = datafile.data[ix, columns[0]]
            for c in columns[1:]:
                data = np.column_stack((data, datafile.data[ix, c]))
    return data


def section(datafile, **kargs):
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
    cols = datafile.setas._get_cols()
    tmp = datafile.clone
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


def select(datafile, *args, **kargs):
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

    result = datafile.clone
    res = np.zeros(len(datafile), dtype=bool)
    for arg in kargs:
        parts = arg.split("__")
        if parts == ["", ""]:
            func = kargs[arg]
            res = np.logical_or(res, np.array([func(r) for r in datafile.data]))
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
            res = np.logical_or(res, np.logical_xor(negate, operator[parts[-1]](datafile.data.i, kargs[arg])))
        else:
            col = "__".join(parts[:end])
            res = np.logical_or(res, np.logical_xor(negate, operator[parts[-1]](datafile.column(col), kargs[arg])))
    result.data = datafile.data[res, :]
    return result


def sort(datafile, *order, **kargs):
    """Sort the data by column name.

    Arguments:
        order (column index or list of indices or callable function):
            One or more sort order keys.

    Keyword Arguments:
        reverse (boolean):
            If true, the sorted array isreversed.

    Returns:
        (datafile):
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
    setas = datafile.setas.clone
    ch = copy.copy(datafile.column_headers)
    if not order:
        if datafile.setas.cols["xcol"] is not None:
            order = [datafile.setas.cols["xcol"]]
        order.extend(datafile.setas.cols["ycol"])
        order.extend(datafile.setas.cols["zcol"])
    if not order:  # Ok, no setas here then
        order = None
    elif len(order) == 1:
        order = order[0]

    if order is None:
        order = list(range(len(datafile.column_headers)))
    recs = datafile.records
    match order:
        case _ if callable(order):
            d = sorted(recs, cmp=order)
        case int() | str() | _pattern_type():
            order = [recs.dtype.names[datafile.find_col(order)]]
            d = np.sort(recs, order=order)
        case Iterable():
            order = [recs.dtype.names[datafile.find_col(x)] for x in order]
            d = np.sort(recs, order=order)
        case _:
            raise KeyError(f"Unable to work out how to sort by a {type(order)}")

    datafile.data = d.view(dtype=datafile.dtype).reshape(len(datafile), len(datafile.column_headers))
    if reverse:
        datafile.data = datafile.data[::-1]
    datafile.data._setas = setas
    datafile.column_headers = ch
    return datafile


def split(datafile, *args, final="files"):
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
        xcol = datafile.setas._get_cols("xcol")
    else:
        args = list(args)
        xcol = args.pop(0)
    data = dict()

    match xcol:
        case int() | str() | _pattern_type():
            for val in np.unique(datafile.column(xcol)):
                newfile = datafile.clone
                newfile.filename = f"{datafile.column_headers[datafile.find_col(xcol)]}={val} {datafile.filename}"
                newfile.data = datafile.search(xcol, val)
                data[val] = newfile
        case _ if callable(xcol):
            try:  # Try to call function with all data in one go
                keys = xcol(datafile.data)
                if not isiterable(keys):
                    keys = [keys] * len(datafile)
            except Exception:  # pylint: disable=W0703  # Ok try instead to do it row by row
                keys = [xcol(r) for r in datafile]
            if not isiterable(keys) or len(keys) != len(datafile):
                raise RuntimeError("Not returning an index of keys")
            keys = np.array(keys)
            for key in np.unique(keys):
                data[key] = datafile.clone
                data[key].data = datafile.data[keys == key, :]
                data[key].filename = f"{xcol.__name__}={key} {datafile.filename}"
                data[key].setas = datafile.setas
        case _:
            raise NotImplementedError(f"Unable to split a file with an argument of type {type(xcol)}")
    out = DataFolder(nolist=True, setas=datafile.setas)
    for k, f in data.items():
        if args:
            out.add_group(k)
            out.groups[k] = f.split(*args)
        else:
            if final == "files":
                out += f
            elif final == "groups":
                out.add_group(k)
                f.filename = datafile.filename
                out.groups[k] += f
            else:
                raise ValueError(f"{final} not recognised as a valid value for final")
    return out


def unique(datafile, col, return_index=False, return_inverse=False):
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
    return np.unique(datafile.column(col), return_index, return_inverse)


def add_column(datafile, column_data, header=None, index=None, func_args=None, replace=False, setas=None):
    """Append a column of data or inserts a column to a datafile instance.

    Args:
        column_data (:py:class:`numpy.array` or list or callable):
            Data to append or insert or a callable function that will generate new data

    Keyword Arguments:
        header (string):
            The text to set the column header to,
            if not supplied then defaults to 'col#'
        index (index type):
            The  index (numeric or string) to insert (or replace) the data
        func_args (dict):
            If column_data is a callable object, then this argument
            can be used to supply a dictionary of function arguments to the callable object.
        replace (bool):
            Replace the data or insert the data (default)
        setas (str):
            Set the type of column (x,y,z data etc - see :py:attr:`Stoner.Core.DataFile.setas`)

    Returns:
        datafile:
            The :py:class:`DataFile` instance with the additional column inserted.

    Note:
        Like most :py:class:`DataFile` methods, this method operates in-place in that it also modifies
        the original DataFile Instance as well as returning it.
    """
    if index is None or isinstance(index, bool) and index:  # Enure index is set
        index = datafile.shape[1]
        replace = False
    elif isinstance(index, int_types) and index == datafile.shape[1]:
        replace = False
    else:
        index = datafile.find_col(index)

    # Sort out the data and get it into an array of values.
    if isinstance(column_data, list):
        column_data = np.array(column_data)

    if isinstance(column_data, DataArray) and header is None:
        header = column_data.column_headers

    if isinstance(column_data, np.ndarray):
        np_data = column_data
    elif callable(column_data):
        if isinstance(func_args, dict):
            new_data = [column_data(x, **func_args) for x in datafile]
        else:
            new_data = [column_data(x) for x in datafile]
        np_data = np.array(new_data)
    else:
        return NotImplemented

    # Sort out the sizes of the arrays
    np_data = np.atleast_2d(np_data).T
    cl, cw = np_data.shape

    # Make setas
    setas = "." * cw if setas is None else setas

    if isiterable(setas) and len(setas) == cw:
        for s in setas:
            if s not in ".-xyzuvwdefpqr":
                raise TypeError(
                    f"setas parameter should be a string or list of letter in the set xyzdefuvw.-, not {setas}"
                )
    else:
        raise TypeError(
            f"""setas parameter should be a string or list of letter the same length as the number of columns
            being added in the set xyzdefuvw.-, not {setas}"""
        )

    # Make sure our current data is at least 2D and get its size
    match datafile.data.shape:
        case (_,):
            datafile.data = np.atleast_2d(datafile.data).T
        case (_, _):
            (dr, dc) = datafile.data.shape
        case _ if not datafile.data.shape:
            datafile.data = np.array([[]])
            (dr, dc) = (0, 0)
        case _:
            raise ValueError("Data should be 1 or 2 dimensional")

    # Expand either our current data or new data to have the same number of rows
    if cl > dr and dc * dr > 0:  # Existing data is finite and too short
        datafile.data = DataArray(np.append(datafile.data, np.zeros((cl - dr, dc)), 0), setas=datafile.setas.clone)
    elif cl < dr:  # New data is too short
        np_data = np.append(np_data, np.zeros((dr - cl, cw)))
        if np_data.ndim == 1:
            np_data = np.atleast_2d(np_data).T
    elif dc == 0:  # Existing data has no width - replace with cl,0
        datafile.data = DataArray(np.zeros((cl, 0)))
    elif dr == 0:  # Existing data has no rows - expand existing data with zeros to have right length
        datafile.data = DataArray(np.append(datafile.data, np.zeros((cl, dr)), axis=0), setas=datafile.setas.clone)

    # If not replacing, then add extra columns to existing data.
    if not replace:
        columns = copy.copy(datafile.column_headers)
        old_setas = datafile.setas.clone
        if index == datafile.data.shape[1]:  # appending column
            datafile.data = DataArray(np.append(datafile.data, np_data, axis=1), setas=datafile.setas.clone)
        else:
            datafile.data = DataArray(
                np.append(
                    datafile.data[:, :index],
                    np.append(np.zeros_like(np_data), datafile.data[:, index:], axis=1),
                    axis=1,
                ),
                setas=datafile.setas.clone,
            )
        for ix in range(0, index):
            datafile.column_headers[ix] = columns[ix]
            datafile.setas[ix] = old_setas[ix]
        for ix in range(index, dc):
            datafile.column_headers[ix + cw] = columns[ix]
            datafile.setas[ix + cw] = old_setas[ix]
    # Check that we don't need to expand to overwrite with the new data
    if index + cw > datafile.shape[1]:
        datafile.data = DataArray(
            np.append(datafile.data, np.zeros((datafile.data.shape[0], datafile.data.shape[1] - index + cw)), axis=1),
            setas=datafile.setas.clone,
        )

    # Put the data into the array
    datafile.data[:, index : index + cw] = np_data

    if header is None:  # This will fix the header if not defined.
        header = [f"Column {ix}" for ix in range(index, index + cw)]
    if isinstance(header, str):
        header = [header]
    if len(header) != cw:
        header.extend(["Column {ix}" for x in range(index, index + cw)])
    for ix, (hdr, s) in enumerate(zip(header, setas)):
        datafile.column_headers[ix + index] = hdr
        datafile.setas[index + ix] = s

    datafile.labels = (
        datafile.labels[:index]
        + datafile.column_headers[index : len(datafile.column_headers) - len(datafile.labels) + index]
        + datafile.labels[index:]
    )

    return datafile


def columns(datafile, not_masked=False, reset=False):
    """Iterate over the columns of data int he datafile.

    Keyword Args:
        no_masked (bool):
            Only iterate over columns that don't have masked elements
        reset (bool):
            If true then reset the iterator (immediately stops the current iteration without returning any data)./

    Yields:
        1D array: Returns the next column of data.
    """
    for ix, col in enumerate(datafile.data.T):
        if not_masked and ma.is_masked(col):
            continue
        if reset:
            return
        else:
            yield datafile.column(ix)


def del_column(datafile, col=None, duplicates=False):
    """Delete a column from the current :py:class:`DataFile` object.

    Args:
        col (int, string, iterable of booleans, list or re):
            is the column index as defined for :py:meth:`DataFile.find_col` to the column to be deleted

    Keyword Arguments:
        duplicates (bool):
            (default False) look for duplicated columns

    Returns:
        datafile:
            The :py:class:`DataFile` object with the column deleted.

    Note:
        - If duplicates is True and col is None then all duplicate columns are removed,
        - if col is not None and duplicates is True then all duplicates of the specified column are removed.
        - If duplicates is False and *col* is either None or False then all masked coplumns are deleeted. If
            *col* is True, then all columns that are not set i the :py:attr:`setas` attrobute are deleted.
        - If col is a list (duplicates should not be None) then the all the matching columns are found.
        - If col is an iterable of booleans, then all columns whose elements are False are deleted.
        - If col is None and duplicates is None, then all columns with at least one elelemtn masked
                will be deleted
    """
    if duplicates:
        ch = datafile.column_headers
        dups = []
        if col is None:
            for i, chi in enumerate(ch):
                if chi in ch[i + 1 :]:
                    dups.append(ch.index(chi, i + 1))
        else:
            col = ch[datafile.find_col(col)]
            i = ch.index(col)
            while True:
                try:
                    i = ch.index(col, i + 1)
                    dups.append(i)
                except ValueError:
                    break
        return datafile.del_column(dups, duplicates=False)
    if col is None or (isinstance(col, bool) and not col):  # Without defining col we just compress by the mask
        datafile.data = ma.mask_cols(datafile.data)
        t = DataArray(datafile.column_headers)
        t.mask = datafile.mask[0]
        datafile.column_headers = list(ma.compressed(t))
        datafile.data = ma.compress_cols(datafile.data)
    elif isinstance(col, bool) and col:  # Without defining col we just compress by the mask
        ch = [datafile.column_headers[ix] for ix, v in enumerate(datafile.setas.set) if v]
        setas = [datafile.setas[ix] for ix, v in enumerate(datafile.setas.set) if v]
        datafile.data = datafile.data[:, datafile.setas.set]
        datafile.setas = setas
        datafile.column_headers = ch
    elif isiterable(col) and all_type(col, bool):  # If col is an iterable of booleans then we index by that
        col = ~np.array(col)
        new_setas = np.array(datafile.setas)[col]
        new_column_headers = np.array(datafile.column_headers)[col]
        datafile.data = datafile.data[:, col]
        datafile.setas = new_setas
        datafile.column_headers = new_column_headers
    else:  # Otherwise find individual columns
        c = datafile.find_col(col)
        ch = datafile.column_headers
        datafile.data = DataArray(np.delete(datafile.data, c, 1), mask=np.delete(datafile.data.mask, c, 1))
        if isinstance(c, list):
            c.sort(reverse=True)
        else:
            c = [c]
        for cl in c:
            del ch[cl]
        datafile.column_headers = ch
    return datafile


def del_rows(datafile, col=None, val=None, invert=False):
    """Search in the numerica data for the lines that match and deletes the corresponding rows.

    Args:
        col (list,slice,int,string, re, callable or None):
            Column containing values to search for.
        val (float or callable):
            Specifies rows to delete. Maybe:
                -   None - in which case the *col* argument is used to identify rows to be deleted,
                -   a float in which case rows whose columncol = val are deleted
                -   or a function - in which case rows where the function evaluates to be true are deleted.
                -   a tuple, in which case rows where column col takes value between the minimum and maximum of
                    the tuple are deleted.

    Keyword Arguments:
        invert (bool):
            Specifies whether to invert the logic of the test to delete a row. If True, keep the rows
            that would have been deleted otherwise.

    Returns:
        datafile:
            The current :py:class:`DataFile` object

    Note:
        If col is None, then all rows with masked data are deleted

        if *col* is callable then it is passed each row as a :py:class:`DataArray` and if it returns
        True, then the row will be deleted or kept depending on the value of *invert*.

        If *val* is a callable it should take two arguments - a float and a
        list. The float is the value of the current row that corresponds to column col abd the second
        argument is the current row.

    Todo:
        Implement val is a tuple for deletinging in a range of values.
    """
    if col is None:
        datafile.data = ma.compress_rows(datafile.data)
    else:
        if isinstance(col, slice) and val is None:  # delete rows with a slice to make a list of indices
            indices = col.indices(len(datafile))
            col = list(range(*indices))
        elif callable(col) and val is None:  # Delete rows usinga callalble taking the whole row
            col = [r.i for r in datafile.rows() if col(r)]
        elif isiterable(col) and all_type(col, bool):  # Delete rows by a list of booleans
            if len(col) < len(datafile):
                col.extend([False] * (len(datafile) - len(col)))
            datafile.data = datafile.data[col]
            return datafile
        if isiterable(col) and all_type(col, int_types) and val is None and not invert:
            col.sort(reverse=True)
            for c in col:
                datafile.del_rows(c)
        elif isinstance(col, list) and all_type(col, int_types) and val is None and invert:
            for i in range(len(datafile) - 1, -1, -1):
                if i not in col:
                    datafile.del_rows(i)
        elif isinstance(col, int_types) and val is None and not invert:
            tmp_mask = datafile.mask
            tmp_setas = datafile.data._setas.clone
            datafile.data = np.delete(datafile.data, col, 0)
            datafile.data.mask = np.delete(tmp_mask, col, 0)
            datafile.data._setas = tmp_setas
        elif isinstance(col, int_types) and val is None and invert:
            datafile.del_rows([c], invert=invert)
        else:
            col = datafile.find_col(col)
            d = datafile.column(col)
            if callable(val):
                rows = np.nonzero(
                    [(bool(val(x[col], x) and bool(x[col] is not ma.masked)) != invert) for x in datafile]
                )[0]
            elif isinstance(val, float):
                rows = np.nonzero([bool(x == val) != invert for x in d])[0]
            elif isiterable(val) and len(val) == 2:
                (upper, lower) = (max(list(val)), min(list(val)))
                rows = np.nonzero([bool(lower <= x <= upper) != invert for x in d])[0]
            else:
                raise SyntaxError("If val is specified it must be a float,callable, or iterable object of length 2")
            tmp_mask = datafile.mask
            tmp_setas = datafile.data._setas.clone
            datafile.data = np.delete(datafile.data, rows, 0)
            datafile.data.mask = np.delete(tmp_mask, rows, 0)
            datafile.data._setas = tmp_setas
    return datafile


def dir(datafile, pattern=None):
    """Return a list of keys in the metadata, filtering with a regular expression if necessary.

    Keyword Arguments:
        pattern (string or re):
            is a regular expression or None to list all keys

    Returns:
        list:
            A list of metadata keys.
    """
    if pattern is None:
        return list(datafile.metadata.keys())
    if isinstance(pattern, _pattern_type):
        test = pattern
    else:
        test = re.compile(pattern)
    possible = [x for x in datafile.metadata.keys() if test.search(x)]
    return possible


def get_filename(datafile, mode):
    """Force the user to choose a new filename using a system dialog box.

    Args:
        mode (string):
            The mode of file operation to be used when calling the dialog box

    Returns:
        str:
            The new filename

    Note:
        The filename attribute of the current instance is updated by this method as well.
    """
    datafile.filename = file_dialog(mode, datafile.filename, datafile.get("Loaded as", "DataFile"), datafile.__class__)
    return datafile.filename


def insert_rows(datafile, row, new_data):
    """Insert new_data into the data array at position row. This is a wrapper for numpy.insert.

    Args:
        row (int):
            Data row to insert into
        new_data (numpy array):
            An array with an equal number of columns as the main data array containing the new row(s) of
            data to insert

    Returns:
        datafile:
            A copy of the modified :py:class:`DataFile` object
    """
    datafile.data = np.insert(datafile.data, row, new_data, 0)
    return datafile


def rename(datafile, old_col, new_col):
    """Rename columns without changing the underlying data.

    Args:
        old_col (string, int, re):
            Old column index or name (using standard rules)
        new_col (string):
            New name of column

    Returns:
        datafile:
            A copy of the modified :py:class:`DataFile` instance
    """
    old_col = datafile.find_col(old_col)
    datafile.column_headers[old_col] = new_col
    return datafile


def reorder_columns(datafile, cols, headers_too=True, setas_too=True):
    """Construct a new data array from the original data by assembling the columns in the order given.

    Args:
        cols (list of column indices):
            (referred to the oriignal data set) from which to assemble the new data set
        headers_too (bool):
            Reorder the column headers in the same way as the data (defaults to True)
        setas_too (bool):
            Reorder the column assignments in the same way as the data (defaults to True)

    Returns:
        datafile:
            A copy of the modified :py:class:`DataFile` object
    """
    if headers_too:
        column_headers = [datafile.column_headers[datafile.find_col(x)] for x in cols]
    else:
        column_headers = datafile.column_headers
    if setas_too:
        setas = [datafile.setas[datafile.find_col(x)] for x in cols]
    else:
        setas = datafile.setas.clone

    newdata = np.atleast_2d(datafile.data[:, datafile.find_col(cols.pop(0))])
    for col in cols:
        newdata = np.append(newdata, np.atleast_2d(datafile.data[:, datafile.find_col(col)]), axis=0)
    datafile.data = DataArray(np.transpose(newdata))
    datafile.setas = setas
    datafile.column_headers = column_headers
    return datafile


def rows(datafile, not_masked=False, reset=False):
    """Iterate over rows of data.

    Keyword Arguments:
        not_masked(bool):
            If a row is masked and this is true, then don't return this row.
        reset (bool):
            If true then reset the iterator (immediately stops the current iteration without returning any data)./

    Yields:
        1D array: Returns the next row of data
    """
    for ix, row in enumerate(datafile.data):
        if not isinstance(row, DataArray):
            row = DataArray([row])
            row.i = ix
            row.setas = datafile.setas
        if ma.is_masked(row) and not_masked:
            continue
        if reset:
            return
        else:
            yield row


def save(datafile, filename=None, **kargs):
    """Save a string representation of the current DataFile object into the file 'filename'.

    Args:
        filename (string, bool or None):
            Filename to save data as, if this is None then the current filename for the object is used. If this
            is not set, then then a file dialog is used. If filename is False then a file dialog is forced.
        as_loaded (bool,str):
            If True, then the *Loaded as* key is inspected to see what the original class of the DataFile was
            and then this class' save method is used to save the data. If a str then
            the keyword value is interpreted as the name of a subclass of the the current DataFile.

    Returns:
        datafile:
            The current :py:class:`DataFile` object
    """
    as_loaded = kargs.pop("as_loaded", False)
    if filename is None:
        filename = datafile.filename
    if filename is None or (isinstance(filename, bool) and not filename):
        # now go and ask for one
        filename = file_dialog("w", datafile.filename, type(datafile), datafile.__class__)
        if not filename:
            raise RuntimeError("Cannot get filename to save")
    if as_loaded:
        loadtype = datafile.get("Loaded as", "DataFile")
        if loadtype != "DataFile":
            saver = best_saver(filename, loadtype)
            ret = saver(datafile, filename)
            datafile.filename = ret.filename
            return datafile
    # Normalise the extension to ensure it's something we like...
    filename, ext = pathlib.Path(filename).with_suffix(""), pathlib.Path(filename).suffix
    saver = best_saver(filename, name=datafile.get("Loaded as", "DataFile"), what="Data")
    if ext not in saver.patterns:
        ext = saver.patterns[0]
    filename = f"{filename}.{ext}"
    header = ["TDI Format 1.5"]
    header.extend(datafile.column_headers[: datafile.data.shape[1]])
    header = "\t".join(header)
    mdkeys = sorted(datafile.metadata)
    if len(mdkeys) > len(datafile):
        mdremains = mdkeys[len(datafile) :]
        mdkeys = mdkeys[0 : len(datafile)]
    else:
        mdremains = []
    mdtext = np.array([datafile.metadata.export(k) for k in mdkeys])
    if len(mdtext) < len(datafile):
        mdtext = np.append(mdtext, np.zeros(len(datafile) - len(mdtext), dtype=str))
    data_out = np.column_stack([mdtext, datafile.data])
    fmt = ["%s"] * data_out.shape[1]
    with io.open(filename, "w", errors="replace", encoding="utf-8") as f:
        np.savetxt(f, data_out, fmt=fmt, header=header, delimiter="\t", comments="")
        for k in mdremains:
            f.write(datafile.metadata.export(k) + "\n")  # (str2bytes(datafile.metadata.export(k) + "\n"))

    datafile.filename = filename
    return datafile


def swap_column(datafile, *swp, **kargs):
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
        datafile:
            A copy of the modified :py:class:`DataFile` objects

    Note:
        If swp is a list, then the function is called recursively on each
        element of the list. Thus in principle the @swp could contain
        lists of lists of tuples
    """
    datafile.data.swap_column(*swp, **kargs)
    return datafile


def to_pandas(datafile):
    """Create a pandas DataFrame from a :py:class:`Stoner.Data` object.

    Notes:
        In addition to transferring the numerical data, the DataFrame's columns are set to
        a multi-level index of the :py:attr:`Stoner.Data.column_headers` and :py:attr:`Stoner.Data.setas`
        values. A pandas DataFrame extension attribute, *metadata* is registered and is used to store
        the metada from the :py:class:1Stoner.Data` object. This pandas extension attribute is in fact a trivial
        subclass of the :py:class:`Stoner.core.TypeHintedDict`.

        The inverse operation can be carried out simply by passing a DataFrame into the copnstructor of the
        :py:class:`Stoner.Data` object.

    Raises:
        **NotImplementedError** if pandas didn't import correctly.
    """
    if pd is None:
        raise NotImplementedError("Pandas not available")
    idx = pd.MultiIndex.from_tuples(zip(*[datafile.column_headers, datafile.setas]), names=("Headers", "Setas"))
    df = pd.DataFrame(datafile.data, columns=idx)
    df.metadata.update(datafile.metadata)
    return df

def format(datafile, key, **kargs):
    r"""Return the contents of key pretty formatted using :py:func:`format_error`.

    Args:
        fmt (str): Specify the output format, options are:

            *  "text" - plain text output
            * "latex" - latex output
            * "html" - html entities

        escape (bool):
            Specifies whether to escape the prefix and units for unprintable characters in non
            text formats )default False)
        mode (string):
            If "float" (default) the number is formatted as is, if "eng" the value and error is converted
            to the next samllest power of 1000 and the appropriate SI index appended. If mode is "sci" then a
            scientific, i.e. mantissa and exponent format is used.
        units (string):
            A suffix providing the units of the value. If si mode is used, then appropriate si
            prefixes are prepended to the units string. In LaTeX mode, the units string is embedded in \mathrm
        prefix (string):
            A prefix string that should be included before the value and error string. in LaTeX mode this is
            inside the math-mode markers, but not embedded in \mathrm.

    Returns:
        A pretty string representation.

    The if key="key", then the value is datafile["key"], the error is datafile["key err"], the default prefix is
    datafile["key label"]+"=" or "key=", the units are datafile["key units"] or "".

    """
    mode = kargs.pop("mode", "float")
    units = kargs.pop("units", datafile.get(key + " units", ""))
    prefix = kargs.pop("prefix", f"{datafile.get(key + ' label', f'{key}')} = ")
    latex = kargs.pop("latex", False)
    fmt = kargs.pop("fmt", "latex" if latex else "text")
    escape = kargs.pop("escape", False)

    try:
        value = float(datafile[key])
    except (ValueError, TypeError) as err:
        raise KeyError(f"{key} should be a floating point value of the metadata not a {type(datafile[key])}.") from err
    try:
        error = float(datafile[f"{key} err"])
    except (TypeError, KeyError):
        error = float_info.epsilon
    return format_error(value, error, fmt=fmt, mode=mode, units=units, prefix=prefix, scape=escape)
