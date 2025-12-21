"""Channel column operations functions for analysis code."""

from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..compat import _pattern_type
from ..tools import all_type, isiterable
from ..tools.typing import Data, Index


def _do_error_calc(datafile: Data, col_a: Index, col_b: Index, error_type: str = "relative") -> NDArray[np.float64]:
    """Do an error calculation."""
    col_a = datafile.find_col(col_a)
    if (
        isinstance(col_a, (list, tuple)) and isinstance(col_b, (list, tuple)) and len(col_a) == 2 and len(col_b) == 2
    ):  # Error columns on
        (col_a, e1) = col_a
        (col_b, e2) = col_b
        e1data = __get_math_val(datafile, e1)[0]
        e2data = __get_math_val(datafile, e2)[0]
        match error_type:
            case "relative":

                def error_calc(adata, bdata):  # pylint: disable=function-redefined
                    """Relative error summation."""
                    return np.sqrt((e1data / adata) ** 2 + (e2data / bdata) ** 2)

            case "absolute":

                def error_calc(adata, bdata):  # pylint: disable=function-redefined, unused-argument
                    """Sum absolute errors."""
                    return np.sqrt(e1data**2 + e2data**2)

            case "diffsum":

                def error_calc(adata, bdata):  # pylint: disable=function-redefined
                    """Calculate error for difference over sum."""
                    return np.sqrt(
                        (1.0 / (adata + bdata) - (adata - bdata) / (adata + bdata) ** 2) ** 2 * e1data**2
                        + (-1.0 / (adata + bdata) - (adata - bdata) / (adata + bdata) ** 2) ** 2 * e2data**2
                    )

            case _:
                raise ValueError(f"Unknown error calculation mode {error_type}")
    else:
        error_calc = None

    adata, aname = __get_math_val(datafile, col_a)
    bdata, bname = __get_math_val(datafile, col_b)
    return adata, bdata, error_calc, aname, bname


def __get_math_val(datafile: Data, col: Index) -> Tuple[Data, str]:
    """Interpret col as either col_a column index or value or an array of values.

    Args:
        datafile (Data):
            If not being used as a bound menthod, specifies the instance of Data to work with.
        col (various):
            If col can be interpreted as col_a column index then return the first matching column.
            If col is col_a 1D array of the same length as the data then just return the data. If col is col_a
            float then just return it as col_a float.

    Returns:
        (tuple of (:py:class:`Stoner.cpre.DataArray`,str)):
            The matching data.
    """
    match col:
        case int() | str() | _pattern_type():
            col = datafile.find_col(col)
            if isinstance(col, list):
                col = col[0]
            data = datafile.column(col)
            name = datafile.column_headers[col]
        case np.ndarray() if col.ndim == 1 and len(col) == len(datafile):
            data = col
            name = "data"
        case float():
            data = col * np.ones(len(datafile))
            name = str(col)
        case _:
            raise RuntimeError(f"Bad column index: {col}")
    return data, name


def add(
    datafile: Data,
    col_a: Index,
    col_b: Index,
    replace: bool = False,
    header: Optional[str] = None,
    index: Optional[Index] = None,
) -> Data:
    """Add one column, number or array (col_b) to another column (col_a).

    Args:
        datafile (Data):
            If not being used as a bound menthod, specifies the instance of Data to work with.
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
    adata, bdata, err_calc, aname, bname = _do_error_calc(datafile, col_a, col_b, error_type="absolute")
    err_header = None
    if isinstance(header, tuple) and len(header) == 2:
        header, err_header = header
    if header is None:
        header = f"{aname}+{bname}"
    if err_calc is not None and err_header is None:
        err_header = "Error in " + header
    index = datafile.shape[1] if index is None else datafile.find_col(index)
    if err_calc is not None:
        err_data = err_calc(adata, bdata)
    datafile.add_column((adata + bdata), header=header, index=index, replace=replace)
    if err_calc is not None:
        datafile.add_column(err_data, header=err_header, index=index + 1, replace=False)
    return datafile


def diffsum(
    datafile: Data,
    col_a: Index,
    col_b: Index,
    replace: bool = False,
    header: Optional[str] = None,
    index: Optional[Index] = None,
) -> Data:
    r"""Calculate :math:`\frac{a-b}{a+b}` for the two columns *a* and *b*.

    Args:
        datafile (Data):
            If not being used as a bound menthod, specifies the instance of Data to work with.
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
    adata, bdata, err_calc, aname, bname = _do_error_calc(datafile, col_a, col_b, error_type="diffsum")
    err_header = None
    if isinstance(header, tuple) and len(header) == 2:
        header, err_header = header
    if header is None:
        header = f"({aname}-{bname})/({aname}+{bname})"
    if err_calc is not None and err_header is None:
        err_header = "Error in " + header
    if err_calc is not None:
        err_data = err_calc(adata, bdata)
    index = datafile.shape[1] if index is None else datafile.find_col(index)
    datafile.add_column((adata - bdata) / (adata + bdata), header=header, index=index, replace=replace)
    if err_calc is not None:
        datafile.add_column(err_data, header=err_header, index=index + 1, replace=False)
    return datafile


def divide(
    datafile: Data,
    col_a: Index,
    col_b: Index,
    replace: bool = False,
    header: Optional[str] = None,
    index: Optional[Index] = None,
) -> Data:
    """Divide one column (col_a) by  another column, number or array (col_b).

    Args:
        datafile (Data):
            If not being used as a bound menthod, specifies the instance of Data to work with.
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
    adata, bdata, err_calc, aname, bname = _do_error_calc(datafile, col_a, col_b, error_type="relative")
    err_header = None
    if isinstance(header, tuple) and len(header) == 2:
        header, err_header = header
    if header is None:
        header = f"{aname}/{bname}"
    if err_calc is not None and err_header is None:
        err_header = "Error in " + header
    index = datafile.shape[1] if index is None else datafile.find_col(index)
    datafile.add_column((adata / bdata), header=header, index=index, replace=replace)
    if err_calc is not None:
        err_data = err_calc(adata, bdata) * np.abs(adata / bdata)
        datafile.add_column(err_data, header=err_header, index=index + 1, replace=False)
    return datafile


def max(  # pylint: disable=redefined-builtin
    datafile: Data, column: Optional[Index] = None, bounds: Optional[Callable] = None
) -> Tuple[float, int]:
    """Find maximum value and index in col_a column of data.

    Args:
        datafile (Data):
            If not being used as a bound menthod, specifies the instance of Data to work with.
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
        col = datafile.setas._get_cols("ycol")
    else:
        col = datafile.find_col(column)
    if bounds is not None:
        datafile._push_mask()
        datafile._set_mask(bounds, True, col)
    result = datafile.data[:, col].max(), datafile.data[:, col].argmax()
    if bounds is not None:
        datafile._pop_mask()
    return result


def mean(
    datafile: Data,
    column: Optional[Index] = None,
    sigma: Optional[Union[NDArray, Index]] = None,
    bounds: Optional[Callable] = None,
) -> float:
    """Find mean value of col_a data column.

    Args:
        datafile (Data):
            If not being used as a bound menthod, specifies the instance of Data to work with.
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
        Fix the row index when the bounds function is used - see note of :py:meth:`Stoner.Data.max`
    """
    _ = datafile._col_args(scalar=True, ycol=column, yerr=sigma)

    if bounds is not None:
        datafile._push_mask()
        datafile._set_mask(bounds, True, _.ycol)

    if isiterable(sigma) and len(sigma) == len(datafile) and all_type(sigma, float):
        sigma = np.array(sigma)
        _["has_yerr"] = True
    elif _.has_yerr:
        sigma = datafile.data[:, _.yerr]

    if not _.has_yerr:
        result = datafile.data[:, _.ycol].mean()
    else:
        ydata = datafile.data[:, _.ycol]
        w = 1 / (sigma**2 + 1e-8)
        norm = w.sum(axis=0)
        error = np.sqrt((sigma**2).sum(axis=0)) / len(sigma)
        result = (ydata * w).mean(axis=0) / norm, error
    if bounds is not None:
        datafile._pop_mask()
    return result


def min(  # pylint: disable=redefined-builtin
    datafile: Data, column: Optional[Index] = None, bounds: Optional[Callable] = None
) -> Tuple[float, int]:
    """Find minimum value and index in col_a column of data.

    Args:
        datafile (Data):
            If not being used as a bound menthod, specifies the instance of Data to work with.
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
        col = datafile.setas._get_cols("ycol")
    else:
        col = datafile.find_col(column)
    if bounds is not None:
        datafile._push_mask()
        datafile._set_mask(bounds, True, col)
    result = datafile.data[:, col].min(), datafile.data[:, col].argmin()
    if bounds is not None:
        datafile._pop_mask()
    return result


def multiply(
    datafile: Data,
    col_a: Index,
    col_b: Index,
    replace: bool = False,
    header: Optional[str] = None,
    index: Optional[Index] = None,
) -> Data:
    """Multiply one column (col_a) by  another column, number or array (col_b).

    Args:
        datafile (Data):
            If not being used as a bound menthod, specifies the instance of Data to work with.
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
    adata, bdata, err_calc, aname, bname = _do_error_calc(datafile, col_a, col_b, error_type="relative")
    err_header = None
    if isinstance(header, tuple) and len(header) == 2:
        header, err_header = header
    if header is None:
        header = f"{aname}*{bname}"
    if err_calc is not None and err_header is None:
        err_header = "Error in " + header
    index = datafile.shape[1] if index is None else datafile.find_col(index)
    if err_calc is not None:
        err_data = err_calc(adata, bdata) * np.abs(adata * bdata)
    datafile.add_column((adata * bdata), header=header, index=index, replace=replace)
    if err_calc is not None:
        datafile.add_column(err_data, header=err_header, index=index + 1, replace=False)
    return datafile


def span(datafile: Data, column: Optional[Index] = None, bounds: Optional[Callable] = None) -> Tuple[float, float]:
    """Return a tuple of the maximum and minimum values within the given column and bounds.

    Args:
        datafile (Data):
            If not being used as a bound menthod, specifies the instance of Data to work with.
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
    return (datafile.min(column, bounds)[0], datafile.max(column, bounds)[0])


def std(
    datafile: Data,
    column: Optional[Index] = None,
    sigma: Optional[Union[NDArray, Index]] = None,
    bounds: Optional[Callable] = None,
):
    """Find standard deviation value of col_a data column.

    Args:
        datafile (Data):
            If not being used as a bound menthod, specifies the instance of Data to work with.
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
        Fix the row index when the bounds function is used - see note of :py:meth:`Stoner.Data.max`
    """
    _ = datafile._col_args(scalar=True, ycol=column, yerr=sigma)

    if bounds is not None:
        datafile._push_mask()
        datafile._set_mask(bounds, True, _.ycol)

    if isiterable(sigma) and len(sigma) == len(datafile) and all_type(sigma, float):
        sigma = np.array(sigma)
    elif _.yerr:
        sigma = datafile.data[:, _.yerr]
    else:
        sigma = np.ones(len(datafile))

    ydata = datafile.data[:, _.ycol]

    sigma = np.abs(sigma) / np.nanmax(np.abs(sigma))
    sigma = np.where(sigma < 1e-8, 1e-8, sigma)
    weights = 1 / sigma**2
    weights[np.isnan(weights)] = 0.0

    result = np.sqrt(np.cov(ydata, aweights=weights))

    if bounds is not None:
        datafile._pop_mask()
    return result


def subtract(
    datafile: Data,
    col_a: Index,
    col_b: Index,
    replace: bool = False,
    header: Optional[str] = None,
    index: Optional[Index] = None,
) -> Data:
    """Subtract one column, number or array (col_b) from another column (col_a).

    Args:
        datafile (Data):
            If not being used as a bound menthod, specifies the instance of Data to work with.
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
    adata, bdata, err_calc, aname, bname = _do_error_calc(datafile, col_a, col_b, error_type="absolute")
    err_header = None
    if isinstance(header, tuple) and len(header) == 2:
        header, err_header = header
    if header is None:
        header = f"{aname}-{bname}"
    if err_calc is not None and err_header is None:
        err_header = "Error in " + header
    index = datafile.shape[1] if index is None else datafile.find_col(index)
    if err_calc is not None:
        err_data = err_calc(adata, bdata)
    datafile.add_column((adata - bdata), header=header, index=index, replace=replace)
    if err_calc is not None:
        col_a = datafile.find_col(col_a)
        datafile.add_column(err_data, header=err_header, index=index + 1, replace=False)
    return datafile
