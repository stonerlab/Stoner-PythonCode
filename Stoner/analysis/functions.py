# -*- coding: utf-8 -*-
"""Generic analysis functions for DataFiles."""

from inspect import getfullargspec
from warnings import warn

import numpy as np
from numpy import ma
from scipy.integrate import cumulative_simpson, cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from typing import Callable, Union, Tuple, Optional

from ..core.exceptions import assertion
from ..tools import isiterable, isTuple
from ..tools.typing import Data, Index, Kwargs, NumericArray
from .utils import threshold as _threshold


def apply(
    datafile: Data, func: Callable, col: Index = None, replace: bool = True, header: str = None, **kwargs: Kwargs
) -> Data:
    """Apply the given function to each row in the data set and adds to the data set.

    Args:
        datafile (Data):
            If not being used as a bound menthod, specifies the instance of Data to work with.
        func (callable):
            The function to apply to each row of the data.
        col (index):
            The column in which to place the result of the function

    Keyword Arguments:
        replace (bool):
            Either replace the existing column/complete data or create a new column or data file.
        header (string or None):
            The new column header(s) (defaults to the name of the function func).
        **kwargs:
            Other keyword arguments.


    Note:
        If any extra keyword arguments are supplied then these are passed to the function directly. If
        you need to pass any arguments that overlap with the keyword arguments to :py:math:`AnalysisMixin.apply`
        then these can be supplied in a dictionary argument *_extra*.

        The callable *func* should have a signature::

            def func(row,**kargs):

        and should return either a single float, in which case it will be used to repalce the specified column,
        or an array, in which case it is used to completely replace the row of data.

        If the function returns a complete row of data, then the *replace* parameter will cause the return
        value to be a new datafile:Data, leaving the original unchanged. The *headers* parameter can give the complete
        column headers for the new data file.

    Returns:
        (:py:class:`Stoner.Data`):
            The newly modified Data object.
    """
    if col is None:
        col = datafile.setas.get("y", [0])[0]
    col = datafile.find_col(col)
    kwargs.update(kwargs.pop("_extra", {}))
    # Check the dimension of the output
    ret = func(next(datafile.rows()), **kwargs)
    try:
        next(datafile.rows(reset=True))
    except (RuntimeError, StopIteration):
        pass
    if isiterable(ret):
        nc = np.zeros((len(datafile), len(ret)))
    else:
        nc = np.zeros(len(datafile))
    # Evaluate the data row by row
    for ix, r in enumerate(datafile.rows()):
        ret = func(r, **kwargs)
        if isiterable(ret) and not isinstance(ret, np.ndarray):
            ret = np.ma.MaskedArray(ret)
        nc[ix] = ret
    # Work out how to handle the result
    if nc.ndim == 1:
        if header is None:
            header = func.__name__
        datafile.add_column(nc, header=header, index=col, replace=replace, setas=datafile.setas[col])
        ret = datafile
    else:
        if not replace:
            ret = datafile.clone
        else:
            ret = datafile
        ret.data = nc
        if header is not None:
            ret.column_headers = header
    return ret


def clip(datafile: Data, clipper: Union[Tuple[float, float], NumericArray], column: Index = None) -> Data:
    """Clips the data based on the column and the clipper value.

    Args:
        datafile (Data):
            If not being used as a bound menthod, specifies the instance of Data to work with.
        column (index):
            Column to look for the maximum in
        clipper (tuple or array):
            Either a tuple of (min,max) or a numpy.ndarray -
            in which case the max and min values in that array will be
            used as the clip limits
    Returns:
        (:py:class:`Stoner.Data`):
            The newly modified Data object.

    Note:
        If column is not defined (or is None) the :py:attr:`DataFile.setas` column
        assignments are used.
    """
    if column is None:
        col = datafile.setas._get_cols("ycol")
    else:
        col = datafile.find_col(column)
    clipper = (min(clipper), max(clipper))
    return datafile.del_rows(col, lambda x, y: x < clipper[0] or x > clipper[1])


def decompose(
    datafile: Data,
    xcol: Optional[Index] = None,
    ycol: Optional[Index] = None,
    sym: Optional[Index] = None,
    asym: Optional[Index] = None,
    replace: bool = True,
    hysteretic: bool = False,
    **kwargs: Kwargs,
) -> Data:
    """Given (x,y) data, decomposes the y part into symmetric and antisymmetric contributions in x.

    Args:
        datafile (Data):
            If not being used as a bound menthod, specifies the instance of Data to work with.

    Keyword Arguments:
        xcol (index):
            Index of column with x data - defaults to first x column in datafile.setas
        ycol (index or list of indices):
            indices of y column(s) data
        sym (index):
            Index of column to place symmetric data in default, append to end of data
        asym (index):
            Index of column for asymmetric part of ata. Defaults to appending to end of data
        replace (bool):
            Overwrite data with output (true)
        hysteretic (book):
            Look separately for outgoing and incoming data first.
        **kwargs:
            Other keyword arguments.

    Returns:
        datafile: The newly modified :py:class:`AnalysisMixin`.

    Example:
        .. plot:: samples/decompose.py
            :include-source:
            :outname: decompose
    """
    if xcol is None and ycol is None:
        startx = kwargs.pop("_startx", 0)
        cols = datafile.setas._get_cols(startx=startx)
        xcol = cols["xcol"]
        ycol = cols["ycol"]
    xcol = datafile.find_col(xcol)
    ycol = datafile.find_col(ycol)
    if not isinstance(ycol, list):
        ycol = [ycol]

    if hysteretic:
        from .Util import split_up_down

        fldr = split_up_down(datafile, datafile.xcol)
        for grp in ["rising", "falling"]:
            for f in fldr[grp][1:]:
                fldr[grp][0] += f
        rising = fldr["rising"][0].sort(xcol)
        falling = fldr["falling"][0].sort(xcol)
    else:
        rising = datafile.clone.sort(xcol)
        falling = rising.clone

    rising_data = rising.deduplicate(xcol, clone=False)
    falling_data = falling.deduplicate(xcol, clone=False)

    falling_func = interp1d(
        falling_data[:, xcol],
        falling_data[:, ycol],
        kind="linear",
        bounds_error=False,
        axis=0,
    )

    rising_func = interp1d(rising_data[:, xcol], rising_data[:, ycol], kind="linear", bounds_error=False, axis=0)
    rising_vals = rising_func((datafile // xcol).view(np.ndarray))
    falling_vals = falling_func(-(datafile // xcol).view(np.ndarray))

    symd = (rising_vals + falling_vals) / 2
    asymd = (rising_vals - falling_vals) / 2

    if sym is None:
        datafile &= symd
        datafile.column_headers[-1] = "Symmetric Data"
    else:
        datafile.add_column(symd, header="Symmetric Data", index=sym, replace=replace)
    if asym is None:
        datafile &= asymd
        datafile.column_headers[-1] = "Asymmetric Data"
    else:
        datafile.add_column(asymd, header="Symmetric Data", index=asym, replace=replace)
    return datafile


def integrate(
    datafile: Data,
    xcol: Optional[Index] = None,
    ycol: Optional[Index] = None,
    result: Optional[bool] = None,
    header: Optional[str] = None,
    result_name: Optional[str] = None,
    output: str = "data",
    bounds: Callable = lambda x, y: True,
    method: str = "simpson",
    **kargs: Kwargs,
) -> Data:
    """Integrate a column of data, optionally returning the cumulative integral.

    Args:
        datafile (Data):
            If not being used as a bound menthod, specifies the instance of Data to work with.
        xcol (index):
            The X data column index (or header)
        ycol (index):
            The Y data column index (or header)

    Keyword Arguments:
        result (index or None):
            Either a column index (or header) to overwrite with the cumulative data,
            or True to add a new column or None to not store the cumulative result.
        result_name (str):
            The metadata name for the final result
        header (str):
            The name of the header for the results column.
        output (Str):
            What to return - 'data' (default) - this object, 'result': final result
        bounds (callable):
            A function that evaluates for each row to determine if the data should be integrated over.
        method (str):
            Either "simpson" (default) or "trapezoid" to select the integration method. See Note.
        **kargs:
            Other keyword arguments are fed direct to the scipy.integrate.cumtrapz method

    Returns:
        (:py:class:`Stoner.Data`):
            The newly modified Data object.

    Note:
        This is a pass through to the :py:func:`scipy.integrate.cumulative_simpson` or
        :py:func:`scipy.integrate.cumulative_trapezoid` function depending on the value of *method* and whether the
        x-data is monotonically increasing. If it isnot and *simpson* has been requested, a warning is issued and it
        falls back to trapezoidal integration.
    """
    _ = datafile._col_args(xcol=xcol, ycol=ycol)

    working = datafile.search(_.xcol, bounds)
    working = ma.mask_rowcols(working, axis=0)
    xdat = working[:, datafile.find_col(_.xcol)]
    ydat = working[:, datafile.find_col(_.ycol)]
    ydat = np.atleast_2d(ydat).T

    if np.any(np.diff(xdat, axis=-1) <= 0) and method == "simpson":  # Must use trapezoid
        warn("X-data is not monotonically increasing, falling back to trapezoid integration")
        method = "trapezoid"

    func = {"simpson": cumulative_simpson, "trapezoid": cumulative_trapezoid}.get(method, cumulative_trapezoid)

    final = []
    for i in range(ydat.shape[1]):
        yd = ydat[:, i]
        resultdata = func(yd, x=xdat, **kargs)
        resultdata = np.append(np.array([0]), resultdata)
        if result is not None:
            header = header if header is not None else f"Integral of {datafile.column_headers[_.ycol]}"
            if isinstance(result, bool) and result:
                datafile.add_column(resultdata, header=header, replace=False)
            else:
                result_name = datafile.column_headers[datafile.find_col(result)]
                datafile.add_column(resultdata, header=header, index=result, replace=(i == 0))
        final.append(resultdata[-1])
    if len(final) == 1:
        final = final[0]
    else:
        final = np.array(final)
    result_name = result_name if result_name is not None else header if header is not None else "Area"
    datafile[result_name] = final
    if output.lower() == "result":
        return final
    return datafile


def normalise(
    datafile: Data,
    target: Optional[Index] = None,
    base: Optional[Index] = None,
    replace: bool = True,
    header: Optional[str] = None,
    scale: Optional[Tuple[float, float]] = None,
    limits: Tuple[float, float] = (0.0, 1.0),
) -> Data:
    """Normalise data columns by dividing through by a base column value.

    Args:
        datafile (Data):
            If not being used as a bound menthod, specifies the instance of Data to work with.

    Keyword Arguments:
        target (index):
            One or more target columns to normalise can be a string, integer or list of strings or integers.
            If None then the default 'y' column is used.
        base (index):
            The column to normalise to, can be an integer or string. **Deprecated** can also be a tuple (low,
            high) being the output range
        replace (bool):
            Set True(default) to overwrite  the target data columns
        header (string or None):
            The new column header - default is target name(norm)
        scale (None or tuple of float,float):
            Output range after normalising - low,high or None to map to -1,1
        limits (float,float):
            (low,high) - Take the input range from the *high* and *low* fraction of the input when sorted.

    Returns:
        (:py:class:`Stoner.Data`):
            The newly modified Data object.

    Notes:
        The *limits* parameter is used to set the input scale being normalised from - if the data has a few
        outliers then this setting can be used to clip the input range before normalising. The parameters in
        the limit are the values at the *low* and *high* fractions of the cumulative distribution function of
        the data.
    """
    _ = datafile._col_args(scalar=True, ycol=target)

    target = _.ycol
    if not isinstance(target, list):
        target = [datafile.find_col(target)]
    for t in target:
        if header is None:
            header = datafile.column_headers[datafile.find_col(t)] + "(norm)"
        else:
            header = str(header)
        if not isTuple(base, float, float) and base is not None:
            datafile.divide(t, base, header=header, replace=replace)
        else:
            if isTuple(base, float, float):
                scale = base
            elif scale is None:
                scale = (-1.0, 1.0)
            if not isTuple(scale, float, float):
                raise ValueError("limit parameter is either None, or limit or base is a tuple of two floats.")
            data = datafile.column(t).ravel()
            data = np.sort(data[~np.isnan(data)])
            if limits != (0.0, 1.0):
                low, high = limits
                low = data[int(low * data.size)]
                high = data[int(high * data.size)]
            else:
                high = data.max()
                low = data.min()
            data = np.copy(datafile.data[:, t])
            data = np.where(data > high, high, np.where(data < low, low, data))
            scl, sch = scale
            data = (data - low) / (high - low) * (sch - scl) + scl
            setas = datafile.setas.clone
            datafile.add_column(data, index=t, replace=replace, header=header)
            datafile.setas = setas
    return datafile


def stitch(
    datafile: Data,
    other: Data,
    xcol: Optional[Index] = None,
    ycol: Optional[Index] = None,
    overlap: Optional[Tuple[float, float]] = None,
    min_overlap: float = 0.0,
    mode: str = "All",
    func: Optional[Callable] = None,
    p0: Optional[NumericArray] = None,
):
    r"""Apply a scaling to this data set to make it stich to another dataset.

    Args:
        datafile (Data):
            If not being used as a bound menthod, specifies the instance of Data to work with.
        other (DataFile):
            Another data set that is used as the base to stitch this one on to
        xcol (index or None):
            The x data column. If left as None then the current setas attribute is used.
        ycol (index or None):
            The y data column. If left as None then the current setas attribute is used.

    Keyword Arguments:
        overlap (tuple of (lower,higher) or None):
            The band of x values that are used in both data sets to match,
            if left as None, thenthe common overlap of the x data is used.
        min_overlap (float):
            If you know that overlap must be bigger than a certain amount, the bounds between the two
            data sets needs to be adjusted. In this case min_overlap shifts the boundary of the overlap
            on this DataFile.
        mode (str):
            Unless *func* is specified, controls which parameters are actually variable, defaults to all of them.
        func (callable):
            a stitching function that transforms :math:`(x,y)\rightarrow(x',y')`. Default is to use
            functions defined by *mode*
        p0 (iterable):
            if func is not None then p0 should be the starting values for the stitching function parameters

    Returns:
        (:py:class:`Stoner.Data`):
            A copy of the current :py:class:`AnalysisMixin` with the x and y data columns adjusted to stitch

    To stitch the data together, the x and y data in the current data file is transforms so that
    :math:`x'=x+A` and :math:`y'=By+C` where :math:`A,B,C` are constants and :math:`(x',y')` are close matches
    to the :math:`(x,y)` data in *other*. The algorithm assumes that the overlap region contains equal
    numbers of :math:`(x,y)` points *mode* controls whether A,B, and C are fixed or adjustable

        - "All" - all three parameters adjustable
        - "Scale y, shift x" - C is fixed at 0.0
        - "Scale and shift y" A is fixed at 0.0
        - "Scale y" - only B is adjustable
        - "Shift y" - Only c is adjsutable
        - "Shift x" - Only A is adjustable
        - "Shift both" - B is fixed at 1.0

    See Also:
        User Guide section :ref:`stitch_guide`

    Example:
        .. plot:: samples/stitch-int-overlap.py
            :include-source:
            :outname:  stitch_int_overlap
    """
    _ = datafile._col_args(xcol=xcol, ycol=ycol, scalar=True)
    points = datafile.column([_.xcol, _.ycol])
    points = points[points[:, 0].argsort(), :]
    points[:, 0] += min_overlap
    otherpoints = other.column([_.xcol, _.ycol])
    otherpoints = otherpoints[otherpoints[:, 0].argsort(), :]
    datafile_second = np.max(points[:, 0]) > np.max(otherpoints[:, 0])
    match overlap:
        case int() if overlap > 0:
            if datafile_second:
                lower = points[0, 0]
                upper = points[overlap, 0]
            else:
                lower = points[-overlap - 1, 0]
                upper = points[-1, 0]
        case (float(), float()):
            lower = min(overlap)
            upper = max(overlap)
        case _:
            lower = max(np.min(points[:, 0]), np.min(otherpoints[:, 0]))
            upper = min(np.max(points[:, 0]), np.max(otherpoints[:, 0]))

    inrange = np.logical_and(points[:, 0] >= lower, points[:, 0] <= upper)
    points = points[inrange]
    num_pts = points.shape[0]
    if datafile_second:
        otherpoints = otherpoints[-num_pts - 1 : -1]
    else:
        otherpoints = otherpoints[0:num_pts]
    x = points[:, 0]
    y = points[:, 1]
    xp = otherpoints[:, 0]
    yp = otherpoints[:, 1]
    if func is None:
        opts = {
            "all": (lambda x, y, A, B, C: (x + A, y * B + C)),
            "scale y and shift x": (lambda x, y, A, B: (x + A, B * y)),
            "scale and shift y": (lambda x, y, B, C: (x, y * B + C)),
            "scale y": (lambda x, y, B: (x, y * B)),
            "shift y": (lambda x, y, C: (x, y + C)),
            "shift both": (lambda x, y, A, C: (x + A, y + C)),
        }
        defaults = {
            "all": [1, 2, 3],
            "scale y,shift x": [1, 2],
            "scale and shift y": [2, 3],
            "scale y": [2],
            "shift y": [3],
            "shift both": [1, 3],
        }
        A0 = np.mean(xp) - np.mean(x)
        C0 = np.mean(yp) - np.mean(y)
        B0 = (np.max(yp) - np.min(yp)) / (np.max(y) - np.min(y))
        p = np.array([0, A0, B0, C0])
        assertion(isinstance(mode, str), "mode keyword should be a string if func is not defined")
        mode = mode.lower()
        assertion(mode in opts, f"mode keyword should be one of {opts.keys}")
        func = opts[mode]
        p0 = p[defaults[mode]]
    else:
        assertion(callable(func), "Keyword func should be callable if given")
        args = getfullargspec(func)[0]  # pylint: disable=W1505
        assertion(isiterable(p0), "Keyword parameter p0 shoiuld be iterable if keyword func is given")
        assertion(len(p0) == len(args) - 2, "Keyword p0 should be the same length as the optional arguments to func")
    # This is a bit of a hack, we turn (x,y) points into a 1D array of x and then y data
    set1 = np.append(x, y)
    set2 = np.append(xp, yp)
    assertion(len(set1) == len(set2), "The number of points in the overlap are different in the two data sets")

    def _transform(set1, *p):
        """Construct the wrapper function to fit for transform."""
        m = int(len(set1) / 2)
        x = set1[:m]
        y = set1[m:]
        tmp = func(x, y, *p)
        out = np.append(tmp[0], tmp[1])
        return out

    popt, pcov = curve_fit(_transform, set1, set2, p0=p0)  # Curve fit for optimal A,B,C
    perr = np.sqrt(np.diagonal(pcov))
    datafile.data[:, _.xcol], datafile.data[:, _.ycol] = func(
        datafile.data[:, _.xcol], datafile.data[:, _.ycol], *popt
    )
    datafile["Stitching Coefficients"] = list(popt)
    datafile["Stitching Coefficient Errors"] = list(perr)
    datafile["Stitching overlap"] = (lower, upper)
    datafile["Stitching Window"] = num_pts

    return datafile


def threshold(
    datafile: Data,
    threshold: float,
    col: Optional[Index] = None,
    rising: bool = True,
    falling: bool = False,
    xcol: Optional[Index] = None,
    transpose: bool = False,
    all_vals: bool = False,
) -> Data:
    """Find partial indices where the data in column passes the threshold, rising or falling.

    Args:
        datafile (Data):
            If not being used as a bound menthod, specifies the instance of Data to work with.
        threshold (float):
            Value to look for in column col

    Keyword Arguments:
        col (index):
            Column index to look for data in
        rising (bool):
            look for case where the data is increasing in value (default True)
        falling (bool):
            look for case where data is fallinh in value (default False)
        xcol (index, bool or None):
            rather than returning a fractional row index, return the
            interpolated value in column xcol. If xcol is False, then return a complete row
            all_vals (bool): return all crossing points of the threshold or just the first. (default False)
        transpose (bbool):
            Swap the x and y columns around - this is most useful when the column assignments
            have been done via the setas attribute
        all_vals (bool):
            Return all values that match the criteria, or just the first in the file.

    Returns:
        (float):
            Either a sing;le fractional row index, or an in terpolated x value

    Note:
        If you don't specify a col value or set it to None, then the assigned columns via the
        :py:attr:`DataFile.setas` attribute will be used.

    Warning:
        There has been an API change. Versions prior to 0.1.9 placed the column before the threshold in the
        positional argument list. In order to support the use of assigned columns, this has been swapped to the
        present order.
    """
    DataArray = type(datafile.data)
    _ = datafile._col_args(xcol=xcol, ycol=col)

    col = _.ycol
    if xcol is None and _.has_xcol:
        xcol = _.xcol

    current = datafile.column(col)

    # Recursively call if we've got an iterable threshold
    if isiterable(threshold):
        if isinstance(xcol, bool) and not xcol:
            ret = np.zeros((len(threshold), datafile.shape[1]))
        else:
            ret = np.zeros_like(threshold).view(type=DataArray)
        for ix, th in enumerate(threshold):
            ret[ix] = datafile.threshold(th, col=col, xcol=xcol, rising=rising, falling=falling, all_vals=all_vals)
        # Now we have to clean up the  retujrn list into a DataArray
        if isinstance(xcol, bool) and not xcol:  # if xcol was False we got a complete row back
            ch = datafile.column_headers
            ret.setas = datafile.setas.clone
            ret.column_headers = ch
            ret.i = ret[0].i
        else:  # Either xcol was None so we got indices or we got a specified column back
            if xcol is not None:  # Specific column
                ret = np.atleast_2d(ret)
                ret.column_headers = [datafile.column_headers[datafile.find_col(xcol)]]
                ret.i = [r.i for r in ret]
                ret.setas = "x"
                ret.isrow = False
            else:
                ret.column_headers = ["Index"]
                ret.isrow = False
        return ret
    ret = _threshold(threshold, current, rising=rising, falling=falling)
    if not all_vals:
        ret = [ret[0]] if np.any(ret) else []

    if isinstance(xcol, bool) and not xcol:
        retval = datafile.interpolate(ret, xcol=False)
        retval.setas = datafile.setas.clone
        retval.setas.shape = retval.shape
        retval.i = ret
        ret = retval
    elif xcol is not None:
        retval = datafile.interpolate(ret, xcol=False)[:, datafile.find_col(xcol)]
        # if retval.ndim>0:   #not sure what this bit does but it's throwing errors for a simple threshold
        # retval.setas=datafile.setas.clone
        # retval.setas.shape=retval.shape
        # retval.i=ret
        ret = retval
    else:
        ret = DataArray(ret)
    if not all_vals:
        if ret.size == 1:
            pass
        elif ret.size > 1:
            ret = ret[0]
        else:
            ret = []
    if isinstance(ret, DataArray):
        ret.isrow = True
    return ret
