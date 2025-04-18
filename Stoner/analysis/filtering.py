#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filtering and smoothing functions for analysis code."""

from copy import deepcopy as copy
from warnings import warn

import numpy as np
from numpy import ma
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import get_window, convolve, savgol_filter

from ..tools import isiterable, isnone, ordinal, isLikeList
from ..compat import int_types, string_types, get_func_params

from .utils import outlier as _outlier, _twoD_fit, GetAffineTransform


def filter(datafile, func=None, cols=None, reset=True):
    """Set the mask on rows of data by evaluating a function for each row.

    Args:
        func (callable):
            is a callable object that should take a single list as a p[parameter representing one row.
        cols (list):
            a list of column indices that are used to form the list of values passed to func.
        reset (bool):
            determines whether the mask is reset before doing the filter (otherwise rows already
            masked out will be ignored in the filter (so the filter is logically or'd)) The default value of
            None results in a complete row being passed into func.

    Returns:
        datafile: The current :py:class:`DataFile` object with the mask set
    """
    if cols is not None:
        cols = [datafile.find_col(c) for c in cols]
    if reset:
        datafile.data.mask = False
    for r in datafile.rows():
        if cols is None:
            datafile.mask[r.i, :] = not func(r)
        else:
            datafile.mask[r.i, :] = not func(r[cols])
    return datafile


def del_nan(datafile, col=None, clone=False):
    """Remove rows that have nan in them.

    eyword Arguments:
        col (index types or None):
            column(s) to look for nan's in. If None or not given, use setas columns.
        clone (boolean):
            if True clone the current object before running and then return the clone not datafile.

    Return:
        datafile (DataFile):
            Returns a copy of the current object (or clone if *clone*=True)
    """
    if clone:  # Set ret to be our clone
        ret = datafile.clone
    else:  # Not cloning so ret is datafile
        ret = datafile

    if col is None:  # If col is still None, use all columns that are set to any value in datafile.setas
        col = [ix for ix, col in enumerate(datafile.setas) if col != "."]
    if not isLikeList(col):  # If col isn't a list, make it one now
        col = [col]
    col = [ret.find_col(c) for c in col]  # Normalise col to be a list of integers
    dels = np.zeros(len(ret)).astype(bool)
    for ix in col:
        dels = np.logical_or(
            dels, np.isnan(ret.data[:, ix])
        )  # dels contains True if any row contains a NaN in columns col
    not_masked = np.logical_not(ma.mask_rows(ret.data).mask[:, 0])  # Get rows wqhich are not masked
    dels = np.logical_and(not_masked, dels)  # And make dels just be unmasked rows with NaNs

    ret.del_rows(np.logical_not(dels))  # Del the those rows
    return ret


def SG_Filter(
    datafile, col=None, xcol=None, points=15, poly=1, order=0, pad=True, result=None, replace=False, header=None
):
    """Implement a Savitsky-Golay filtering of data for smoothing and differentiating data.

    Args:
        col (column index):
            Column of Data to be filtered. if None, first y-column in setas is filtered.
        points (int):
            Number of data points to use in the filtering window. Should be an odd number > poly+1 (default 15)

    Keyword Arguments:
        xcol (coilumn index):
            If *order*>1 then can be used to specify an x-column to differentiate with respect to.
        poly (int):
            Order of polynomial to fit to the data. Must be equal or greater than order (default 1)
        order (int):
            Order of differentiation to carry out. Default=0 meaning smooth the data only.
        pad (bool or float):
            Pad the start and end of the array with the mean value (True, default) or specired value (float) or
            leave as is.
        result (None,True, or column_index):
            If not None, column index to insert new data, or True to append as last column
        header (string or None):
            Header for new column if result is not None. If header is Nne, a suitable column header is generated.

    Returns:
        (numpy array or datafile):
            If result is None, a numpy array representing the smoothed or differentiated data is returned.
            Otherwise, a copy of the modified Stoner.Data object is returned.

    Notes:
        If col is not specified or is None then the :py:attr:`DataFile.setas` column assignments are used
        to set an x and y column. If col is a tuple, then it is assumed to specify and x-column and y-column
        for differentiating data. This is now a pass through to :py:func:`scipy.signal.savgol_filter`

        Padding can help stop wildly wrong artefacts in the data at the start and enf of the data, particularly
        when the differential order is >1.

    See Also:
        User guide section :ref:`smoothing_guide`
    """
    points = int(points)
    if points % 2 == 0:  # Ensure window length is odd
        points += 1

    _ = datafile._col_args(scalar=False, ycol=col, xcol=xcol)

    if _.xcol is not None:
        if not isinstance(_.xcol, list):
            col = _.ycol + [_.xcol]
        else:
            col = _.ycol + _.xcol
        data = datafile.column(col).T
    else:
        col = _.ycol
        data = datafile.column(list(col)).T
        data = np.vstack((data, np.arange(data.shape[1])))

    ddata = savgol_filter(data, window_length=points, polyorder=poly, deriv=order, mode="interp")
    if isinstance(pad, bool) and pad:
        offset = int(np.ceil(points * (order + 1) ** 2 / 8))
        padv = np.mean(ddata[:, offset:-offset], axis=1)
        pad = np.ones((ddata.shape[0], offset))
        for ix, v in enumerate(padv):
            pad[ix] *= v
    elif isinstance(pad, float):
        offset = int(np.ceil(points / 2))
        pad = np.ones((ddata.shape[0], offset)) * pad

    if np.all(pad) and offset > 0:
        ddata[:, :offset] = pad
        ddata[:, -offset:] = pad
    if order >= 1:
        r = ddata[:-1] / ddata[-1]
    else:
        r = ddata

    if result is not None:
        if not isinstance(header, string_types):
            header = []
            for column in col[:-1]:
                header.append(f"{datafile.column_headers[column]} after {ordinal(order)} order Savitsky-Golay Filter")
        else:
            header = [header] * (len(col) - 1)
        if r.shape[0] > len(header):
            iterdata = r[: len(header)]
        else:
            iterdata = r
        for column, head in zip(iterdata, header):
            datafile.add_column(column.ravel(), header=head, index=result, replace=replace)
        return datafile
    return r


def bin(datafile, xcol=None, ycol=None, bins=0.03, mode="log", clone=True, **kargs):
    """Bin x-y data into new values of x with an error bar.

    Args:
        xcol (index):
            Index of column of data with X values
        ycol (index):
            Index of column of data with Y values
        bins (int, float or 1d array):
            Number of bins (if integer) or size of bins (if float), or bin edges (if array)
        mode (string):
            "log" or "lin" for logarithmic or linear binning

    Keyword Arguments:
        yerr (index):
            Column with y-error data if present.
        bin_start (float):
            Manually override the minimum bin value
        bin_stop (float):
            Manually override the maximum bin value
        clone (bool):
            Return a clone of the current Stoner.Data with binned data (True)
            or just the numbers (False).

    Returns:
        (:py:class:`Stoner.Data` or tuple of 4 array-like):
            Either a clone of the current data set with the new binned data or
            tuple of (bin centres, bin values, bin errors, number points/bin),
            depending on the *clone* parameter.

    Note:
        Algorithm inspired by MatLab code wbin,    Copyright (c) 2012:
        Michael Lindholm Nielsen


    See Also:
        User Guide section :ref:`binning_guide`
    """
    if None in (xcol, ycol):
        cols = datafile.setas._get_cols()
        if xcol is None:
            xcol = cols["xcol"]
        if ycol is None:
            ycol = cols["ycol"]
    yerr = kargs.pop("yerr", cols["yerr"] if cols["has_yerr"] else None)

    bin_left, bin_right, bin_centres = datafile.make_bins(xcol, bins, mode, **kargs)

    ycol = datafile.find_col(ycol)
    if yerr is not None:
        yerr = datafile.find_col(yerr)

    ybin = np.zeros((len(bin_left), len(ycol)))
    ebin = np.zeros((len(bin_left), len(ycol)))
    nbins = np.zeros((len(bin_left), len(ycol)))
    xcol = datafile.find_col(xcol)
    i = 0

    for limits in zip(bin_left, bin_right):
        data = datafile.search(xcol, limits)
        if len(data) > 1:
            ok = np.logical_not(np.isnan(data.y))
            data = data[ok]
        elif len(data) == 0 or (len(data) == 1 and np.isnan(data.y)):
            shape = list(data.shape)
            shape[0] = 0
            data = np.zeros(shape)
        if yerr is not None:
            w = 1.0 / data[:, yerr] ** 2
            W = np.sum(w, axis=0)
            if data.shape[0] > 3:
                e = max(np.std(data[:, ycol], axis=0) / np.sqrt(data.shape[0]), (1.0 / np.sqrt(W)) / data.shape[0])
            else:
                e = 1.0 / np.sqrt(np.where(W > 0, W, np.nan))
        else:
            w = np.ones((data.shape[0], len(ycol)))
            W = data.shape[0]
            if data[:, ycol].size > 1:
                e = np.std(data[:, ycol], axis=0) / np.sqrt(W)
            else:
                e = np.nan
        if data.shape[0] == 0 and datafile.debug:
            warn(f"Empty bin at {limits}")
        y = np.sum(data[:, ycol] * (w / W), axis=0)
        ybin[i, :] = y
        ebin[i, :] = e
        nbins[i, :] = data.shape[0]
        i += 1
    if clone:
        ret = datafile.clone
        ret.data = np.atleast_2d(bin_centres).T
        ret.column_headers = [datafile.column_headers[xcol]]
        ret.setas = ["x"]
        for i in range(ybin.shape[1]):
            head = str(datafile.column_headers[ycol[i]])

            ret.add_column(ybin[:, i], header=head)
            ret.add_column(ebin[:, i], header=f"d{head}")
            ret.add_column(nbins[:, i], header=f"#/bin {head}")
            s = list(ret.setas)
            s[-3:] = ["y", "e", "."]
            ret.setas = s
    else:
        ret = (bin_centres, ybin, ebin, nbins)
    return ret


def extrapolate(datafile, new_x, xcol=None, ycol=None, yerr=None, overlap=20, kind="linear", errors=None):
    """Extrapolate data based on local fit to x,y data.

    Args:
        new_x (float or array):
            New values of x data.

    Keyword Arguments:
        xcol (column index, None):
            column containing x-data or None to use setas attribute
        ycol (column index(es) or None):
            column(s) containing the y-data or None to use setas attribute.
        yerr (column index(es) or None):
            y error data column or None to use setas attribute
        overlap (float or int):
            range of x-data used for the local fit for extrapolating. If int then overlap number of
            points is used, if float then that range x-axis space is used.
        kind (str or callable):
            Determines local fitting function. If string should be "linear", "quadratic" or "cubic" if
            callable, then represents a function to be fitted to the data.
        errors (callable or None):
            If *kind* is a callable function, then errs must be defined and must also be a callable function.

    Returns:
        (array):
            Extrapolated values.

    Note:
        If the new_x values lie outside the span of the x-data, then the nearest *overlap* portion of the data
        is used to estimate the values. If the new_x values are within the span of the x-data then the portion
        of the data centred about the point and overlap points long will be used to interpolate a value.

        If *kind* is callable, it should take x values in the first parameter and free fitting parameters as
        the other parameters (i.e. as with :py:meth:`Stoner.Data.curve_fit`).
    """
    _ = datafile._col_args(xcol=xcol, ycol=ycol, yerr=yerr, scalar=False)
    kinds = {
        "linear": lambda x, m, c: m * x + c,
        "quadratic": lambda x, a, b, c: a * x**2 + b * x + c,
        "cubic": lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
    }
    errs = {
        "linear": lambda x, me, ce, popt: np.sqrt((me * x) ** 2 + ce**2),
        "quadratic": lambda x, ae, be, ce, popt: np.sqrt((2 * x**2 * ae) ** 2 + (x * be) ** 2 + ce**2),
        "cubic": lambda x, ae, be, ce, de, popt: np.sqrt(
            (3 * ae * x**3) ** 2 + (2 * x**2 * be) ** 2 + (x * ce) ** 2 + de**2
        ),
    }

    if callable(kind):
        kindf = kind
        if not callable(errors):
            raise TypeError("If kind is a callable, then errs must be defined and be callable as well")
        errsf = errors
    elif kind in kinds:
        kindf = kinds[kind]
        errsf = errs[kind]
    else:
        raise RuntimeError(f"Failed to recognise extrpolation function '{kind}'")
    scalar_x = not isiterable(new_x)
    if scalar_x:
        new_x = [new_x]
    if isinstance(new_x, ma.MaskedArray):
        new_x = new_x.compressed
    results = np.zeros((len(new_x), 2 * len(_.ycol)))
    work = datafile.clone
    for ix, x in enumerate(new_x):
        r = datafile.closest(x, xcol=_.xcol)
        match overlap:
            case int():
                if (r.i - overlap / 2) < 0:
                    ll = 0
                    hl = min(len(datafile), overlap)
                elif (r.i + overlap / 2) > len(datafile):
                    hl = len(datafile)
                    ll = max(hl - overlap, 0)
                else:
                    ll = r.i - overlap / 2
                    hl = r.i + overlap / 2
                bounds = {"_i__between": (ll, hl)}
                mid_x = (datafile[ll, _.xcol] + datafile[hl - 1, _.xcol]) / 2.0
            case float():
                if (r[_.xcol] - overlap / 2) < datafile.min(_.xcol)[0]:
                    ll = datafile.min(_.xcol)[0]
                    hl = ll + overlap
                elif (r[_.xcol] + overlap / 2) > datafile.max(_.xcol)[0]:
                    hl = datafile.max(_.xcol)[0]
                    ll = hl - overlap
                else:
                    ll = r[_.xcol] - overlap / 2
                    hl = r[_.xcol] + overlap / 2
                bounds = {f"{datafile.column_headers[_.xcol]}__between": (ll, hl)}
                mid_x = (ll + hl) / 2.0
            case _:
                raise TypeError(f"Overlap should be an integer or floating point number not a {type(overlap)}")
        pointdata = work.select(**bounds)
        pointdata.data[:, _.xcol] = pointdata.column(_.xcol) - mid_x
        ret = pointdata.curve_fit(kindf, _.xcol, _.ycol, sigma=_.yerr, absolute_sigma=True)
        if isinstance(ret, tuple):
            ret = [ret]
        for iy, rt in enumerate(ret):
            popt, pcov = rt
            perr = np.sqrt(np.diag(pcov))
            results[ix, 2 * iy] = kindf(x - mid_x, *popt)
            results[ix, 2 * iy + 1] = errsf(x - mid_x, *perr, popt)
    if scalar_x:
        results = results[0]
    return results


def interpolate(datafile, newX, kind="linear", xcol=None, replace=False):
    """Interpolate a dataset to get a new set of values for a given set of x data.

    Args:
        ewX (1D array or None):
            Row indices or X column values to interpolate with. If None, then the
            :py:meth:`Stoner.Data.interpolate` returns an interpolation function. Unlike the raw interpolation
            function from scipy, this interpolation function will work with MaskedArrays by compressing them
            first.

    Keyword Arguments:
        kind (string):
            Type of interpolation function to use - does a pass through from numpy. Default is linear.
        xcol (index or None):
            Column index or label that contains the data to use with newX to determine which rows to return.
            Defaults to None.
        replace (bool):
            If true, then the current Stoner.Data's data is replaced with the  newly interpolated data and the
            current Stoner.Data is returned.

    Returns:
        (2D numpy array):
            Section of the current object's data if replace is False(default) or the modofied Stoner.Data if
            replace is true.

    Note:
        Returns complete rows of data corresponding to the indices given in newX. if xcol is None, then newX is
        interpreted as (fractional) row indices. Otherwise, the column specified in xcol is thresholded with the
        values given in newX and the resultant row indices used to return the data.

        If the positional argument, newX is None, then the return value is an interpolation function. This
        interpolation function takes one argument - if *xcol* was None, this argument is interpreted as
        array indices, but if *xcol* was specified, then this argument is interpreted as an array of xvalues.
    """
    DataArray = type(datafile.data)  # pylint: disable=E0203
    lines = np.shape(datafile.data)[0]  # pylint: disable=E0203
    index = np.arange(lines)
    if xcol is None:
        xcol = datafile.setas._get_cols("xcol")
    elif isinstance(xcol, bool) and not xcol:
        xcol = None

    if isinstance(newX, ma.MaskedArray):
        newX = newX.compressed()

    if xcol is not None and newX is not None:  # We need to convert newX to row indices
        xfunc = interp1d(datafile.column(xcol), index, kind, 0)  # xfunc(x) returns partial index
        newX = xfunc(newX)
    inter = interp1d(index, datafile.data, kind, 0)  # pylint: disable=E0203

    if newX is None:  # Ok, we're going to return an interpolation function

        def wrapper(newX):
            """Wrap the interpolation function."""
            if isinstance(newX, ma.MaskedArray):
                newX = newX.compressed()
            else:
                newX = np.array(newX)
            if xcol is not None and newX is not None:  # We need to convert newX to row indices
                xfunc = interp1d(datafile.column(xcol), index, kind, 0)  # xfunc(x) returns partial index
                newX = xfunc(newX)
            return inter(newX)

        return wrapper

    if replace:
        datafile.data = inter(newX)
        ret = datafile
    else:
        ret = DataArray(inter(newX), isrow=True)
        ret.setas = datafile.setas.clone
    return ret


def make_bins(datafile, xcol, bins, mode="lin", **kargs):
    """Generate bin boundaries and centres along an axis.

    Args:
        xcol (index):
            Column of data with X values
        bins (1d_)array or int or float):
            Number of bins (int) or width of bins (if float)
        mode (string):
            "lin" for linear binning, "log" for logarithmic binning.

    Keyword Arguments:
        bin_start (float):
            Override minimum bin value
        bin_stop (float):
            Override the maximum bin value

    Returns:
        (tuple of 4 arrays):
            bin_start,bin_stop,bin_centres (1D arrays): The locations of the bin
            boundaries and centres for each bin.
    """
    xmin = kargs.pop("bin_start", (datafile // xcol).min())
    xmax = kargs.pop("bin_sop", (datafile // xcol).max())

    if isinstance(bins, int):  # Given a number of bins
        if mode.lower().startswith("lin"):
            bin_width = float(xmax - xmin) / bins
            bin_start = np.linspace(xmin, xmax - bin_width, bins)
            bin_stop = np.linspace(xmin + bin_width, xmax, bins)
            bin_centres = (bin_start + bin_stop) / 2.0
        elif mode.lower().startswith("log"):
            xminl = np.log(xmin)
            xmaxl = np.log(xmax)
            bin_width = float(xmaxl - xminl) / bins
            bin_start = np.linspace(xminl, xmaxl - bin_width, bins)
            bin_stop = np.linspace(xminl + bin_width, xmaxl, bins)
            bin_centres = (bin_start + bin_stop) / 2.0
            bin_start = np.exp(bin_start)
            bin_stop = np.exp(bin_stop)
            bin_centres = np.exp(bin_centres)
        else:
            raise ValueError(f"mode should be either lin(ear) or log(arthimitc) not {mode}")
    elif isinstance(bins, float):  # Given a bin with as a flot
        if mode.lower().startswith("lin"):
            bin_width = bins
            bins = int(np.ceil(abs(float(xmax - xmin) / bins)))
            bin_start = np.linspace(xmin, xmax - bin_width, bins)
            bin_stop = np.linspace(xmin + bin_width, xmax, bins)
            bin_centres = (bin_start + bin_stop) / 2.0
        elif mode.lower().startswith("log"):
            if not 0.0 < bins <= 1.0:
                raise ValueError("Bin width must be between 0 and 1 for log binning")
            if xmin <= 0:
                raise ValueError("The start of the binning must be a positive value in log mode.")
            xp = xmin
            splits = []
            centers = []
            while xp < xmax:
                splits.append(xp)
                centers.append(xp * (1 + bins / 2))
                xp = xp * (1 + bins)
            splits.append(xmax)
            bin_start = np.array(splits[:-1])
            bin_stop = np.array(splits[1:])
            bin_centres = np.array(centers)
        else:
            raise ValueError(f"mode should be either lin(ear) or log(arthimitc) not {mode}")
    elif isinstance(bins, np.ndarray) and bins.ndim == 1:  # Yser provided manuals bins
        bin_start = bins[:-1]
        bin_stop = bins[1:]
        if mode.lower().startswith("lin"):
            bin_centres = (bin_start + bin_stop) / 2.0
        elif mode.lower().startswith("log"):
            bin_start = np.where(bin_start <= 0, 1e-9, bin_start)
            bin_centres = np.exp(np.log(bin_start) + np.log(bin_stop) / 2.0)
        else:
            raise ValueError(f"mode should be either lin(ear) or log(arthimitc) not {mode}")
    else:
        raise TypeError(f"bins must be either an integer or a float, not a {type(bins)}")
    if len(bin_start) > len(datafile):
        raise ValueError("Attempting to bin into more bins than there is data.")
    return bin_start, bin_stop, bin_centres


def outlier_detection(
    datafile, column=None, window=7, shape="boxcar", certainty=3.0, action="mask", width=1, func=None, **kargs
):
    """Detect outliers in a column of data.

    Args:
        column(column index):
            specifying column for outlier detection. If not set,
            defaults to the current y set column.

    Keyword Arguments:
        window(int):
            data window for anomaly detection
        shape(str):
            The name of a :py:mod:`scipy.signal` windowing function to use when averaging the data.
            Defaults to 'boxcar' for a flat average.
        certainty(float):
            eg 3 detects data 3 standard deviations from average
        action(str or callable):
            what to do with outlying points, options are
            * 'mask' outlier points are masked (default)
            * 'mask row' outlier rows are masked
            * 'delete'  outlier rows are deleted
            * callable  the value of the action keyword is called with the outlier row
            * anything else defaults to do nothing.

        width(odd integer):
            Number of rows that an outliing spike could occupy. Defaults to 1.
        func (callable):
            A function that determines if the current row is an outlier.
        action_args (tuple):
            if *action* is callable, then action_args can be used to pass extra arguments to the action callable
        action_kargs (dict):
            If *action* is callable, then action_kargs can be used to pass extra keyword arguments to the action
            callable.

    Returns:
        (:py:class:`Stoner.Data`):
            The newly modified Data object.

    outlier_detection will add row numbers of detected outliers to the metadata
    of d, also will perform action depending on request eg 'mask', 'delete'
    (any other action defaults to doing nothing).

    The detection looks at a window of the data, takes the average and looks
    to see if the current data point falls certainty * std deviations away from
    data average.

    The outlier detection function has the signatrure::

        def outlier(row,column,window,certainty,**kargs)
            #code
            return True # or False

    All extra keyword arguments are passed to the outlier detector.

    IF *action* is a callable function then it should take the form of::

        def action(i,column, data, *action_args, **action_kargs):
            pass

    where *i* is the number of the outlier row, *column* the same value as above
    and *data* is the complete set of data.

    In all cases the indices of the outlier rows are added to the ;outlier' metadata.

    Example:
        .. plot:: samples/outlier.py
            :include-source:
            :outname: outlier
    """
    if func is None:
        func = _outlier

    if action not in ["delete", "mask", "mask row"] and not callable(action):
        raise ValueError(f"Do'n know what to do with action={action}")
    _ = datafile._col_args(scalar=True, ycol=column, **kargs)
    column = _.ycol
    params = get_func_params(func)
    for p in params:
        if p in _:
            kargs[p] = _[p]
    kargs.setdefault("ycol", column)
    if not callable(action) and ("action_args" in kargs or "acation_kargs" in kargs):
        raise SyntaxError("Can only have action_args and action_kargs keywords in action is callable")
    action_args = kargs.pop("action_args", ())
    action_kargs = kargs.pop("action_kargs", {})
    kargs["shape"] = shape
    for k in list(kargs.keys()):
        if k not in params:
            kargs.pop(k)
    index = np.zeros(len(datafile), dtype=bool)
    for i, t in enumerate(datafile.rolling_window(window, wrap=False, exclude_centre=width)):
        index[i] = func(datafile.data[i], t, metric=certainty, **kargs)
    datafile["outliers"] = np.arange(len(datafile))[index]  # add outlier indices to metadata
    if action == "mask" or action == "mask row":
        if action == "mask":
            datafile.mask[index, column] = True
        else:
            datafile.mask[index, :] = True
    elif action == "delete":
        datafile.data = datafile.data[~index]
    elif callable(action):  # this will call the action function with each row in turn from back to start
        for i in np.arange(len(datafile))[index][::-1]:
            action(i, column, datafile.data, *action_args, **action_kargs)
    return datafile


def scale(datafile, other, xcol=None, ycol=None, **kargs):
    """Scale the x and y data in this DataFile to match the x and y data in another DataFile.

    Args:
        other (DataFile):
            The other instance of a datafile to match to

    Keyword Arguments:
        xcol (column index):
            Column with x points in it, default to None to use setas attribute value
        ycol (column index):
            Column with ypoints in it, default to None to use setas attribute value
        xmode ('affine', 'linear','scale','offset'):
            How to manipulate the x-data to match up
        ymode ('linear','scale','offset'):
            How to manipulate the y-data to match up.
        bounds (callable):
            Used to identiyf the set of (x,y) points to be used for scaling. Defaults to the whole data set if
            not speicifed.
        otherbounds (callable):
            Used to detemrine the set of (x,y) points in the other data file. Defaults to bounds if not given.
        use_estimate (bool or 3x2 array):
            Specifies whether to estimate an initial transformation value or to use the provided one, or
            start with an identity transformation.
        replace (bool):
            Whether to map the x,y data to the new coordinates and return a copy of this Stoner.Data (true)
            or to just return the results of the scaling.
        headers (2-element list or tuple of strings):
            new column headers to use if replace is True.

    Returns:
        (various):
            Either a copy of the :py:class:Stoner.Data` modified so that the x and y columns match *other*
            if *replace* is True, or *opt_trans*,*trans_err*,*new_xy_data*. Where *opt_trans* is the optimum
            affine transformation, *trans_err* is a matrix giving the standard error in the transformation
            matrix components and  *new_xy_data* is an (n x 2) array of the transformed data.

    Example:
        .. plot:: samples/scale_curves.py
            :include-source:
            :outname: scale
    """
    _ = datafile._col_args(xcol=xcol, ycol=ycol)
    #
    # Sort out keyword srguments
    #
    bounds = kargs.pop("bounds", lambda x, r: True)
    otherbounds = kargs.pop("otherbounds", bounds)
    replace = kargs.pop("replace", True)
    headers = kargs.pop("headers", None)
    xmode = kargs.pop("xmode", "linear")
    ymode = kargs.pop("ymode", "linear")
    use_estimate = kargs.pop("use_estimate", False)

    # Get our working data from this DataFile and remove masked rows

    working = datafile.search(_.xcol, bounds)
    working = ma.mask_rowcols(working, axis=0)
    xdat = working[:, datafile.find_col(_.xcol)]
    ydat = working[:, datafile.find_col(_.ycol)]

    # Get data from the other. If it is already an ndarray, check size and dimensions

    if isinstance(other, datafile._baseclass):
        working2 = other.search(_.xcol, otherbounds)
        working2 = ma.mask_rowcols(working2, axis=0)
        xdat2 = working2[:, other.find_col(_.xcol)]
        ydat2 = working2[:, other.find_col(_.ycol)]
        if len(xdat2) != len(xdat):
            raise RuntimeError(f"Data lengths don't match {len(xdat)}!={len(xdat2)}")
    elif isinstance(other, np.ndarray):
        if other.ndim == 1:
            other = np.atleast_2d(other).T
        if other.shape[0] != len(xdat) or not 1 <= other.shape[1] <= 2:
            raise RuntimeError(
                (
                    "If other is a numpy array it must be the same length as the number of points to match "
                    + "to and 1 or 2 columns. (other shape={})"
                ).format(other.shape)
            )
        if other.shape[1] == 1:
            xdat2 = xdat
            ydat2 = other[:, 0]
        else:
            xdat2 = other[:, 0]
            ydat2 = other[:, 1]
    else:
        raise RuntimeError(f"other should be either a numpy array or subclass of DataFile, not a {type(other)}")

    # Need two nx2 arrays of points now

    xy1 = np.column_stack((xdat, ydat))
    xy2 = np.column_stack((xdat2, ydat2))

    # We're going to use three points to get an estimate for the affine transform to apply

    if isinstance(use_estimate, bool) and use_estimate:
        mid = int(len(xdat) / 2)
        try:  # may go wrong if three points are co-linear
            m0 = GetAffineTransform(xy1[[0, mid, -1], :], xy2[[0, mid, -1], :])
        except (RuntimeError, np.linalg.LinAlgError):  # So use an idnetify transformation instead
            m0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    elif isinstance(use_estimate, np.ndarray) and use_estimate.shape == (
        2,
        3,
    ):  # use_estimate is an initial value transformation
        m0 = use_estimate
    else:  # Don't try to be clever
        m0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    popt, perr, trans = _twoD_fit(xy1, xy2, xmode=xmode, ymode=ymode, m0=m0)
    data = datafile.data[:, [_.xcol, _.ycol]]
    new_data = trans(data)
    if replace:  # In place scaling, replace and return datafile
        datafile.metadata["Transform"] = popt
        datafile.metadata["Transform Err"] = perr
        datafile.data[:, _.xcol] = new_data[:, 0]
        datafile.data[:, _.ycol] = new_data[:, 1]
        if headers:
            datafile.column_headers[_.xcol] = headers[0]
            datafile.column_headers[_.ycol] = headers[1]
        ret = datafile
    else:  # Return results but don't change datafile.
        ret = popt, perr, new_data
    return ret


def smooth(datafile, window="boxcar", xcol=None, ycol=None, size=None, **kargs):
    """Smooth data by convoluting with a window.

    Args:
        window (string or tuple):
            Defines the window type to use by passing to :py:func:`scipy.signal.get_window`.

    Keyword Arguments:
        xcol(column index or None):
            Data to use as x data if needed to define a window. If None, use :py:attr:`Stoner.Core.DataFile.setas`
        ycvol (column index or None):
            Data to be smoothed
        size (int or float):
            If int, then the number of points to use in the smoothing window. If float, then the size in x-data
            to be used.
        result (bool or column index):
            Whether to add the smoothed data to the dataset and if so where.
        replace (bool):
            Replace the exiting data or insert as a new column.
        header (string):
            New column header for the new data.

    Returns:
        (datafile or array):
            If result is False, then the return value will be a copy of the smoothed data, otherwise the return
            value is a copy of the Stoner.Data object with the smoothed data added,

    Notes:
        If size is float, then it is necessary to map the X-data to a number of rows and to ensure that the data
        is evenly spaced in x. To do this, the number of rows in the window is found by dividing the span in x
        by the size and multiplying by the total lenfth. Then the data is interpolated to a new set of evenly
        space X over the same range, smoothed and then interpoalted back to the original x values.
    """
    _ = datafile._col_args(xcol=xcol, ycol=ycol)
    replace = kargs.pop("replace", True)
    result = kargs.pop("result", True)  # overwrite existing y column data
    header = kargs.pop("header", datafile.column_headers[_.ycol])

    # Sort out window size
    if isinstance(size, float):
        interp_data = True
        xl, xh = datafile.span(_.xcol)
        size = int(np.ceil((size / (xh - xl)) * len(datafile)))
        nx = np.linspace(xl, xh, len(datafile))
        data = datafile.interpolate(nx, kind="linear", xcol=_.xcol, replace=False)
        datafile["Smoothing window size"] = size
    elif isinstance(size, int_types):
        data = copy(datafile.data)
        interp_data = False
    else:
        raise ValueError(f"size should either be a float or integer, not a {type(size)}")

    window = get_window(window, size)
    # Handle multiple or single y columns
    if not isiterable(_.ycol):
        _.ycol = [_.ycol]

    # Do the convolution itdatafile
    for yc in _.ycol:
        data[:, yc] = convolve(data[:, yc], window, mode="same") / size

    # Reinterpolate the smoothed data back if necessary
    if interp_data:
        nx = datafile.data[:, _.xcol]
        tmp = datafile.clone
        tmp.data = data
        data = tmp.interpolate(nx, kind="linear", xcol=_.xcol, replace=False)

    # Fix return value
    if isinstance(result, bool) and not result:
        return data[:, _.ycol]
    for yc in _.ycol:
        datafile.add_column(data[:, yc], header=header, index=result, replace=replace)
    return datafile


def spline(datafile, xcol=None, ycol=None, sigma=None, **kargs):
    """Construct a spline through x and y data and replace, add new data or return spline function.

    Keyword Arguments:
        xcol (column index):
            Column with x data or if None, use setas attribute.
        ycol (column index):
            Column with y data or if None, use the setas attribute
        sigma (column index, or array of data):
            Column with weights, or if None use the 1/yerr column.
        replace (Boolean or column index or None):
            If True then the y-column data is repalced, if a column index then the
            new data is added after the specified index, if False then the new y-data is returned and if None,
            then spline object is returned.
        header (string):
            If *replace* is True or a column index then use this string as the new column header.
        order (int):
            The order of spline to use (1-5)
        smoothing (float or None):
            The smoothing factor to use when fitting the spline. A value of zero will create an
            interpolating spline.
        bbox (tuple of length 2):
            Bounding box for the spline - defaults to range of x values
        ext (int or str):
            How to extrapolate, default is "extrapolate", but can also be "raise","zeros" or "const".

    Returns:
        (various):
            Depending on the value of *replace*, returns a copy of the Stoner.Data, a 1D numpy array of
            data or an :[y:class:`scipy.interpolate.UniverateSpline` object.

    This is really just a pass through to the scipy.interpolate.UnivariateSpline function. Also used in the
    extrapolate function.
    """
    _ = datafile._col_args(xcol=xcol, ycol=ycol)
    if sigma is None and (isnone(_.yerr) or _.yerr):
        if not isnone(_.yerr):
            sigma = 1.0 / (datafile // _.yerr)
        else:
            sigma = np.ones(len(datafile))
    replace = kargs.pop("replace", True)
    result = kargs.pop("result", True)  # overwrite existing y column data
    header = kargs.pop("header", datafile.column_headers[_.ycol])
    k = kargs.pop("order", 3)
    s = kargs.pop("smoothing", None)
    bbox = kargs.pop("bbox", [None] * 2)
    ext = kargs.pop("ext", "extrapolate")
    x = datafile // _.xcol
    y = datafile // (_.ycol)
    spline = UnivariateSpline(x, y, w=sigma, bbox=bbox, k=k, s=s, ext=ext)
    new_y = spline(x)

    if header is None:
        header = datafile.column_headers[_.ycol]

    if not (result is None or (isinstance(result, bool) and not result)):
        datafile.add_column(new_y, header, index=result, replace=replace)
        return datafile
    if result is None:
        return new_y
    return spline
