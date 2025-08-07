#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filtering and smoothing functions for analysis code."""

__all__ = ["FilteringOpsMixin"]

from copy import deepcopy as copy
from warnings import warn

import numpy as np
from numpy import ma
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import get_window, convolve, savgol_filter

from Stoner.tools import isiterable, isnone
from Stoner.compat import int_types, string_types, get_func_params

from .utils import outlier as _outlier, _twoD_fit, GetAffineTransform


def _bin_weighted(x, y_vals, bin_edges, y_errs=None):
    """Do the actual work of binning the data.

    Args:
        x (1D array of N float):
            x data values
        y_vals (2D array (N,m) floats):
            Y data values
        bin_edges (1D array of n+1 float):
            Edges of the new x value bins

    Keyword Arguments:
        y_errs (2D array (N,m) floats or None):
            uncertainities in y values. Default value is None in which case all vbalues equally weighted.

    Returns:
        tuple[1D array of x, 2D array of y, 2D array of y_errs, binned_counts]
    """
    n_bins = bin_edges.size - 1
    m = y_vals.shape[1]
    if y_errs is None:
        y_errs = np.ones_like(y_vals)

    # Compute bin indices for each x
    bin_indices = np.digitize(x, bin_edges) - 1

    # Mask out-of-range values
    valid = (bin_indices >= 0) & (bin_indices < n_bins)
    x = x[valid]
    y_vals = y_vals[valid]
    y_errs = y_errs[valid]
    bin_indices = bin_indices[valid]

    # Deal with 0s in y_errs
    min_y_err = 1 if y_errs.max() == 0 else y_errs.max() * 1e-8
    y_errs = np.where(y_errs == 0, min_y_err, y_errs)

    # Compute weights: inverse variance
    weights = 1.0 / (y_errs**2)

    # Prepare output arrays
    binned_y = np.zeros((n_bins, m))
    binned_err = np.zeros((n_bins, m))
    binned_counts = np.zeros((n_bins, m))

    # Use bincount for each column
    for j in range(m):
        w = weights[:, j]
        yw = y_vals[:, j] * w

        sum_w = np.bincount(bin_indices, weights=w, minlength=n_bins)
        sum_w[sum_w <= 0] = np.nan
        sum_yw = np.bincount(bin_indices, weights=yw, minlength=n_bins)
        binned_counts[:, j] = np.bincount(bin_indices, minlength=n_bins)

        binned_y[:, j] = sum_yw / sum_w

        # Calculate the weighted uncertainity
        variance = (y_vals[:, j] - binned_y[:, j][bin_indices]) ** 2

        variance_n = variance / binned_counts[:, j][bin_indices] * w / sum_w[bin_indices]
        variance_b = np.bincount(bin_indices, weights=variance_n, minlength=n_bins)
        std_err = np.sqrt(variance_b)
        std_err[np.isnan(std_err)] = 0.0

        binned_err[:, j] = std_err

    # Bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_centers, binned_y, binned_err, binned_counts


class FilteringOpsMixin:
    """Provide additional filtering sndsmoothing methods to :py:class:`Stoner.Data`."""

    def SG_Filter(
        self, col=None, xcol=None, points=15, poly=1, order=0, pad=True, result=None, replace=False, header=None
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
            (numpy array or self):
                If result is None, a numpy array representing the smoothed or differentiated data is returned.
                Otherwise, a copy of the modified AnalysisMixin object is returned.

        Notes:
            If col is not specified or is None then the :py:attr:`DataFile.setas` column assignments are used
            to set an x and y column. If col is a tuple, then it is assumed to specify and x-column and y-column
            for differentiating data. This is now a pass through to :py:func:`scipy.signal.savgol_filter`

            Padding can help stop wildly wrong artefacts in the data at the start and enf of the data, particularly
            when the differential order is >1.

        See Also:
            User guide section :ref:`smoothing_guide`
        """
        from Stoner.Util import ordinal

        points = int(points)
        if points % 2 == 0:  # Ensure window length is odd
            points += 1

        _ = self._col_args(scalar=False, ycol=col, xcol=xcol)

        if _.xcol is not None:
            if not isinstance(_.xcol, list):
                col = _.ycol + [_.xcol]
            else:
                col = _.ycol + _.xcol
            data = self.column(col).T
        else:
            col = _.ycol
            data = self.column(list(col)).T
            data = np.vstack((data, np.arange(data.shape[1])))

        ddata = savgol_filter(
            data, window_length=points, polyorder=poly, deriv=order, mode="interp", axis=len(data.shape) - 1
        )
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
                    header.append(f"{self.column_headers[column]} after {ordinal(order)} order Savitsky-Golay Filter")
            else:
                header = [header] * (len(col) - 1)
            if r.shape[0] > len(header):
                iterdata = r[: len(header)]
            else:
                iterdata = r
            for column, head in zip(iterdata, header):
                self.add_column(column.ravel(), header=head, index=result, replace=replace)
            return self
        return r

    def bin(self, xcol=None, ycol=None, bins=0.03, mode="log", clone=True, **kargs):
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
                Return a clone of the current AnalysisMixin with binned data (True)
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
            cols = self.setas._get_cols()
            if xcol is None:
                xcol = cols["xcol"]
            if ycol is None:
                ycol = cols["ycol"]
        yerr = kargs.pop("yerr", cols["yerr"] if cols["has_yerr"] else None)

        bin_edges, bin_centres = self.make_bins(xcol, bins, mode, **kargs)

        xcol = self.find_col(xcol)
        ycol = self.find_col(ycol, force_list=True)
        if yerr is not None:
            yerr = self.find_col(yerr, force_list=True)
            yerr = self.data[:, yerr]

        bin_centres, y_vals, y_errs, bin_counts = _bin_weighted(
            self.data[:, xcol], self.data[:, ycol], bin_edges, yerr
        )

        if not clone:
            return bin_centres, y_vals, y_errs, bin_counts

        ret = self.clone
        ret.data = np.zeros((len(bin_centres), 3 * y_vals.shape[1]))
        ret.data[:, 0] = bin_centres
        ret.data[:, 1::3] = y_vals
        ret.data[:, 2::3] = y_errs
        ret.data[:, 3::3] = bin_counts

        columns = np.zeros(ret.data.shape[1], dtype=str)
        columns[0] = self.column_headers[xcol]
        columns[1::3] = self.column_headers[ycol]
        columns[2::3] = [f"d{h}" for h in self.column_headers[ycol]]
        columns[3::3] = [f"#/bin {h}" for h in self.column_headers[ycol]]
        setas = np.ones_like(columns, dtype=str)
        setas[0] = "x"
        setas[1::3] = "y"
        setas[2::3] = "e"
        setas[3::3] = "."
        ret.setas = list(setas)
        ret.column_headers = columns

        return ret

    def deduplicate(self, col, action="average", clone=True):
        """Remove rows with duplicated values in the given column.

        Args:
            col (Index type):
                Column to look for duplicate values.

        Keyword Arguments:
            action (str):
                What to do with duplicate values:
                    - *average* - work out the average of the rows which are duplicated
                    - *median* - work out the median value of all the rows with duplicate search column values
                    - *first*, *last* - use the first or last value of duplicated search column values
            clone (bool):
                Return a clone of the current AnalysisMixin with depuplicated data (True)
                or just the data (False).

        Returns:
            (:py:class:`Stoner.Data` or tuple of 4 array-like):
                Either a clone of the current data set with the depuplciated data, or just a data array.
        """
        cols = self.find_col(col, force_list=True)
        idx = []
        for row in self.data[:, cols]:
            if row.size > 1:
                idx.append((x for x in row))
            else:
                idx.append(row)
        idx = np.array(idx)
        vals, rev, idy, nums = np.unique(idx, return_index=True, return_inverse=True, return_counts=True)

        select = np.zeros_like(idx, dtype=bool)
        select[rev] = True
        ix = np.arange(len(self))
        vals = vals[nums > 1]
        nums = nums[nums > 1]
        data = self.data.copy()
        for val, i, num in zip(vals, idx, nums):
            subset = data[idx == val]
            indices = ix[idx == val]
            match action:
                case "average":
                    data[indices] = np.average(subset, axis=0)
                case "median":
                    data[indices] = np.median(subset, axis=0)
                case "first":
                    data[indices] = data[indices.min()]
                case "last":
                    data[indices] = data[indices.max()]
                case _:
                    raise ValueError(f"Unknown deduplication action {action}")
        if clone:
            ret = self.clone
            ret.data = self.data[select, :]
            return ret
        return self.data[select, :]

    def extrapolate(self, new_x, xcol=None, ycol=None, yerr=None, overlap=20, kind="linear", errors=None):
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
            the other parameters (i.e. as with :py:meth:`AnalysisMixin.curve_fit`).
        """
        _ = self._col_args(xcol=xcol, ycol=ycol, yerr=yerr, scalar=False)
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
        work = self.clone
        for ix, x in enumerate(new_x):
            r = self.closest(x, xcol=_.xcol)
            match overlap:
                case int():
                    if (r.i - overlap / 2) < 0:
                        ll = 0
                        hl = min(len(self), overlap)
                    elif (r.i + overlap / 2) > len(self):
                        hl = len(self)
                        ll = max(hl - overlap, 0)
                    else:
                        ll = r.i - overlap / 2
                        hl = r.i + overlap / 2
                    bounds = {"_i__between": (ll, hl)}
                    mid_x = (self[ll, _.xcol] + self[hl - 1, _.xcol]) / 2.0
                case float():
                    if (r[_.xcol] - overlap / 2) < self.min(_.xcol)[0]:
                        ll = self.min(_.xcol)[0]
                        hl = ll + overlap
                    elif (r[_.xcol] + overlap / 2) > self.max(_.xcol)[0]:
                        hl = self.max(_.xcol)[0]
                        ll = hl - overlap
                    else:
                        ll = r[_.xcol] - overlap / 2
                        hl = r[_.xcol] + overlap / 2
                    bounds = {f"{self.column_headers[_.xcol]}__between": (ll, hl)}
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

    def interpolate(self, newX, kind="linear", xcol=None, replace=False):
        """Interpolate a dataset to get a new set of values for a given set of x data.

        Args:
            ewX (1D array or None):
                Row indices or X column values to interpolate with. If None, then the
                :py:meth:`AnalysisMixin.interpolate` returns an interpolation function. Unlike the raw interpolation
                function from scipy, this interpolation function will work with MaskedArrays by compressing them
                first.

        Keyword Arguments:
            kind (string):
                Type of interpolation function to use - does a pass through from numpy. Default is linear.
            xcol (index or None):
                Column index or label that contains the data to use with newX to determine which rows to return.
                Defaults to None.
            replace (bool):
                If true, then the current AnalysisMixin's data is replaced with the  newly interpolated data and the
                current AnalysisMixin is returned.

        Returns:
            (2D numpy array):
                Section of the current object's data if replace is False(default) or the modofied AnalysisMixin if
                replace is true.

        Note:
            Returns complete rows of data corresponding to the indices given in newX. if xcol is None, then newX is
            interpreted as (fractional) row indices. Otherwise, the column specified in xcol is thresholded with the
            values given in newX and the resultant row indices used to return the data.

            If the positional argument, newX is None, then the return value is an interpolation function. This
            interpolation function takes one argument - if *xcol* was None, this argument is interpreted as
            array indices, but if *xcol* was specified, then this argument is interpreted as an array of xvalues.
        """
        DataArray = type(self.data)  # pylint: disable=E0203
        lines = np.shape(self.data)[0]  # pylint: disable=E0203
        index = np.arange(lines)
        if xcol is None:
            xcol = self.setas._get_cols("xcol")
        elif isinstance(xcol, bool) and not xcol:
            xcol = None

        if isinstance(newX, ma.MaskedArray):
            newX = newX.compressed()

        if xcol is not None and newX is not None:  # We need to convert newX to row indices
            xfunc = interp1d(self.column(xcol), index, kind, 0)  # xfunc(x) returns partial index
            newX = xfunc(newX)
        inter = interp1d(index, self.data, kind, 0)  # pylint: disable=E0203

        if newX is None:  # Ok, we're going to return an interpolation function

            def wrapper(newX):
                """Wrap the interpolation function."""
                if isinstance(newX, ma.MaskedArray):
                    newX = newX.compressed()
                else:
                    newX = np.array(newX)
                if xcol is not None and newX is not None:  # We need to convert newX to row indices
                    xfunc = interp1d(self.column(xcol), index, kind, 0)  # xfunc(x) returns partial index
                    newX = xfunc(newX)
                return inter(newX)

            return wrapper

        if replace:
            self.data = inter(newX)
            ret = self
        else:
            ret = DataArray(inter(newX), isrow=True)
            ret.setas = self.setas.clone
        return ret

    def make_bins(self, xcol, bins, mode="lin", **kargs):
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
        xmin = kargs.pop("bin_start", (self // xcol).min())
        xmax = kargs.pop("bin_sop", (self // xcol).max())

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
            elif mode.lower().startswith("spa"):  # Equal spacing mode
                xdata = np.array(sorted(self // xcol))[
                    [int(x) for x in np.round(np.linspace(0, len(self) - 1, bins + 1))]
                ]
                bin_start = xdata[:-1]
                bin_stop = xdata[1:]
                bin_centres = (bin_start + bin_stop) / 2
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
        if len(bin_start) > len(self):
            raise ValueError("Attempting to bin into more bins than there is data.")
        bin_edges = np.append(bin_start, bin_stop[-1])
        return bin_edges, bin_centres

    def outlier_detection(
        self, column=None, window=7, shape="boxcar", certainty=3.0, action="mask", width=1, func=None, **kargs
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
        _ = self._col_args(scalar=True, ycol=column, **kargs)
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
        index = np.zeros(len(self), dtype=bool)
        for i, t in enumerate(self.rolling_window(window, wrap=False, exclude_centre=width)):
            index[i] = func(self.data[i], t, metric=certainty, **kargs)
        self["outliers"] = np.arange(len(self))[index]  # add outlier indices to metadata
        if action == "mask" or action == "mask row":
            if action == "mask":
                self.mask[index, column] = True
            else:
                self.mask[index, :] = True
        elif action == "delete":
            self.data = self.data[~index]
        elif callable(action):  # this will call the action function with each row in turn from back to start
            for i in np.arange(len(self))[index][::-1]:
                action(i, column, self.data, *action_args, **action_kargs)
        return self

    def scale(self, other, xcol=None, ycol=None, **kargs):
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
                Whether to map the x,y data to the new coordinates and return a copy of this AnalysisMixin (true)
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
        _ = self._col_args(xcol=xcol, ycol=ycol)
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

        working = self.search(_.xcol, bounds)
        working = ma.mask_rowcols(working, axis=0)
        xdat = working[:, self.find_col(_.xcol)]
        ydat = working[:, self.find_col(_.ycol)]

        # Get data from the other. If it is already an ndarray, check size and dimensions

        if isinstance(other, self._baseclass):
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
        data = self.data[:, [_.xcol, _.ycol]]
        new_data = trans(data)
        if replace:  # In place scaling, replace and return self
            self.metadata["Transform"] = popt
            self.metadata["Transform Err"] = perr
            self.data[:, _.xcol] = new_data[:, 0]
            self.data[:, _.ycol] = new_data[:, 1]
            if headers:
                self.column_headers[_.xcol] = headers[0]
                self.column_headers[_.ycol] = headers[1]
            ret = self
        else:  # Return results but don't change self.
            ret = popt, perr, new_data
        return ret

    def smooth(self, window="boxcar", xcol=None, ycol=None, size=None, **kargs):
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
            (self or array):
                If result is False, then the return value will be a copy of the smoothed data, otherwise the return
                value is a copy of the AnalysisMixin object with the smoothed data added,

        Notes:
            If size is float, then it is necessary to map the X-data to a number of rows and to ensure that the data
            is evenly spaced in x. To do this, the number of rows in the window is found by dividing the span in x
            by the size and multiplying by the total lenfth. Then the data is interpolated to a new set of evenly
            space X over the same range, smoothed and then interpoalted back to the original x values.
        """
        _ = self._col_args(xcol=xcol, ycol=ycol)
        replace = kargs.pop("replace", True)
        result = kargs.pop("result", True)  # overwrite existing y column data
        header = kargs.pop("header", self.column_headers[_.ycol])

        # Sort out window size
        if isinstance(size, float):
            interp_data = True
            xl, xh = self.span(_.xcol)
            size = int(np.ceil((size / (xh - xl)) * len(self)))
            nx = np.linspace(xl, xh, len(self))
            data = self.interpolate(nx, kind="linear", xcol=_.xcol, replace=False)
            self["Smoothing window size"] = size
        elif isinstance(size, int_types):
            data = copy(self.data)
            interp_data = False
        else:
            raise ValueError(f"size should either be a float or integer, not a {type(size)}")

        window = get_window(window, size)
        # Handle multiple or single y columns
        if not isiterable(_.ycol):
            _.ycol = [_.ycol]

        # Do the convolution itself
        for yc in _.ycol:
            data[:, yc] = convolve(data[:, yc], window, mode="same") / size

        # Reinterpolate the smoothed data back if necessary
        if interp_data:
            nx = self.data[:, _.xcol]
            tmp = self.clone
            tmp.data = data
            data = tmp.interpolate(nx, kind="linear", xcol=_.xcol, replace=False)

        # Fix return value
        if isinstance(result, bool) and not result:
            return data[:, _.ycol]
        for yc in _.ycol:
            self.add_column(data[:, yc], header=header, index=result, replace=replace)
        return self

    def spline(self, xcol=None, ycol=None, sigma=None, **kargs):
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
                Depending on the value of *replace*, returns a copy of the AnalysisMixin, a 1D numpy array of
                data or an :[y:class:`scipy.interpolate.UniverateSpline` object.

        This is really just a pass through to the scipy.interpolate.UnivariateSpline function. Also used in the
        extrapolate function.
        """
        _ = self._col_args(xcol=xcol, ycol=ycol)
        if sigma is None and (isnone(_.yerr) or _.yerr):
            if not isnone(_.yerr):
                sigma = 1.0 / (self // _.yerr)
            else:
                sigma = np.ones(len(self))
        replace = kargs.pop("replace", True)
        result = kargs.pop("result", True)  # overwrite existing y column data
        header = kargs.pop("header", self.column_headers[_.ycol])
        k = kargs.pop("order", 3)
        s = kargs.pop("smoothing", None)
        bbox = kargs.pop("bbox", [None] * 2)
        ext = kargs.pop("ext", "extrapolate")
        x = self // _.xcol
        y = self // (_.ycol)
        spline = UnivariateSpline(x, y, w=sigma, bbox=bbox, k=k, s=s, ext=ext)
        new_y = spline(x)

        if header is None:
            header = self.column_headers[_.ycol]

        if not (result is None or (isinstance(result, bool) and not result)):
            self.add_column(new_y, header, index=result, replace=replace)
            return self
        if result is None:
            return new_y
        return spline
