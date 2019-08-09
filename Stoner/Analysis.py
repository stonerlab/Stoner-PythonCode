"""Stoner .Analysis provides a subclass of :class:`.Data` that has extra analysis routines builtin.
"""

__all__ = ["AnalysisMixin", "GetAffineTransform", "ApplyAffineTransform"]
from warnings import warn
import numpy as _np_
import numpy.ma as ma

from scipy.integrate import cumtrapz
from scipy.signal import get_window, convolve
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import curve_fit, newton
from scipy.signal import savgol_filter

from .compat import python_v3, string_types, int_types, index_types, get_func_params
from .tools import isNone, isiterable, all_type, istuple
from .core.exceptions import assertion
from copy import deepcopy as copy

# from matplotlib.pylab import * #Surely not?
if python_v3:
    from inspect import getfullargspec
else:
    from inspect import getargspec as getfullargspec


# ==========================================================================================================================================
# Module Private Functions
# ==========================================================================================================================================


def _outlier(row, window, metric, ycol=None):
    """Internal function for outlier detector.

    Calculates if the current row is an outlier from the surrounding data by looking
    at the number of standard deviations away from the average of the window it is.

    Args:
        row (array): Single row of the dataset.
        window (array): Section of data surrounding the row being examined.
        metric (float): distance the current row is from the local mean.

    Keyword Arguments:
        ycol (column index or None): If set, specifies the column containing the data to check.

    Returns:
        (bool):
            If True, then the current row is an outlier from the local data.
    """
    av = _np_.average(window[:, ycol])
    std = _np_.std(window[:, ycol])  # standard deviation
    return abs(row[ycol] - av) > metric * std


def _threshold(threshold, data, rising=True, falling=False):
    """Internal function that implements the threshold method - also used in peak-finder

    Args:
        threshold (float):
            Threshold valuye in data to look for
        rising (bool):
            Find points where data is rising up past threshold
        falling (bool):
            Find points where data is falling below the threshold

    Returns:
        (array):
            Fractional indices where the data has crossed the threshold assuming a
            straight line interpolation between two points.
    """
    # First we find all points where we cross zero in the correct direction
    current = data
    previous = _np_.roll(current, 1)
    current = _np_.atleast_1d(current)
    index = _np_.arange(len(current))
    sdat = _np_.column_stack((index, current, previous))
    if rising and not falling:
        expr = lambda x: (x[1] >= threshold) & (x[2] < threshold)
    elif rising and falling:
        expr = lambda x: ((x[1] >= threshold) & (x[2] < threshold)) | ((x[1] <= threshold) & (x[2] > threshold))
    elif falling and not rising:
        expr = lambda x: (x[1] <= threshold) & (x[2] > threshold)
    else:
        expr = lambda x: False

    # Now we refine the estimate of zero crossing with a cubic interpolation
    # and use Newton's root finding method to locate the zero in the interpolated data

    intr = interp1d(index, data.ravel() - threshold, kind="cubic")
    roots = []
    for ix, x in enumerate(sdat):
        if ix > 0 and expr(x):  # There's a root somewhere here !
            try:
                roots.append(newton(intr, ix))
            except (ValueError, RuntimeError):  # fell off the end here
                pass
    return _np_.array(roots)


def _twoD_fit(xy1, xy2, xmode="linear", ymode="linear", m0=None):
    r"""Calculae an optimal transformation of points :math:`(x_1,y_1)\rightarrow(x_2,y_2)`.

    Arguments:
        xy1 ( n by 2 array of float):
            Set of points to be mapped from.
        xy2 ( n by 2 array of floats):
            Set of points to be mapped to.

    Keyword Arguments:
        xmode ('affine', 'linear', 'scale' 'offset' or 'fixed'):
            How to manipulate the x-data
        ymode ('linear', 'scale' 'offset' or 'fixed'):
            How to manipulate the y-data
        m0 (3x2 array):
            Initial and fixed values of the transformation. Defaults to using an identity transformation.

    Returns:
        (tuple of opt_trans,trans_err,mapping func)

    The most general case is an affine transform which includes rotation, scale, translation and skew. This is represented as a 2 x 3 matrix
    of coordinates. The *xmode* and *ymode* parameters control the possible operations to align the data in x and y directions, in addition
    to which the *xmode* parameter can take the value 'affine' which allows a full affine transformation. The returned values are the
    affine transformation matrix, the uncertainities in this and a function to map co-ordinates with the optimal affine transformation.

    Note:
        *m0* combines both giving an initial value and fixed values for the transformation. If *m0* is set, then it is used to provide initial
        balues of the free parameters. Which elelemnts of *m0* that are free parameters and which are fixed is determined by the *xmode*
        and *ymode* parameters. IF *xmode* and *ymode* are both fixed, however, no scaling is done at all.
    """
    if xy1.shape != xy2.shape or xy1.shape[1] != 2:
        raise RuntimeError(
            "co-ordinate arrays must be equal length with two columns, not {} and {}".format(xy1.shape, xy2.shape)
        )
    xvarp = {
        "affine": [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]],
        "linear": [[0, 0], [0, 2]],
        "scale": [[0, 0]],
        "offset": [[0, 2]],
        "fixed": [[]],
    }
    yvarp = {"linear": [[1, 1], [1, 2]], "scale": [[1, 1]], "offset": [[1, 2]], "fixed": [[]]}

    if xmode not in xvarp or ymode not in yvarp:
        raise RuntimeError(
            "xmode and ymode must be one of 'linear','scale','offset','fixed' not {} and {}".format(xmode, ymode)
        )

    if xmode == "affine":
        ymode = "fixed"

    xunknowns = len(xvarp[xmode])
    yunknowns = len(yvarp[ymode])
    if xunknowns + yunknowns == 0:  # shortcircuit for the trivial case
        return _np_.array([[1, 0], [1, 0]]), _np_.zeros((2, 2)), lambda x: x

    mapping = xvarp[xmode] + yvarp[ymode]
    mapping = [m for m in mapping if m != []]  # remove empty mappings
    data = _np_.column_stack((xy1, xy2)).T

    if isinstance(m0, list):
        m0 = _np_.array(m0)

    if m0 is None:
        p0s = {"affine": [1, 0, 0, 0, 1, 0], "linear": [1, 0], "scale": [1], "offset": [0], "fixed": []}
        p0 = p0s[xmode] + p0s[ymode]
        default = _np_.array([[1.00, 0.0, 0.0], [0.0, 1.0, 0.0]])
    elif isinstance(m0, _np_.ndarray) and m0.shape == (2, 3):
        p0 = [0] * len(mapping)
        for i, [u, v] in enumerate(mapping):
            p0[i] = m0[u, v]
            default = m0
    else:
        raise RuntimeError("m0 starting matrix should be a numpy array of size (2,3) not {}".format(m0))

    result = _np_.zeros(len(xy1))

    def transform(xy, *p):  # Construct the fitting function
        """Fitting function to find the transfoprm."""
        xy1 = _np_.column_stack((xy[:2, :].T, _np_.ones(xy.shape[1]))).T
        xy2 = xy[2:, :]
        for pi, (u, v) in zip(p, mapping):
            default[u, v] = pi
        xyt = _np_.dot(default, xy1)
        ret = _np_.sqrt(_np_.sum((xy2 - xyt) ** 2, axis=0))
        return ret

    popt, pcov = curve_fit(transform, data, result, p0=p0)
    perr = _np_.sqrt(_np_.diag(pcov))

    # Initialise the return values
    default = _np_.array([[1.00, 0.0, 0.0], [0.0, 1.0, 0.0]])
    for pi, (u, v) in zip(popt, mapping):
        default[u, v] = pi
    default_err = _np_.zeros((2, 3))
    for pi, (u, v) in zip(perr, mapping):
        default_err[u, v] = pi

    transform = lambda xy: ApplyAffineTransform(xy, default)

    return (default, default_err, transform)


def ApplyAffineTransform(xy, transform):
    """Apply a given afffine transform to a set of xy data points.


    Args:
        xy (n by 2 array):
            Set of x,y coordinates to be transformed
        transform (2 by 3 array):
            Affine transform matrix.

    Returns:
        (n x 2 array).
        Transformed co-ordinates.
    """
    xyt = _np_.row_stack((xy.T, _np_.ones(len(xy))))
    xyt = _np_.dot(transform, xyt)
    return xyt.T


def GetAffineTransform(p, pd):
    """Calculate an affine transofrm from 2 sets of three points.

    Args:
        p (3x2 array):
            Coordinates of points to transform from.
        pd (3x2 array):
            Cooridinates of points to transform to.

    Returns:
        (2x3 array):
            matrix representing the affine transform.
    """
    if _np_.shape(p) != (3, 2) and _np_.shape(pd) != (3, 2):
        raise RuntimeError("Must supply three points")

    p = _np_.append(p, _np_.atleast_2d(_np_.ones(3)).T, axis=1)
    transform = _np_.linalg.solve(p, pd)
    return transform.T


def _poly_outlier(row, window, metric=3.0, ycol=None, xcol=None, order=1, yerr=None):
    """Alternative outlier detection function that fits a polynomial locally over the window.

    Args:
        row (1D array):
            Current row of data
        column int):
            Column index of y values to examine
        window (2D array):
            Local window of data

    Keyyword Arguments:
        metric (float):
            Some measure of how sensitive the dection should be
        xcol (column index):
            Column of data to use for X values. Defaults to current setas value
        order (int):
            Order of polynomial to fit. Must be < length of window-1

    Returns:
        (bool):
            True if current row is an outlier
    """
    if order > window.shape[0] - 2:
        raise ValueError("order should be smaller than the window length. {} vs {}".format(order, window.shape[0] - 2))

    x = window[:, xcol] - row[xcol]
    y = window[:, ycol]
    if yerr:
        w = 1.0 / window[:, yerr]
    else:
        w = None

    popt, pcov = _np_.polyfit(x, y, w=w, deg=order, cov=True)
    pval = _np_.polyval(popt, 0.0)
    perr = _np_.sqrt(_np_.diag(pcov))[-1]
    return (pval - row[ycol]) ** 2 > metric * perr


def __threshold(threshold, data, rising=True, falling=False):
    """Internal function that implements the threshold method - also used in peak-finder

    Args:
        threshold (float):
            Threshold valuye in data to look for
        rising (bool):
            Find points where data is rising up past threshold
        falling (bool):
            Find points where data is falling below the threshold

    Returns:
        (array):
            Fractional indices where the data has crossed the threshold assuming a
            straight line interpolation between two points.
    """
    # First we find all points where we cross zero in the correct direction
    current = data
    mask = ma.getmaskarray(data)
    previous = _np_.roll(current, 1)
    index = _np_.arange(len(current))
    sdat = _np_.column_stack((index, current, previous))
    if rising and not falling:
        expr = lambda x: (x[1] >= threshold) & (x[2] < threshold)
    elif rising and falling:
        expr = lambda x: ((x[1] >= threshold) & (x[2] < threshold)) | ((x[1] <= threshold) & (x[2] > threshold))
    elif not rising and falling:
        expr = lambda x: (x[1] <= threshold) & (x[2] > threshold)
    else:
        expr = lambda x: False

    current = ma.masked_array(current)
    current.mask = mask
    # Now we refine the estimate of zero crossing with a cubic interpolation
    # and use Newton's root finding method to locate the zero in the interpolated data

    intr = interp1d(index, data - threshold, kind="cubic")
    roots = []
    for ix, x in enumerate(sdat):
        if expr(x) and ix > 0 and ix < len(data) - 1:  # There's a root somewhere here !
            try:
                roots.append(newton(intr, ix))
            except ValueError:  # fell off the end here
                pass
    return _np_.array(roots)


class AnalysisMixin(object):

    """A mixin calss designed to work with :py:class:`Stoner.Core.DataFile` to provide additional analysis methods."""

    def SG_Filter(
        self, col=None, xcol=None, points=15, poly=1, order=0, pad=True, result=None, replace=False, header=None
    ):
        """Implements Savitsky-Golay filtering of data for smoothing and differentiating data.

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
                Pad the start and end of the array with the mean value (True, default) or specired value (float) or leave as is.
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
            to set an x and y column. If col is a tuple, then it is assumed to secify and x-column and y-column
            for differentiating data. This is now a pass through to :py:func:`scipy.signal.savgol_filter`

            Padding can help stop wildly wrong artefacts in the data at the start and enf of the data, particularly when the differntial order is >1.

        See Also:
            User guide section :ref:`smoothing_guide`
        """
        from Stoner.Util import ordinal

        points = int(points)
        if points % 2 == 0:  # Ensure window length is odd
            points += 1

        _ = self._col_args(scalar=False, ycol=col, xcol=xcol)

        if _.xcol is not None:
            col = _.ycol + [_.xcol]
            data = self.column(list(col)).T
        else:
            col = _.ycol
            data = self.column(list(col)).T
            data = _np_.row_stack((data, _np_.arange(data.shape[1])))

        ddata = savgol_filter(data, window_length=points, polyorder=poly, deriv=order, mode="interp")
        if isinstance(pad, bool) and pad:
            offset = int(points * order ** 2 / 8)
            padv = _np_.mean(ddata[:, offset:-offset], axis=1)
            pad = _np_.ones((ddata.shape[0], offset))
            for ix, v in enumerate(padv):
                pad[ix] *= v
        elif isinstance(pad, float):
            offset = int(points / 2)
            pad = _np_.ones((ddata.shape[0], offset)) * pad

        if _np_.all(pad) and offset > 0:
            ddata[:, :offset] = pad
            ddata[:, -offset:] = pad
        if order >= 1:
            r = ddata[:-1] / ddata[-1]
        else:
            r = ddata[:, :-1]

        if result is not None:
            if not isinstance(header, string_types):
                header = []
                for column in col[:-1]:
                    header.append(
                        "{} after {} order Savitsky-Golay Filter".format(self.column_headers[column], ordinal(order))
                    )
            else:
                header = [header] * (len(col) - 1)
            for column, head in zip(r[:-1], header):
                self.add_column(column.ravel(), header=head, index=result, replace=replace)
            return self
        else:
            return r

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
        elif isinstance(col, _np_.ndarray) and len(col.shape) == 1 and len(col) == len(self):
            data = col
            name = "data"
        elif isinstance(col, float):
            data = col * _np_.ones(len(self))
            name = str(col)
        else:
            raise RuntimeError("Bad column index: {}".format(col))
        return data, name

    def __dir__(self):
        """Handles the local attributes as well as the inherited ones"""
        attr = dir(type(self))
        attr.extend(super(AnalysisMixin, self).__dir__())
        attr.extend(list(self.__dict__.keys()))
        attr = list(set(attr))
        return sorted(attr)

    def add(self, a, b, replace=False, header=None):
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
            err_calc = lambda adata, bdata, e1data, e2data: _np_.sqrt(e1data ** 2 + e2data ** 2)
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
        if err_calc is not None:
            err_data = err_calc(adata, bdata, e1data, e2data)
        self.add_column((adata + bdata), header=header, index=a, replace=replace)
        if err_calc is not None:
            self.add_column(err_data, header=err_header, index=a + 1, replace=False)
        return self

    def apply(self, func, col=None, replace=True, header=None, **kargs):
        """Applies the given function to each row in the data set and adds to the data set.

        Args:
            func (callable):
                The function to apply to each row of the data.
            col (index):
                The column in which to place the result of the function

        Keyword Arguments:
            replace (bool):
                Either replace the existing column/complete data or create a new column or data file.
            header (string or None):
                The new column header(s) (defaults to the name of the function func

        Note:
            If any extra keyword arguments are supplied then these are passed to the function directly. If
            you need to pass any arguments that overlap with the keyword arguments to :py:math:`AnalysisMixin.apply`
            then these can be supplied in a dictionary argument *_extra*.

            The callable *func* should have a signature::

                def func(row,**kargs):

            and should return either a single float, in which case it will be used to repalce the specified column, or an
            array, in which case it is used to completely replace the row of data.

            If the function returns a complete row of data, then the *replace* parameter will cause the return value to be
            a new datafile, leaving the original unchanged. The *headers* parameter can give the complete column headers for
            the new data file.

        Returns:
            (:py:class:`Stoner.Data`):
                The newly modified Data object.
        """
        if col is None:
            col = self.setas.get("y", [0])[0]
        col = self.find_col(col)
        kargs.update(kargs.pop("_extra", dict()))
        # Check the dimension of the output
        ret = func(next(self.rows()), **kargs)
        try:
            next(self.rows(reset=True))
        except (RuntimeError, StopIteration):
            pass
        if isiterable(ret):
            nc = _np_.zeros((len(self), len(ret)))
        else:
            nc = _np_.zeros(len(self))
        # Evaluate the data row by row
        for ix, r in enumerate(self.rows()):
            ret = func(r, **kargs)
            if isiterable(ret) and not isinstance(ret, _np_.ndarray):
                ret = _np_.ma.MaskedArray(ret)
            nc[ix] = ret
        # Work out how to handle the result
        if nc.ndim == 1:
            if header is None:
                header = func.__name__
            self = self.add_column(nc, header=header, index=col, replace=replace, setas=self.setas[col])
            ret = self
        else:
            if not replace:
                ret = self.clone
            else:
                ret = self
            ret.data = nc
            if header is not None:
                ret.column_headers = header
        return ret

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
        if "yerr" in kargs:
            yerr = kargs["yerr"]
        else:
            yerr = None

        if None in (xcol, ycol):
            cols = self.setas._get_cols()
            if xcol is None:
                xcol = cols["xcol"]
            if ycol is None:
                ycol = cols["ycol"]
            if "yerr" not in kargs and cols["has_yerr"]:
                yerr = cols["yerr"]

        bin_left, bin_right, bin_centres = self.make_bins(xcol, bins, mode, **kargs)

        ycol = self.find_col(ycol)
        if yerr is not None:
            yerr = self.find_col(yerr)

        ybin = _np_.zeros((len(bin_left), len(ycol)))
        ebin = _np_.zeros((len(bin_left), len(ycol)))
        nbins = _np_.zeros((len(bin_left), len(ycol)))
        xcol = self.find_col(xcol)
        i = 0

        for limits in zip(bin_left, bin_right):
            data = self.search(xcol, limits)
            if len(data) > 1:
                ok = _np_.logical_not(_np_.isnan(data.y))
                data = data[ok]
            elif len(data) == 1 and _np_.isnan(data.y):
                shape = list(data.shape)
                shape[0] = 0
                data = _np_.zeros(shape)
            if yerr is not None:
                w = 1.0 / data[:, yerr] ** 2
                W = _np_.sum(w, axis=0)
                if data.shape[0] > 3:
                    e = max(
                        _np_.std(data[:, ycol], axis=0) / _np_.sqrt(data.shape[0]),
                        (1.0 / _np_.sqrt(W)) / data.shape[0],
                    )
                else:
                    e = 1.0 / _np_.sqrt(W)
            else:
                w = _np_.ones((data.shape[0], len(ycol)))
                W = data.shape[0]
                if data[:, ycol].size > 1:
                    e = _np_.std(data[:, ycol], axis=0) / _np_.sqrt(W)
                else:
                    e = _np_.nan
            if data.shape[0] == 0:
                warn("Empty bin at {}".format(limits))
            y = _np_.sum(data[:, ycol] * (w / W), axis=0)
            ybin[i, :] = y
            ebin[i, :] = e
            nbins[i, :] = data.shape[0]
            i += 1
        if clone:
            ret = self.clone
            ret.data = _np_.atleast_2d(bin_centres).T
            ret.column_headers = [self.column_headers[xcol]]
            ret.setas = ["x"]
            for i in range(ybin.shape[1]):
                head = str(self.column_headers[ycol[i]])

                ret.add_column(ybin[:, i], header=head)
                ret.add_column(ebin[:, i], header="d{}".format(head))
                ret.add_column(nbins[:, i], header="#/bin {}".format(head))
                s = list(ret.setas)
                s[-3:] = ["y", "e", "."]
                ret.setas = s
        else:
            ret = (bin_centres, ybin, ebin, nbins)
        return ret

    def clip(self, clipper, column=None):
        """Clips the data based on the column and the clipper value.

        Args:
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
            col = self.setas._get_cols("ycol")
        else:
            col = self.find_col(column)
        clipper = (min(clipper), max(clipper))
        self = self.del_rows(col, lambda x, y: x < clipper[0] or x > clipper[1])
        return self

    def decompose(self, xcol=None, ycol=None, sym=None, asym=None, replace=True, **kwords):
        """Given (x,y) data, decomposes the y part into symmetric and antisymmetric contributions in x.

        Keyword Arguments:
            xcol (index):
                Index of column with x data - defaults to first x column in self.setas
            ycol (index or list of indices):
                indices of y column(s) data
            sym (index):
                Index of column to place symmetric data in default, append to end of data
            asym (index):
                Index of column for asymmetric part of ata. Defaults to appending to end of data
            replace (bool):
                Overwrite data with output (true)

        Returns:
            self: The newly modified :py:class:`AnalysisMixin`.

        Example:
            .. plot:: samples/decompose.py
                :include-source:
                :outname: decompose
        """
        if xcol is None and ycol is None:
            if "_startx" in kwords:
                startx = kwords["_startx"]
                del kwords["_startx"]
            else:
                startx = 0
            cols = self.setas._get_cols(startx=startx)
            xcol = cols["xcol"]
            ycol = cols["ycol"]
        xcol = self.find_col(xcol)
        ycol = self.find_col(ycol)
        if isinstance(ycol, list):
            ycol = ycol[0]  # FIXME should work with multiple output columns
        pxdata = self.search(xcol, lambda x, r: x > 0, xcol)
        xdata = _np_.sort(_np_.append(-pxdata, pxdata))
        self.data = self.interpolate(xdata, xcol=xcol)
        ydata = self.data[:, ycol]
        symd = (ydata + ydata[::-1]) / 2
        asymd = (ydata - ydata[::-1]) / 2
        if sym is None:
            self &= symd
            self.column_headers[-1] = "Symmetric Data"
        else:
            self.add_column(symd, header="Symmetric Data", index=sym, replace=replace)
        if asym is None:
            self &= asymd
            self.column_headers[-1] = "Asymmetric Data"
        else:
            self.add_column(asymd, header="Symmetric Data", index=asym, replace=replace)

        return self

    def diffsum(self, a, b, replace=False, header=None):
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
            err_calc = lambda adata, bdata, e1data, e2data: _np_.sqrt(
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
        self.add_column((adata - bdata) / (adata + bdata), header=header, index=a, replace=replace)
        if err_calc is not None:
            self.add_column(err_data, header=err_header, index=a + 1, replace=False)
        return self

    def divide(self, a, b, replace=False, header=None):
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
            err_calc = (
                lambda adata, bdata, e1data, e2data: _np_.sqrt((e1data / adata) ** 2 + (e2data / bdata) ** 2)
                * adata
                / bdata
            )
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
        if err_calc is not None:
            err_data = err_calc(adata, bdata, e1data, e2data)
        self.add_column((adata / bdata), header=header, index=a, replace=replace)
        if err_calc is not None:
            self.add_column(err_data, header=err_header, index=a + 1, replace=False)
        return self

    def extrapolate(self, new_x, xcol=None, ycol=None, yerr=None, overlap=20, kind="linear"):
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

        Returns:
            (array):
                Extrapolated values.

        Note:
            If the new_x values lie outside the span of the x-data, then the nearest *overlap* portion of the data is used
            to estimate the values. If the new_x values are within the span of the x-data then the portion of the data
            centred about the point and overlap points long will be used to interpolate a value.

            If *kind* is callable, it should take x values in the first parameter and free fitting parameters as the other
            parameters (i.e. as with :py:meth:`AnalysisMixin.curve_fit`).
        """
        _ = self._col_args(xcol=xcol, ycol=ycol, yerr=yerr, scalar=False)
        kinds = {
            "linear": lambda x, m, c: m * x + c,
            "quadratic": lambda x, a, b, c: a * x ** 2 + b * x + c,
            "cubic": lambda x, a, b, c, d: a * x ** 3 + b * x ** 2 + c * x + d,
        }
        errs = {
            "linear": lambda x, me, ce: _np_.sqrt((me * x) ** 2 + ce ** 2),
            "quadratic": lambda x, ae, be, ce: _np_.sqrt((2 * x ** 2 * ae) ** 2 + (x * be) ** 2 + ce ** 2),
            "cubic": lambda x, ae, be, ce, de: _np_.sqrt(
                (3 * ae * x ** 3) ** 2 + (2 * x ** 2 * be) ** 2 + (x * ce) ** 2 + de ** 2
            ),
        }

        if callable(kind):
            pass
        elif kind in kinds:
            kindf = kinds[kind]
        else:
            raise RuntimeError("Failed to recognise extrpolation function '{}'".format(kind))
        scalar_x = not isiterable(new_x)
        if scalar_x:
            new_x = [new_x]
        if isinstance(new_x, ma.MaskedArray):
            new_x = new_x.compressed
        results = _np_.zeros((len(new_x), 2 * len(_.ycol)))
        work = self.clone
        for ix, x in enumerate(new_x):
            r = self.closest(x, xcol=_.xcol)
            if isinstance(overlap, int):
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
            elif isinstance(overlap, float):
                if (r[_.xcol] - overlap / 2) < self.min(_.xcol)[0]:
                    ll = self.min(_.xcol)[0]
                    hl = ll + overlap
                elif (r[_.xcol] + overlap / 2) > self.max(_.xcol)[0]:
                    hl = self.max(_.xcol)[0]
                    ll = hl - overlap
                else:
                    ll = r[_.xcol] - overlap / 2
                    hl = r[_.xcol] + overlap / 2
                bounds = {"{}__between".format(self.column_headers[_.xcol]): (ll, hl)}
                mid_x = (ll + hl) / 2.0
            pointdata = work.select(**bounds)
            pointdata.data[:, _.xcol] = pointdata.column(_.xcol) - mid_x
            ret = pointdata.curve_fit(kindf, _.xcol, _.ycol, sigma=_.yerr, absolute_sigma=True)
            if isinstance(ret, tuple):
                ret = [ret]
            for iy, rt in enumerate(ret):
                popt, pcov = rt
                perr = _np_.sqrt(_np_.diag(pcov))
                results[ix, 2 * iy] = kindf(x - mid_x, *popt)
                results[ix, 2 * iy + 1] = errs[kind](x - mid_x, *perr)
        if scalar_x:
            results = results[0]
        return results

    def integrate(self, xcol=None, ycol=None, result=None, result_name=None, bounds=lambda x, y: True, **kargs):
        """Inegrate a column of data, optionally returning the cumulative integral.

        Args:
            xcol (index):
                The X data column index (or header)
            ycol (index)
            The Y data column index (or header)

        Keyword Arguments:
            result (index or None):
                Either a column index (or header) to overwrite with the cumulative data,
                or True to add a new column or None to not store the cumulative result.
            result_name (string):
                The new column header for the results column (if specified)
            bounds (callable):
                A function that evaluates for each row to determine if the data should be integrated over.
            **kargs:
                Other keyword arguements are fed direct to the scipy.integrate.cumtrapz method

        Returns:
            (:py:class:`Stoner.Data`):
                The newly modified Data object.

        Note:
            This is a pass through to the scipy.integrate.cumtrapz routine which just uses trapezoidal integration. A better alternative would be
            to offer a variety of methods including simpson's rule and interpolation of data. If xcol or ycol are not specified then
            the current values from the :py:attr:`Stoner.Core.DataFile.setas` attribute are used.
        """
        if xcol is None or ycol is None:
            cols = self.setas._get_cols()
            if xcol is None:
                xcol = cols["xcol"]
            if ycol is None:
                ycol = cols["ycol"]
        working = self.search(xcol, bounds)
        working = ma.mask_rowcols(working, axis=0)
        xdat = working[:, self.find_col(xcol)]
        ydat = working[:, self.find_col(ycol)]
        final = []
        for i in range(ydat.shape[1]):
            yd = ydat[:, i]
            resultdata = cumtrapz(xdat, yd, **kargs)
            resultdata = _np_.append(_np_.array([0]), resultdata)
            if result is not None:
                if isinstance(result, bool) and result:
                    self.add_column(resultdata, header=result_name, replace=False)
                else:
                    result_name = self.column_headers[self.find_col(result)]
                    self.add_column(resultdata, header=result_name, index=result, replace=(i == 0))
            final.append(resultdata[-1])
        if len(final) == 1:
            final = final[0]
        else:
            final = _np_.array(final)
        return final

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
                Column index or label that contains the data to use with newX to determine which rows to return. Defaults to None.
            replace (bool):
                If true, then the current AnalysisMixin's data is replaced with the  newly interpolated data and the current AnalysisMixin is
                returned.

        Returns:
            (2D numpy array):
                Section of the current object's data if replace is False(default) or the modofied AnalysisMixin if replace is true.

        Note:
            Returns complete rows of data corresponding to the indices given in newX. if xcol is None, then newX is interpreted as (fractional) row indices.
            Otherwise, the column specified in xcol is thresholded with the values given in newX and the resultant row indices used to return the data.

            If the positional argument, newX is None, then the return value is an interpolation function. This interpolation function takes one argument
            - if *xcol* was None, this argument is interpreted as array indices, but if *xcol* was specified, then this argument is interpreted as
            an array of xvalues.
        """
        DataArray = self.data.__class__
        l = _np_.shape(self.data)[0]
        index = _np_.arange(l)
        if xcol is None:
            xcol = self.setas._get_cols("xcol")
        elif isinstance(xcol, bool) and not xcol:
            xcol = None

        if isinstance(newX, ma.MaskedArray):
            newX = newX.compressed()

        if xcol is not None and newX is not None:  # We need to convert newX to row indices
            xfunc = interp1d(self.column(xcol), index, kind, 0)  # xfunc(x) returns partial index
            newX = xfunc(newX)
        inter = interp1d(index, self.data, kind, 0)

        if newX is None:  # Ok, we're going to return an interpolation function

            def wrapper(newX):
                """Wrapper for interpolation function."""
                if isinstance(newX, ma.MaskedArray):
                    newX = newX.compressed()
                else:
                    newX = _np_.array(newX)
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
        """Utility method to generate bin boundaries and centres along an axis.

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
        (xmin, xmax) = self.span(xcol)
        if mode not in ["lin", "log"]:
            raise ValueError("Mode should be either  'lin' or 'log' not {}".format(mode))

        if "bin_start" in kargs:
            xmin = kargs["bin_start"]
        if "bin_stop" in kargs:
            xmax = kargs["bin_stop"]
        if isinstance(bins, int):  # Given a number of bins
            if mode.lower().startswith("lin"):
                bin_width = float(xmax - xmin) / bins
                bin_start = _np_.linspace(xmin, xmax - bin_width, bins)
                bin_stop = _np_.linspace(xmin + bin_width, xmax, bins)
                bin_centres = (bin_start + bin_stop) / 2.0
            elif mode.lower().startswith("log"):
                xminl = _np_.log(xmin)
                xmaxl = _np_.log(xmax)
                bin_width = float(xmaxl - xminl) / bins
                bin_start = _np_.linspace(xminl, xmaxl - bin_width, bins)
                bin_stop = _np_.linspace(xminl + bin_width, xmaxl, bins)
                bin_centres = (bin_start + bin_stop) / 2.0
                bin_start = _np_.exp(bin_start)
                bin_stop = _np_.exp(bin_stop)
                bin_centres = _np_.exp(bin_centres)
            else:
                raise ValueError("mode should be either lin(ear) or log(arthimitc) not {}".format(mode))
        elif isinstance(bins, float):  # Given a bin with as a flot
            if mode.lower().startswith("lin"):
                bin_width = bins
                bins = _np_.ceil(abs(float(xmax - xmin) / bins))
                bin_start = _np_.linspace(xmin, xmax - bin_width, bins)
                bin_stop = _np_.linspace(xmin + bin_width, xmax, bins)
                bin_centres = (bin_start + bin_stop) / 2.0
            elif mode.lower().startswith("log"):
                if not 0.0 < bins <= 1.0:
                    raise ValueError("Bin width must be between 0 ans 1 for log binning")
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
                bin_start = _np_.array(splits[:-1])
                bin_stop = _np_.array(splits[1:])
                bin_centres = _np_.array(centers)
            else:
                raise ValueError("mode should be either lin(ear) or log(arthimitc) not {}".format(mode))
        elif isinstance(bins, _np_.ndarray) and bins.ndim == 1:  # Yser provided manuals bins
            bin_start = bins[:-1]
            bin_stop = bins[1:]
            if mode.lower().startwith("lin"):
                bin_centres = (bin_start + bin_stop) / 2.0
            elif mode.lower().startswith("log"):
                bin_centres = _np_.exp(_np_.log(bin_start) + _np_.log(bin_stop) / 2.0)
            else:
                raise ValueError("mode should be either lin(ear) or log(arthimitc) not {}".format(mode))
        else:
            raise TypeError("bins must be either an integer or a float, not a {}".format(type(bins)))
        if len(bin_start) > len(self):
            raise ValueError("Attempting to bin into more bins than there is data.")
        return bin_start, bin_stop, bin_centres

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
            sigma = _np_.array(sigma)
            _["has_yerr"] = True
        elif _.has_yerr:
            sigma = self.data[:, _.yerr]

        if not _.has_yerr:
            result = self.data[:, _.ycol].mean()
        else:
            ydata = self.data[:, _.ycol]
            w = 1 / (sigma ** 2 + 1e-8)
            norm = w.sum(axis=0)
            error = _np_.sqrt((sigma ** 2).sum(axis=0)) / len(sigma)
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

    def multiply(self, a, b, replace=False, header=None):
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
            err_calc = (
                lambda adata, bdata, e1data, e2data: _np_.sqrt((e1data / adata) ** 2 + (e2data / bdata) ** 2)
                * adata
                * bdata
            )
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
        if err_calc is not None:
            err_data = err_calc(adata, bdata, e1data, e2data)
        self.add_column((adata * bdata), header=header, index=a, replace=replace)
        if err_calc is not None:
            self.add_column(err_data, header=err_header, index=a + 1, replace=False)
        return self

    def normalise(self, target=None, base=None, replace=True, header=None, scale=None, limits=(0.0, 1.0)):
        """Normalise data columns by dividing through by a base column value.

        Args:
            target (index):
                One or more target columns to normalise can be a string, integer or list of strings or integers.
                If None then the default 'y' column is used.

        Keyword Arguments:
            base (index):
                The column to normalise to, can be an integer or string. **Depricated** can also be a tuple (low,high) being the output range
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

            The *limits* parameter is used to set the input scale being normalised from - if the data has a few outliers then
            this setting can be used to clip the input range before normalising. The parameters in the limit are the values at
            the *low* and *high* fractions of the cumulative distribution function of the data.
        """
        _ = self._col_args(scalar=True, ycol=target)

        target = _.ycol
        if not isinstance(target, list):
            target = [self.find_col(target)]
        for t in target:
            if header is None:
                header = self.column_headers[self.find_col(t)] + "(norm)"
            else:
                header = str(header)
            if not istuple(base, float, float) and base is not None:
                self.divide(t, base, header=header, replace=replace)
            else:
                if istuple(base, float, float):
                    scale = base
                elif scale is None:
                    scale = (-1.0, 1.0)
                if not istuple(scale, float, float):
                    raise ValueError("limit parameter is either None, or limit or base is a tuple of two floats.")
                data = self.column(t).ravel()
                data = _np_.sort(data[~_np_.isnan(data)])
                if limits != (0.0, 1.0):
                    low, high = limits
                    low = data[int(low * data.size)]
                    high = data[int(high * data.size)]
                else:
                    high = data.max()
                    low = data.min()
                data = _np_.copy(self.data[:, t])
                data = _np_.where(data > high, high, _np_.where(data < low, low, data))
                scl, sch = scale
                data = (data - low) / (high - low) * (sch - scl) + scl
                setas = self.setas.clone
                self.add_column(data, index=t, replace=replace, header=header)
                self.setas = setas
        return self

    def outlier_detection(self, column=None, window=7, certainty=3.0, action="mask", width=1, func=None, **kargs):
        """Function to detect outliers in a column of data.

        Args:
            column(column index):
                specifing column for outlier detection. If not set,
                defaults to the current y set column.

        Keyword Arguments:
            window(int):
                data window for anomoly detection
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

            def outlier(row,column,window,certainity,**kargs)
                #code
                return True # or False

        All extra keyword arguments are passed to the outlier detector.

        IF *action* is a callable function then it should take the form of::

            def action(i,column,row):
                pass

        where *i* is the number of the outlier row, *column* the same value as above
        and *row* is the data for the row.

        In all cases the indices of the outlier rows are added to the ;outlier' metadata.

        Example:
            .. plot:: samples/outlier.py
                :include-source:
                :outname: outlier
        """
        if func is None:
            func = _outlier

        _ = self._col_args(scalar=True, ycol=column, **kargs)
        params = get_func_params(func)
        for p in params:
            if p in _:
                kargs[p] = _[p]

        if action not in ["delete", "mask", "mask row"] and not callable(action):
            raise ValueError("Do'n know what to do with action={}".format(action))
        index = []
        for i, t in enumerate(self.rolling_window(window, wrap=False, exclude_centre=width)):
            if func(self.data[i], t, metric=certainty, **kargs):
                index.append(i)
        self["outliers"] = index  # add outlier indecies to metadata
        index.reverse()  # Always reverse the index in case we're deleting rows in sucession
        if action == "mask" or action == "mask row":
            for i in index:
                if action == "mask":
                    self.mask[i, column] = True
                else:
                    self.mask[i, :] = True
        elif action == "delete":
            for i in index:
                self.del_rows(i)
        elif callable(action):  # this will call the action function with each row in turn from back to start
            for i in index:
                action(i, column, self.data[i])
        return self

    def peaks(self, **kargs):
        """Locates peaks and/or troughs in a column of data by using SG-differentiation.

        Args:
            ycol (index):
                the column name or index of the data in which to search for peaks
            width (int or float):
                the expected minium halalf-width of a peak in terms of the number of data points (int) or distance in x (float).
                This is used in the differnetiation code to find local maxima. Bigger equals less sensitive
                to experimental noise, smaller means better eable to see sharp peaks
            poly (int):
                the order of polynomial to use when differentiating the data to locate a peak. Must >=2, higher numbers
                will find sharper peaks more accurately but at the risk of finding more false positives.

        Keyword Arguments:
            significance (float):
                used to decide whether a local maxmima is a significant peak. Essentially just the curvature
                of the data. Bigger means less sensistive, smaller means more likely to detect noise. Default is the maximum curvature/(2*width)
            xcol (index or None):
                name or index of data column that p[rovides the x-coordinate (default None)
            peaks (bool):
                select whether to measure peaks in data (default True)
            troughs (bool):
                select whether to measure troughs in data (default False)
            sort (bool):
                Sor the results by significance of peak
            modify (book):
                If true, then the returned object is a copy of self with only the peaks/troughs left in the data.
            full_data (bool):
                If True (default) then all columns of the data at which peaks in the *ycol* column are found. *modify* true implies
                *full_data* is also true. If *full_data* is False, then only the x-column values of the peaks are returned.

        Returns:
            (various):
                If *modify* is true, then returns a the AnalysisMixin with the data set to just the peaks/troughs. If *modify* is false (default),
                then the return value depends on *ycol* and *xcol*. If *ycol* is not None and *xcol* is None, then returns conplete rows of
                data corresponding to the found peaks/troughs. If *xcol* is not None, or *ycol* is None and *xcol* is None, then
                returns a 1D array of the x positions of the peaks/troughs.

        See Also:
            User guide section :ref:`peak_finding`
        """
        width = kargs.pop("width", int(len(self) / 20))
        peaks = kargs.pop("peaks", True)
        troughs = kargs.pop("troughs", False)
        poly = kargs.pop("poly", 2)
        assertion(
            poly >= 2, "poly must be at least 2nd order in peaks for checking for significance of peak or trough"
        )

        sort = kargs.pop("sort", False)
        modify = kargs.pop("modify", False)
        full_data = kargs.pop("full_data", True)
        _ = self._col_args(scalar=False, xcol=kargs.pop("xcol", None), ycol=kargs.pop("ycol", None))
        xcol, ycol = _.xcol, _.ycol
        if isiterable(ycol):
            ycol = ycol[0]
        if isinstance(width, float):  # Convert a floating point width unto an integer.
            xmin, xmax = self.span(xcol)
            width = int(len(self) * width / (xmax - xmin))
        width = max(width, poly + 1)
        setas = self.setas.clone
        self.setas = ""
        d1 = self.SG_Filter(ycol, xcol=xcol, points=width, poly=poly, order=1).ravel()
        d2 = self.SG_Filter(
            ycol, xcol=xcol, points=2 * width, poly=poly, order=2
        ).ravel()  # 2nd differential requires more smoothing

        # We're going to ignore the start and end of the arrays
        index_offset = int(width / 2)
        d1 = d1[index_offset:-index_offset]
        d2 = d2[index_offset:-index_offset]

        # Pad the ends of d2 with the mean value
        pad = _np_.mean(d2[index_offset:-index_offset])
        d2[:index_offset] = pad
        d2[-index_offset:] = pad

        # Set the significance from the 2nd ifferential if not already set
        significance = kargs.pop(
            "significance", _np_.max(_np_.abs(d2)) / (2 * width)
        )  # Base an apriori significance on max d2y/dx2 / 20
        if isinstance(significance, int):  # integer significance is inverse to floating
            significance = _np_.max(_np_.abs(d2)) / significance  # Base an apriori significance on max d2y/dx2 / 20

        d2_interp = interp1d(_np_.arange(len(d2)), d2, kind="cubic")
        # Ensure we have some X-data
        if xcol is None:
            xdata = _np_.arange(len(self))
        else:
            xdata = self.column(xcol)
        xdata = interp1d(_np_.arange(len(self)), xdata, kind="cubic")

        possible_peaks = _np_.array(_threshold(0, d1, rising=troughs, falling=peaks))
        curvature = _np_.abs(d2_interp(possible_peaks))

        # Filter just the significant peaks
        possible_peaks = _np_.array([p for ix, p in enumerate(possible_peaks) if abs(curvature[ix]) > significance])
        # Sort in order of significance
        if sort:
            possible_peaks = _np_.take(possible_peaks, _np_.argsort(_np_.abs(d2_interp(possible_peaks))))

        xdat = xdata(possible_peaks + index_offset)

        if modify:
            self.data = self.interpolate(xdat, xcol=xcol, kind="cubic")
            ret = self
        elif full_data:
            ret = self.interpolate(xdat, kind="cubic", xcol=False)
        else:
            ret = xdat
        self.setas = setas
        # Return - but remembering to add back on the offset that we took off due to differentials not working at start and end
        return ret

    def polyfit(
        self,
        xcol=None,
        ycol=None,
        polynomial_order=2,
        bounds=lambda x, y: True,
        result=None,
        replace=False,
        header=None,
    ):
        """ Pass through to numpy.polyfit.

            Args:
                xcol (index):
                    Index to the column in the data with the X data in it
                ycol (index):
                    Index to the column int he data with the Y data in it
                polynomial_order (int):
                    Order of polynomial to fit (default 2)
                bounds (callable):
                    A function that evaluates True if the current row should be included in the fit
                result (index or None):
                    Add the fitted data to the current data object in a new column (default don't add)
                replace (bool):
                    Overwrite or insert new data if result is not None (default False)
                header (string or None):
                    Name of column_header of replacement data. Default is construct a string from the y column headser and polynomial order.

            Returns:
                (numpy.poly):
                    The best fit polynomial as a numpy.poly object.

            Note:
                If the x or y columns are not specified (or are None) the the setas attribute is used instead.

                This method is depricated and may be removed in a future version in favour of the more general curve_fit
            """
        from Stoner.Util import ordinal

        _ = self._col_args(xcol=xcol, ycol=ycol, scalar=False)

        working = self.search(_.xcol, bounds)
        if not isiterable(_.ycol):
            _.ycol = [_.ycol]
        p = _np_.zeros((len(_.ycol), polynomial_order + 1))
        for i, ycol in enumerate(_.ycol):
            p[i, :] = _np_.polyfit(
                working[:, self.find_col(_.xcol)], working[:, self.find_col(ycol)], polynomial_order
            )
            if result is not None:
                if header is None:
                    header = "Fitted {} with {} order polynomial".format(
                        self.column_headers[self.find_col(ycol)], ordinal(polynomial_order)
                    )
                self.add_column(
                    _np_.polyval(p[i, :], x=self.column(_.xcol)), index=result, replace=replace, header=header
                )
        if len(_.ycol) == 1:
            p = p[0, :]
        self["{}-order polyfit coefficients".format(ordinal(polynomial_order))] = list(p)
        return p

    def scale(self, other, xcol=None, ycol=None, **kargs):
        """Scale the x and y data in this DataFile to match the x and y data in another DataFile.

        Args:
            other (DataFile):
                The other isntance of a datafile to match to

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
                Used to identiyf the set of (x,y) points to be used for scaling. Defaults to the whole data set if not speicifed.
            otherbounds (callable):
                Used to detemrine the set of (x,y) points in the other data file. Defaults to bounds if not given.
            use_estimate (bool or 3x2 array):
                Specifies whether to estimate an initial transformation value or to use the provided one, or
                start with an identity transformation.
            replace (bool):
                Whether to map the x,y data to the new co-ordinates and return a copy of this AnalysisMixin (true) or to just return
                the results of the scaling.
            headers (2-element list or tuple of strings):
                new column headers to use if replace is True.

        Returns:
            (various):
                Either a copy of the :py:class:Stoner.Data` modified so that the x and y columns match *other* if *replace* is True, or
                *opt_trans*,*trans_err*,*new_xy_data*. Where *opt_trans* is the optimum affine transformation, *trans_err* is a matrix
                giving the standard error in the transformation matrix components and  *new_xy_data* is an (n x 2) array of the transformed data.

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
                raise RuntimeError("Data lengths don't match {}!={}".format(len(xdat), len(xdat2)))
        elif isinstance(other, _np_.ndarray):
            if len(other.shape) == 1:
                other = _np_.atleast_2d(other).T
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
            raise RuntimeError(
                "other should be either a numpy array or subclass of DataFile, not a {}".format(type(other))
            )

        # Need two nx2 arrays of points now

        xy1 = _np_.column_stack((xdat, ydat))
        xy2 = _np_.column_stack((xdat2, ydat2))

        # We're going to use three points to get an estimate for the affine transform to apply

        if isinstance(use_estimate, bool) and use_estimate:
            mid = len(xdat) / 2
            try:  # may go wrong if three points are co-linear
                m0 = GetAffineTransform(xy1[[0, mid, -1], :], xy2[[0, mid, -1], :])
            except Exception:  # So use an idnetify transformation instead
                m0 = _np_.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        elif isinstance(use_estimate, _np_.ndarray) and use_estimate.shape == (
            2,
            3,
        ):  # use_estimate is an initial value transformation
            m0 = use_estimate
        else:  # Don't try to be clever
            m0 = _np_.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
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
                If int, then the number of points to use in the smoothing window. If float, then the size in x-data to be used.
            result (bool or column index):
                Whether to add the smoothed data to the dataset and if so where.
            replace (bool):
                Replace the exiting data or insert as a new column.
            header (string):
                New column header for the new data.

        Returns:
            (self or array):
                If result is False, then the return value will be a copy of the smoothed data, otherwise the return value
                is a copy of the AnalysisMixin object with the smoothed data added,

        Notes:
            If size is float, then it is necessary to map the X-data to a number of rows and to ensure that the data is evenly spaced in x.
            To do this, the number of rows in the window is found by dividing the span in x by the size and multiplying by the total
            lenfth. Then the data is interpolated to a new set of evenly space X over the same range, smoothed and then interpoalted back
            to the original x values.
        """
        _ = self._col_args(xcol=xcol, ycol=ycol)
        replace = kargs.pop("replace", True)
        result = kargs.pop("result", True)  # overwirte existing y column data
        header = kargs.pop("header", self.column_headers[_.ycol])

        # Sort out window size
        if isinstance(size, float):
            interp_data = True
            xl, xh = self.span(_.xcol)
            size = int(_np_.ceil((size / (xh - xl)) * len(self)))
            nx = _np_.linspace(xl, xh, len(self))
            data = self.interpolate(nx, kind="linear", xcol=_.xcol, replace=False)
            self["Smoothing window size"] = size
        elif isinstance(size, int_types):
            data = copy(self.data)
            interp_data = False
        else:
            raise ValueError("size should either be a float or integer, not a {}".format(type(size)))

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

    def span(self, column=None, bounds=None):
        """Returns a tuple of the maximum and minumum values within the given column and bounds by calling into :py:meth:`AnalysisMixin.max` and :py:meth:`AnalysisMixin.min`.

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
                new data is added after the specified index, if False then the new y-data is returned and if None, then
                spline object is returned.
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
                Depending on the value of *replace*, returns a copy of the AnalysisMixin, a 1D numpy array of data or an
                scipy.interpolate.UniverateSpline object.

        This is really jsut a pass through to the scipy.interpolate.UnivariateSpline function. Also used in the extrapolate
        function.
        """
        _ = self._col_args(xcol=xcol, ycol=ycol)
        if sigma is None and (isNone(_.yerr) or _.yerr):
            if not isNone(_.yerr):
                sigma = 1.0 / (self // _.yerr)
            else:
                sigma = _np_.ones(len(self))
        replace = kargs.pop("replace", True)
        result = kargs.pop("result", True)  # overwirte existing y column data
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
        elif result is None:
            return new_y
        else:
            return spline

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
            sigma = _np_.array(sigma)
            _["has_yerr"] = True
        elif _.has_yerr:
            sigma = self.data[:, _.yerr]

        if not _.has_yerr:
            result = self.data[:, _.ycol].mean()
        else:
            ydata = self.data[:, _.ycol]
            w = 1 / (sigma ** 2 + 1e-8)
            norm = w.sum(axis=0)
            error = _np_.sqrt((sigma ** 2).sum(axis=0)) / len(sigma)
            result = (ydata * w).std(axis=0) / norm, error
        if bounds is not None:
            self._pop_mask()
        return result

    def stitch(self, other, xcol=None, ycol=None, overlap=None, min_overlap=0.0, mode="All", func=None, p0=None):
        r"""Apply a scaling to this data set to make it stich to another dataset.

        Args:
            other (DataFile):
                Another data set that is used as the base to stitch this one on to
            xcol,ycol (index or None):
                The x and y data columns. If left as None then the current setas attribute is used.

        Keyword Arguments:
            overlap (tuple of (lower,higher) or None):
                The band of x values that are used in both data sets to match,
                if left as None, thenthe common overlap of the x data is used.
            min_overlap (float):
                If you know that overlap must be bigger than a certain amount, the bounds between the two
                data sets needs to be adjusted. In this case min_overlap shifts the boundary of the overlap on this DataFile.
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
        :math:`x'=x+A` and :math:`y'=By+C` where :math:`A,B,C` are constants and :math:`(x',y')` are close matches to the
        :math:`(x,y)` data in *other*. The algorithm assumes that the overlap region contains equal
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
        """
        _ = self._col_args(xcol=xcol, ycol=ycol, scalar=True)
        points = self.column([_.xcol, _.ycol])
        points = points[points[:, 0].argsort(), :]
        points[:, 0] += min_overlap
        otherpoints = other.column([_.xcol, _.ycol])
        otherpoints = otherpoints[otherpoints[:, 0].argsort(), :]
        self_second = _np_.max(points[:, 0]) > _np_.max(otherpoints[:, 0])
        if overlap is None:  # Calculate the overlap
            lower = max(_np_.min(points[:, 0]), _np_.min(otherpoints[:, 0]))
            upper = min(_np_.max(points[:, 0]), _np_.max(otherpoints[:, 0]))
        elif isinstance(overlap, int) and overlap > 0:
            if self_second:
                lower = points[0, 0]
                upper = points[0, overlap]
            else:
                lower = points[0, -overlap - 1]
                upper = points[0, -1]
        elif (
            isinstance(overlap, tuple)
            and len(overlap) == 2
            and isinstance(overlap[0], float and isinstance(overlap[1], float))
        ):
            lower = min(overlap)
            upper = max(overlap)
        inrange = _np_.logical_and(points[:, 0] >= lower, points[:, 0] <= upper)
        points = points[inrange]
        num_pts = points.shape[0]
        if self_second:
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
            A0 = _np_.mean(xp) - _np_.mean(x)
            C0 = _np_.mean(yp) - _np_.mean(y)
            B0 = (_np_.max(yp) - _np_.min(yp)) / (_np_.max(y) - _np_.min(y))
            p = _np_.array([0, A0, B0, C0])
            assertion(isinstance(mode, string_types), "mode keyword should be a string if func is not defined")
            mode = mode.lower()
            assertion(mode in opts, "mode keyword should be one of {}".format(opts.keys))
            func = opts[mode]
            p0 = p[defaults[mode]]
        else:
            assertion(callable(func), "Keyword func should be callable if given")
            args = getfullargspec(func)[0]  # pylint: disable=W1505
            assertion(isiterable(p0), "Keyword parameter p0 shoiuld be iterable if keyword func is given")
            assertion(
                len(p0) == len(args) - 2, "Keyword p0 should be the same length as the optional arguments to func"
            )
        # This is a bit of a hack, we turn (x,y) points into a 1D array of x and then y data
        set1 = _np_.append(x, y)
        set2 = _np_.append(xp, yp)
        assertion(len(set1) == len(set2), "The number of points in the overlap are different in the two data sets")

        def transform(set1, *p):
            """Wrapper function to fit for transform."""
            m = int(len(set1) / 2)
            x = set1[:m]
            y = set1[m:]
            tmp = func(x, y, *p)
            out = _np_.append(tmp[0], tmp[1])
            return out

        popt, pcov = curve_fit(transform, set1, set2, p0=p0)  # Curve fit for optimal A,B,C
        perr = _np_.sqrt(_np_.diagonal(pcov))
        self.data[:, _.xcol], self.data[:, _.ycol] = func(self.data[:, _.xcol], self.data[:, _.ycol], *popt)
        self["Stitching Coefficients"] = list(popt)
        self["Stitching Coeffient Errors"] = list(perr)
        self["Stitching overlap"] = (lower, upper)
        self["Stitching Window"] = num_pts

        return self

    def subtract(self, a, b, replace=False, header=None):
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
            err_calc = lambda adata, bdata, e1data, e2data: _np_.sqrt(e1data ** 2 + e2data ** 2)
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
        if err_calc is not None:
            err_data = err_calc(adata, bdata, e1data, e2data)
        self.add_column((adata - bdata), header=header, index=a, replace=replace)
        if err_calc is not None:
            a = self.find_col(a)
            self.add_column(err_data, header=err_header, index=a + 1, replace=False)
        return self

    def threshold(self, threshold, **kargs):
        """Finds partial indices where the data in column passes the threshold, rising or falling.

        Args:
            threshold (float):
                Value to look for in column col

        Keyword Arguments:
            col (index):
                Column index to look for data in
            rising (bool):
                look for case where the data is increasing in value (defaukt True)
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
            If you don't sepcify a col value or set it to None, then the assigned columns via the
            :py:attr:`DataFile.setas` attribute will be used.

        Warning:
            There has been an API change. Versions prior to 0.1.9 placed the column before the threshold in the positional
            argument list. In order to support the use of assigned columns, this has been swapped to the present order.
        """
        DataArray = self.data.__class__
        col = kargs.pop("col", None)
        if col is None:
            col = self.setas._get_cols("ycol")
            xcol = kargs.pop("xcol", self.setas._get_cols("xcol"))
        else:
            xcol = kargs.pop("xcol", None)

        rising = kargs.pop("rising", True)
        falling = kargs.pop("falling", False)
        all_vals = kargs.pop("all_vals", False)

        current = self.column(col)

        # Recursively call if we've got an iterable threshold
        if isiterable(threshold):
            if isinstance(xcol, bool) and not xcol:
                ret = _np_.zeros((len(threshold), self.shape[1]))
            else:
                ret = _np_.zeros_like(threshold).view(type=DataArray)
            for ix, th in enumerate(threshold):
                ret[ix] = self.threshold(th, col=col, xcol=xcol, rising=rising, falling=falling, all_vals=all_vals)
            # Now we have to clean up the  retujrn list into a DataArray
            if isinstance(xcol, bool) and not xcol:  # if xcol was False we got a complete row back
                ch = self.column_headers
                ret.setas = self.setas.clone
                ret.column_headers = ch
                ret.i = ret[0].i
            else:  # Either xcol was None so we got indices or we got a specified column back
                if xcol is not None:  # Specific column
                    ret = _np_.atleast_2d(ret)
                    ret.column_headers = [self.column_headers[self.find_col(xcol)]]
                    ret.i = [r.i for r in ret]
                    ret.setas = "x"
                    ret.isrow = False
                else:
                    ret.column_headers = ["Index"]
                    ret.isrow = False
            return ret
        else:
            ret = _threshold(threshold, current, rising=rising, falling=falling)
            if not all_vals:
                ret = [ret[0]] if _np_.any(ret) else []

        if isinstance(xcol, bool) and not xcol:
            retval = self.interpolate(ret, xcol=False)
            retval.setas = self.setas.clone
            retval.setas.shape = retval.shape
            retval.i = ret
            ret = retval
        elif xcol is not None:
            retval = self.interpolate(ret, xcol=False)[:, self.find_col(xcol)]
            # if retval.ndim>0:   #not sure what this bit does but it's throwing errors for a simple threshold
            # retval.setas=self.setas.clone
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


def AnalyseFile(*args, **kargs):
    """Issue a warning and then create a class anyway."""
    warn("AnalyseFile is deprecated in favour of Stoner.Data or the AnalysisMixin", DeprecationWarning)
    import Stoner.Core as _SC_

    class AnalyseFile(AnalysisMixin, _SC_.DataFile):

        """:py:class:`Stoner.Analysis.AnalyseFile` extends :py:class:`Stoner.Core.DataFile` with numpy and scipy passthrough functions.

        Note:
            There is no separate constructor for this class - it inherits from DataFile

        """

        pass

    return AnalyseFile(*args, **kargs)
