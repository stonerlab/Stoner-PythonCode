# -*- coding: utf-8 -*-
"""Functions used by the AnalysisMixin class."""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, newton
from scipy.signal import get_window

__all__ = ["outlier", "threshold", "_twoD_fit", "ApplyAffineTransform", "GetAffineTransform", "poly_outlier"]


def outlier(row, window, metric, ycol=None, shape="bopxcar"):
    """Outlier detector function.

    Calculates if the current row is an outlier from the surrounding data by looking
    at the number of standard deviations away from the average of the window it is.

    Args:
        row (array):
            Single row of the dataset.
        window (array):
            Section of data surrounding the row being examined.
        metric (float):
            distance the current row is from the local mean.

    Keyword Arguments:
        ycol (column index or None):
            If set, specifies the column containing the data to check.

    Returns:
        (bool):
            If True, then the current row is an outlier from the local data.
    """
    windowing = get_window(shape, len(window))
    windowing /= windowing.sum() / windowing.size
    av = np.average(window[:, ycol] * windowing)
    std = np.std(window[:, ycol])  # standard deviation
    return abs(row[ycol] - av) > metric * std


def poly_outlier(row, window, metric=3.0, ycol=None, xcol=None, order=1, yerr=None):
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
        raise ValueError(f"order should be smaller than the window length. {order} vs {window.shape[0] - 2}")

    x = window[:, xcol] - row[xcol]
    y = window[:, ycol]
    if yerr:
        w = 1.0 / window[:, yerr]
    else:
        w = None

    popt, pcov = np.polyfit(x, y, w=w, deg=order, cov=True)
    pval = np.polyval(popt, 0.0)
    perr = np.sqrt(np.diag(pcov))[-1]
    return (pval - row[ycol]) ** 2 > metric * perr


def threshold(threshold, data, rising=True, falling=False):
    """Implement the threshold method - also used in peak-finder.

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
    previous = np.roll(current, 1)
    current = np.atleast_1d(current)
    index = np.arange(len(current))
    sdat = np.column_stack((index, current, previous))
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
    return np.array(roots)


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

    The most general case is an affine transform which includes rotation, scale, translation and skew. This is
    represented as a 2 x 3 matrix  of coordinates. The *xmode* and *ymode* parameters control the possible operations
    to align the data in x and y directions, in addition to which the *xmode* parameter can take the value 'affine'
    which allows a full affine transformation. The returned values are the affine transformation matrix, the
    uncertainties in this and a function to map coordinates with the optimal affine transformation.

    Note:
        *m0* combines both giving an initial value and fixed values for the transformation. If *m0* is set, then it
        is used to provide initial balues of the free parameters. Which elelemnts of *m0* that are free parameters
        and which are fixed is determined by the *xmode* and *ymode* parameters. IF *xmode* and *ymode* are both
        fixed, however, no scaling is done at all.
    """
    if xy1.shape != xy2.shape or xy1.shape[1] != 2:
        raise RuntimeError(f"coordinate arrays must be equal length with two columns, not {xy1.shape} and {xy2.shape}")
    xvarp = {
        "affine": [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]],
        "linear": [[0, 0], [0, 2]],
        "scale": [[0, 0]],
        "offset": [[0, 2]],
        "fixed": [[]],
    }
    yvarp = {"linear": [[1, 1], [1, 2]], "scale": [[1, 1]], "offset": [[1, 2]], "fixed": [[]]}

    if xmode not in xvarp or ymode not in yvarp:
        raise RuntimeError(f"xmode and ymode must be one of 'linear','scale','offset','fixed' not {xmode} and {ymode}")

    if xmode == "affine":
        ymode = "fixed"

    xunknowns = len(xvarp[xmode])
    yunknowns = len(yvarp[ymode])
    if xunknowns + yunknowns == 0:  # shortcircuit for the trivial case
        return np.array([[1, 0], [1, 0]]), np.zeros((2, 2)), lambda x: x

    mapping = xvarp[xmode] + yvarp[ymode]
    mapping = [m for m in mapping if m != []]  # remove empty mappings
    data = np.column_stack((xy1, xy2)).T

    if isinstance(m0, list):
        m0 = np.array(m0)

    if m0 is None:
        p0s = {"affine": [1, 0, 0, 0, 1, 0], "linear": [1, 0], "scale": [1], "offset": [0], "fixed": []}
        p0 = p0s[xmode] + p0s[ymode]
        default = np.array([[1.00, 0.0, 0.0], [0.0, 1.0, 0.0]])
    elif isinstance(m0, np.ndarray) and m0.shape == (2, 3):
        p0 = [0] * len(mapping)
        for i, [u, v] in enumerate(mapping):
            p0[i] = m0[u, v]
            default = m0
    else:
        raise RuntimeError(f"m0 starting matrix should be a numpy array of size (2,3) not {m0}")

    result = np.zeros(len(xy1))

    def transform(xy, *p):  # Construct the fitting function
        """Fitting function to find the transfoprm."""
        xy1 = np.column_stack((xy[:2, :].T, np.ones(xy.shape[1]))).T
        xy2 = xy[2:, :]
        for pi, (u, v) in zip(p, mapping):
            default[u, v] = pi
        xyt = np.dot(default, xy1)
        ret = np.sqrt(np.sum((xy2 - xyt) ** 2, axis=0))
        return ret

    popt, pcov = curve_fit(transform, data, result, p0=p0)
    perr = np.sqrt(np.diag(pcov))

    # Initialise the return values
    default = np.array([[1.00, 0.0, 0.0], [0.0, 1.0, 0.0]])
    for pi, (u, v) in zip(popt, mapping):
        default[u, v] = pi
    default_err = np.zeros((2, 3))
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
        Transformed coordinates.
    """
    xyt = np.vstack((xy.T, np.ones(len(xy))))
    xyt = np.dot(transform, xyt)
    return xyt.T


def GetAffineTransform(p, pd):
    """Calculate an affine transform from 2 sets of three points.

    Args:
        p (3x2 array):
            Coordinates of points to transform from.
        pd (3x2 array):
            Coordinates of points to transform to.

    Returns:
        (2x3 array):
            matrix representing the affine transform.
    """
    if np.shape(p) != (3, 2) and np.shape(pd) != (3, 2):
        raise RuntimeError("Must supply three points")

    p = np.append(p, np.atleast_2d(np.ones(3)).T, axis=1)
    transform = np.linalg.solve(p, pd)
    return transform.T
