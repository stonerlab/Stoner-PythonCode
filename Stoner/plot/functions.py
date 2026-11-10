"""Retain independently callable numerical gridding for Data."""

import numpy as np
from scipy.interpolate import griddata as sp_griddata

from ..compat import int_types
from ..tools.tests import isanynone

__all__ = ["griddata"]
from ..core.column_selection import _single_column


def _span_slice(datafile, col, num):
    """Preserve numerical gridding's existing inclusive span and spacing."""
    v1, v2 = datafile.span(col)
    values = np.linspace(v1, v2, num)
    delta = values[1] - values[0]
    v2 = v2 + delta if isinstance(delta, int_types) else v2 + delta / 2
    return slice(v1, v2, delta)


def griddata(
    datafile,
    xcol=None,
    ycol=None,
    zcol=None,
    ucol=None,
    shape=None,
    xlim=None,
    ylim=None,
    zlim=None,
    method="linear",
    **kwargs,
):
    """Interpolate scalar samples onto the existing numerical coordinate grid.

    Args:
        datafile (Data):
            Measurement table when called without a bound instance.
        xcol, ycol, zcol (column selector or None):
            Coordinates and scalar response, defaulting to assigned roles.

    Keyword Arguments:
        ucol (column selector, ndarray or None):
            Volume response; an assigned U role selects volumetric interpolation.
        shape (tuple or None):
            Retained numerical argument; existing grid counts are derived from
            distinct coordinate values rather than this argument.
        xlim, ylim, zlim (tuple or None):
            Coordinate bounds, otherwise derived from the measurement span.
        method (str):
            SciPy interpolation method, defaulting to linear.
        **kwargs (dict):
            startx selects the assigned coordinate group.

    Returns:
        tuple of ndarray:
            X/Y/scalar grids, or X/Y/Z corner grids and an intensity volume.

    Notes:
        This retains the numerical API's interpolation, grid and extent rules.
        It creates no figure and changes no measurements. Rendering methods use
        their separately documented mask-aware preparation in Data.plot.
    """
    del shape  # The numerical API derives grid counts from distinct coordinates.
    dims, selectors = _grid_selectors(datafile, (xcol, ycol, zcol, ucol), kwargs.pop("startx", 0))
    counts = [
        np.unique(datafile.column(column)).size
        for column in selectors[:2 if getattr(selectors[2], "size", 0) else 3]
    ]
    limits, extents = _grid_slices(datafile, selectors, counts, (xlim, ylim, zlim), dims)
    if dims == 2:
        points = np.mgrid[limits[0], limits[1]].T  # pylint: disable=no-member
        samples = np.array([datafile.column(column) for column in selectors[:2]]).T
        values = _grid_response(datafile, selectors[2])
        return points[:, :, 0], points[:, :, 1], _interpolate_response(samples, values, points, method)
    points = np.mgrid[limits[0], limits[1], limits[2]].T  # pylint: disable=no-member
    corners = np.mgrid[extents[0], extents[1], extents[2]].T  # pylint: disable=no-member
    samples = np.array([datafile.column(column) for column in selectors[:3]]).T
    values = _grid_response(datafile, selectors[3])
    return corners[:, :, :, 0], corners[:, :, :, 1], corners[:, :, :, 2], _interpolate_response(
        samples, values, points, method
    )


def _grid_selectors(datafile, selectors, startx):
    """Resolve numerical grid selectors while retaining the historical role defaults."""
    xcol, ycol, zcol, ucol = selectors
    columns = datafile.setas._get_cols(startx=startx)
    if isanynone(xcol, ycol, zcol):
        xcol = columns["xcol"] if xcol is None else xcol
        ycol = columns["ycol"][0] if ycol is None else ycol
        if zcol is None and len(columns["zcol"]) > 0:
            zcol = columns["zcol"][0]
    if ucol is None and len(columns["ucol"]) > 0:
        ucol = columns["ucol"][0]
    dims = 3 if columns["axes"] == 3 and columns["has_ucol"] else 2
    xcol = _single_column(datafile, xcol, "xcol")
    ycol = _single_column(datafile, ycol, "ycol")
    if dims == 3:
        zcol = _single_column(datafile, zcol, "zcol")
    return dims, (xcol, ycol, zcol, ucol)


def _grid_slices(datafile, selectors, shape, limits, dims):
    """Prepare inclusive numerical grid centres and half-cell boundary extents."""
    slices, extents = list(limits), list(limits)
    for dimension, (limit, column) in enumerate(zip(limits[:dims], selectors)):
        if limit is None:
            limit = _span_slice(datafile, column, shape[dimension])
        elif isinstance(limit, tuple) and len(limit) > 1:
            limit = limit + ((limit[1] - limit[0]) / shape[dimension],)
            limit = slice(limit[0], limit[1], limit[2])
        else:
            raise RuntimeError(f"{'xyz'[dimension]} limit speciifcation not understood")
        slices[dimension] = limit
        extents[dimension] = slice(limit.start - limit.step / 2, limit.stop + limit.step / 2, limit.step)
    return slices, extents


def _grid_response(datafile, selector):
    """Obtain a numerical response vector or complete externally supplied channels."""
    if selector is None:
        return np.zeros(len(datafile))
    if isinstance(selector, np.ndarray) and selector.shape[0] == len(datafile):
        return selector
    return datafile.column(selector)


def _interpolate_response(samples, values, points, method):
    """Interpolate each response channel independently on the prepared sample grid."""
    if values.ndim == 1:
        return sp_griddata(samples, values, points, method=method)
    if values.ndim == 2:
        result = np.zeros((points.shape[0], points.shape[1], values.shape[1]))
        for index in range(values.shape[1]):
            result[:, :, index] = sp_griddata(samples, values[:, index], points, method=method)
        return result
    raise ValueError("Grid responses must be a vector or a matrix of response channels.")
