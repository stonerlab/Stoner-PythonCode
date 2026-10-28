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
    startx = kwargs.pop("startx", 0)
    cols = datafile.setas._get_cols(startx=startx)
    if isanynone(xcol, ycol, zcol):
        if xcol is None:
            xcol = cols["xcol"]
        if ycol is None:
            ycol = cols["ycol"][0]
        if zcol is None and len(cols["zcol"]) > 0:
            zcol = cols["zcol"][0]
    if ucol is None and len(cols["ucol"]) > 0:
        ucol = cols["ucol"][0]

    dims = 3 if cols["axes"] == 3 and cols["has_ucol"] else 2
    xcol = _single_column(datafile, xcol, "xcol")
    ycol = _single_column(datafile, ycol, "ycol")
    if dims == 3:
        zcol = _single_column(datafile, zcol, "zcol")
    if getattr(zcol, "size", 0) != 0:
        shape = [np.unique(datafile.column(x)).size for x in [xcol, ycol]]
    else:
        shape = [np.unique(datafile.column(x)).size for x in [xcol, ycol, zcol]]

    lims = [xlim, ylim, zlim]
    extents = [xlim, ylim, zlim]
    for dim, (lim, col) in enumerate(zip(lims, [xcol, ycol, zcol])):
        if dim >= dims:  # Don;t bother analysing more dimensions than we have
            break
        if lim is None:
            lim = _span_slice(datafile, col, shape[dim])
        elif isinstance(lim, tuple) and len(lim) > 1:
            lim = lim + ((lim[1] - lim[0]) / shape[dim],)
            lim = slice(lim[0], lim[1], lim[2])
        else:
            raise RuntimeError(f"{'xyz'[dim]} limit speciifcation not understood")
        lims[dim] = lim
        extents[dim] = slice(lim.start - lim.step / 2, lim.stop + lim.step / 2, lim.step)

    xlim = lims[0]
    ylim = lims[1]
    zlim = lims[2]

    if dims == 2:
        pts = np.mgrid[xlim, ylim].T  # pylint: disable=no-member
        points = np.array([datafile.column(xcol), datafile.column(ycol)]).T
        if zcol is None:
            zdata = np.zeros(len(datafile))
        elif isinstance(zcol, np.ndarray) and zcol.shape[0] == len(datafile):  # zcol is some data
            zdata = zcol
        else:
            zdata = datafile.column(zcol)
        if zdata.ndim == 1:
            Z = sp_griddata(points, zdata, pts, method=method)
        elif zdata.ndim == 2:
            Z = np.zeros((pts.shape[0], pts.shape[1], zdata.shape[1]))
            for i in range(zdata.shape[1]):
                Z[:, :, i] = sp_griddata(points, zdata[:, i], pts, method=method)

        return pts[:, :, 0], pts[:, :, 1], Z
    if dims == 3:
        pts = np.mgrid[xlim, ylim, zlim].T  # pylint: disable=no-member
        vpts = np.mgrid[extents[0], extents[1], extents[2]].T  # pylint: disable=no-member

        points = np.array([datafile.column(xcol), datafile.column(ycol), datafile.column(zcol)]).T
        if ucol is None:
            udata = np.zeros(len(datafile))
        elif isinstance(ucol, np.ndarray) and ucol.shape[0] == len(datafile):  # zcol is some data
            udata = ucol
        else:
            udata = datafile.column(ucol)
        if udata.ndim == 1:
            U = sp_griddata(points, udata, pts, method=method)
        elif udata.ndim == 2:
            U = np.zeros((pts.shape[0], pts.shape[1], udata.shape[1]))
            for i in range(udata.shape[1]):
                U[:, :, i] = sp_griddata(points, udata[:, i], pts, method=method)

        return vpts[:, :, :, 0], vpts[:, :, :, 1], vpts[:, :, :, 2], U
    return None
