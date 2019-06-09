# -*- coding: utf-8 -*-
"""Stoner.Utils - a module of some slightly experimental routines that use the Stoner classes

Created on Tue Oct 08 20:14:34 2013

@author: phygbu
"""
__all__ = ["split_up_down", "ordinal", "hysteresis_correct"]
from .compat import int_types
from .tools import format_error
import Stoner.Core as _SC_  # pylint: disable=import-error
from .Folders import DataFolder as _SF_
from .Fit import linear
from . import Data
import numpy as np
from numpy import max, sqrt, diag, argmax, mean, array  # pylint: disable=redefined-builtin
from scipy.stats import sem


def _step(x, m, c, h):
    """
    Simple sloping step function for fitting to the extrema of a hysteresis loop.

    Args:
        x (array-like): Field (x) data of loop.
        m (float): susceptibility(slopee) of loop.
        c (float): vertical offset of loop.
        h (float): Saturatted moement (height) of loop.

    Returns:
        y (Tarray-like): Calculated moment of loop.
    """

    mid = (x.max() + x.min()) / 2.0

    X = x - mid
    y = m * x + c + np.sign(X) * h
    return y


def _up_down(data):
    """Split data d into rising and falling sections and then add and sort the two sets.

    This routine searches for the local maxima and minima in the x-axis data by identifying points
    where the x-data is more than 95% of the total span from the mid-point. It thenuses this to identify
    ranges of rows where the x value passes from one extreme to the other and splits the data file up on
    that basis. This assumes that the x-data is reasonably well behaved in the the increments between
    successive x-values are larger than the noise in x. This will work best if the x value is derived from a set value
    rather than the read-back value.

    Args:
        data (Data): DataFile like object with x and y columns set

    Returns:
        (Data, Data): Tuple of two DataFile like instances for the rising and falling data.
    """
    # Calculate x span and mid-point for working out limits to search for maxima.
    lx, hx = data.span(data.setas.xcol)
    mid = (lx + hx) / 2.0
    span = hx - lx
    high = data.x > mid + 0.45 * span
    low = data.x < mid - 0.45 * span

    # Locate points where we cross a threhold
    t = np.zeros((2, len(data) + 2), dtype=bool)
    t[0, 1:-1] = high
    t[1, 1:-1] = low
    t = np.diff(t)

    # Find  the indiices of the highest and lowest extrema
    high_i = np.arange(len(data) + 1, dtype=int)[t[0, :]]
    low_i = np.arange(len(data) + 1, dtype=int)[t[1, :]]
    if low_i.size > 2:
        low_i = np.reshape(low_i, (2, -1)).mean(axis=1)
    else:
        low_i = np.array(low_i.mean())
    if high_i.size > 2:
        high_i = np.reshape(high_i, (2, -1)).mean(axis=1)
    else:
        high_i = np.array(high_i.mean())
    # Build a sorted list of extrema positions + the start and end of the data
    indices = np.unique(np.append(low_i, np.append(high_i, np.array([0, len(data) - 1]))))
    indices = np.ceil(indices).astype(int)
    indices.sort()
    if indices[-1] >= len(data):  # Remove possible over-range element
        indices = indices[0:-1]

    # Build a boolean array to index the data for rows that where x is increasing
    rising = np.zeros(len(data), dtype=bool)
    for ix, iy in zip(indices[:-1], indices[1:]):
        if data.x[iy] > data.x[ix]:
            rising[ix:iy] = True

    # Clone the data and select rows that are either rising or falling in the two clones.
    up = data.clone
    up.data = up.data[rising]
    down = data.clone
    down.data = down.data[~rising]
    # Done.
    return up, down


def split_up_down(data, col=None, folder=None):
    """Splits the DataFile data into several files where the column *col* is either rising or falling

    Args:
        data (:py:class:`Stoner.Core.DataFile`): 
            object containign the data to be sorted
        col (index): 
            is something that :py:meth:`Stoner.Core.DataFile.find_col` can use
        folder (:py:class:`Stoner.Folders.DataFolder` or None): 
            if this is an instance of :py:class:`Stoner.Folders.DataFolder` then add
            rising and falling files to groups of this DataFolder, otherwise create a new one

    Returns:
        (:py:class:`Sonter.Folder.DataFolder`): with two groups, rising and falling
    """
    a = Data(data)
    if col is None:
        _ = a._col_args()
        col = _.xcol
    width = int(len(a) / 10)
    if width % 2 == 0:  # Ensure the window for Satvisky Golay filter is odd
        width += 1
    setas = a.setas.clone
    a.setas = ""
    peaks = list(a.peaks(ycol=col, width=width, full_data=False))
    troughs = list(a.peaks(ycol=col, width=width, peaks=False, troughs=True, full_data=False))
    a.setas = setas
    if peaks and troughs:  # Ok more than up down here
        order = peaks[0] < troughs[0]
    elif peaks:  # Rise then fall
        order = True
    elif troughs:  # Fall then rise
        order = False
    else:  # No peaks or troughs so just return a single rising
        ret = _SF_(readlist=False)
        ret += data
        return ret
    splits = [0, len(a)]
    splits.extend(peaks)
    splits.extend(troughs)
    splits.sort()
    splits = [int(s) for s in splits]
    if not isinstance(folder, _SF_):  # Create a new DataFolder object
        output = _SF_(readlist=False)
    else:
        output = folder
    output.add_group("rising")
    output.add_group("falling")

    if order:
        risefall = ["rising", "falling"]
    else:
        risefall = ["falling", "rising"]
    for i in range(len(splits) - 1):
        working = data.clone
        working.data = data.data[splits[i] : splits[i + 1], :]
        output.groups[risefall[i % 2]].append(working)
    return output


Hickeyify = format_error


def ordinal(value):
    """Format an integer into an ordinal string.

    Args:
        value (int):
            Number to be written as an ordinal string

    Return:
        (str): Ordinal String such as '1st','2nd' etc.
    """
    if not isinstance(value, int):
        raise ValueError

    last_digit = value % 10
    if value % 100 in [11, 12, 13]:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th", "th", "th", "th", "th", "th"][last_digit]

    return "{}{}".format(value, suffix)


def hysteresis_correct(data, **kargs):
    """Peform corrections to a hysteresis loop.

    Args:
        data (Data):
            The data containing the hysteresis loop. The :py:attr:`DataFile.setas` attribute
            should be set to give the H and M axes as x and y.

    Keyword Arguments:
        correct_background (bool): 
            Correct for a diamagnetic or paramagnetic background to the hystersis loop
            also recentres the loop about zero moment (default True).
        correct_H (bool): 
            Finds the co-ercive fields and sets them to be equal and opposite. If the loop is sysmmetric
            this will remove any offset in filed due to trapped flux (default True)
        saturated_fraction (float):
            The fraction of the horizontal (field) range where the moment can be assumed to be
            fully saturated. If an integer is given it will use that many data points at the end of the loop.
        xcol (column index):
            Column with the x data in it
        ycol (column_index): 
            Column with the y data in it
        setas (string or iterable): 
            Column assignments.

    Returns:
        (:py:class:`Stoner.Data`):
            The original loop with the x and y columns replaced with corrected data and extra metadata added to give the
            background suceptibility, offset in moment, co-ercive fields and saturation magnetisation.
    """
    if isinstance(data, _SC_.DataFile):
        cls = data.__class__
    else:
        cls = Data
    data = cls(data)

    if "setas" in kargs:  # Allow us to override the setas variable
        data.setas = kargs.pop("setas")

    xcol = kargs.pop("xcol", data.setas.xcol)
    ycol = kargs.pop("ycol", data.setas.ycol)
    # Get xcol and ycols from kargs if specified
    _ = data._col_args(xcol=xcol, ycol=ycol)
    data.setas(x=_.xcol, y=_.ycol)
    # Split into two sets of data:

    # Get other keyword arguments
    correct_background = kargs.pop("correct_background", True)
    correct_H = kargs.pop("correct_H", True)
    saturation_fraction = kargs.pop("saturated_fraction", 0.2)

    up, down = _up_down(data)

    mid = (data.x.max() + data.x.min()) / 2.0
    span = data.x.max() - data.x.min()
    low = mid - span * (1 - saturation_fraction) / 2
    high = mid + span * (1 - saturation_fraction) / 2

    popt, pcov = data.curve_fit(_step, bounds=lambda x, r: not low < x < high)

    perr = np.sqrt(np.diag(pcov))

    Ms = popt[2]
    Ms_err = perr[2]

    data["Ms"] = Ms  # mean(Ms)
    data["Ms Error"] = Ms_err
    data["Offset Moment"] = popt[1]
    data["Offset Moment Error"] = perr[1]
    data["Background susceptibility"] = popt[0]
    data["Background Susceptibility Error"] = perr[0]

    if correct_background:
        fixes = [data, up, down]
    else:
        fixes = [up, down]
    m = popt[0]
    c = popt[1] - m * mid

    for d in fixes:
        d.y = d.y - linear(d.x, c, m)

    Hc = [None, None]
    Hc_err = [None, None]
    Hsat = [None, None]
    Hsat_err = [None, None]
    Mr = [None, None]
    Mr_err = [None, None]

    m_sat = [Ms - 2 * Ms_err, -Ms + 2 * Ms_err]
    single_side = False

    for i, (d, sat) in enumerate(zip([up, down], m_sat)):
        if len(d) < 2:
            single_side = True
            continue
        hc = d.threshold(0.0, all_vals=True, rising=i == 0, falling=i != 0)  # Get the Hc value
        Hc[i] = mean(hc)
        if hc.size > 1:
            Hc_err[i] = sem(hc)
        hs = d.threshold(sat, all_vals=True, rising=i == 0, falling=i != 0)  # Get the H_sat value
        Hsat[1 - i] = mean(hs)  # Get the H_sat value
        if hs.size > 1:
            Hsat_err[1 - i] = sem(hs)
        mr = d.threshold(0.0, col=_.xcol, xcol=_.ycol, all_vals=True, rising=True, falling=True)
        Mr[i] = mean(mr)
        if mr.size > 1:
            Mr_err[i] = sem(mr)

    if correct_H and not single_side:
        Hc_mean = mean(Hc)
        for d in [data, up, down]:
            d.x = d.x - Hc_mean
        data["Exchange Bias offset"] = Hc_mean
    else:
        Hc_mean = 0.0

    if not single_side:
        data["Hc"] = (Hc[1] - Hc_mean, Hc[0] - Hc_mean)
        data["Hc_mean"] = np.abs(np.array(Hc)).mean()
        data["Hsat"] = (Hsat[1] - Hc_mean, Hsat[0] - Hc_mean)
        data["Hsat_mean"] = np.abs(np.array(Hsat)).mean()
    else:
        data["Hc"] = [x for x in Hc if x is not None]
        data["Hc_mean"] = abs(data["Hc"][0])
        data["Hsat"] = [x for x in Hsat if x is not None]
        data["Hsat_mean"] = abs(data["Hsat"][0])
    data["Remenance"] = Mr

    bh = (-data.x) * data.y
    i = argmax(bh)
    data["BH_Max"] = max(bh)
    data["BH_Max_H"] = data.x[i]

    data["Area"] = data.integrate()
    return cls(data)
