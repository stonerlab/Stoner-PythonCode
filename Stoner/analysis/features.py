#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature Finding functions for analysis code."""

__all__ = ["FeatureOpsMixin"]

from inspect import getfullargspec

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from ..compat import string_types
from ..tools import isiterable, isTuple
from ..core.exceptions import assertion
from .utils import threshold as _threshold


class FeatureOpsMixin:

    """Mixin to provide additional functions to support finding features in a dataset."""

    def peaks(self, **kargs):
        """Locates peaks and/or troughs in a column of data by using SG-differentiation.

        Args:
            ycol (index):
                the column name or index of the data in which to search for peaks
            width (int or float):
                the expected minimum halalf-width of a peak in terms of the number of data points (int) or distance
                in x (float). This is used in the differnetiation code to find local maxima. Bigger equals less
                sensitive to experimental noise, smaller means better eable to see sharp peaks
            poly (int):
                the order of polynomial to use when differentiating the data to locate a peak. Must >=2, higher numbers
                will find sharper peaks more accurately but at the risk of finding more false positives.

        Keyword Arguments:
            significance (float):
                used to decide whether a local maxmima is a significant peak. Essentially just the curvature
                of the data. Bigger means less sensitive, smaller means more likely to detect noise. Default is the
                maximum curvature/(2*width)
            xcol (index or None):
                name or index of data column that p[provides the x-coordinate (default None)
            peaks (bool):
                select whether to measure peaks in data (default True)
            troughs (bool):
                select whether to measure troughs in data (default False)
            sort (bool):
                Sor the results by significance of peak
            modify (book):
                If true, then the returned object is a copy of self with only the peaks/troughs left in the data.
            full_data (bool):
                If True (default) then all columns of the data at which peaks in the *ycol* column are found.
                *modify* true implies *full_data* is also true. If *full_data* is False, then only the x-column
                values of the peaks are returned.

        Returns:
            (various):
                If *modify* is true, then returns a the AnalysisMixin with the data set to just the peaks/troughs.
                If *modify* is false (default), then the return value depends on *ycol* and *xcol*. If *ycol* is
                not None and *xcol* is None, then returns complete rows of data corresponding to the found
                peaks/troughs. If *xcol* is not None, or *ycol* is None and *xcol* is None, then returns a 1D array
                of the x positions of the peaks/troughs.

        See Also:
            User guide section :ref:`peak_finding`
        """
        width = kargs.pop("width", int(len(self) / 20))
        peaks = kargs.pop("peaks", True)
        troughs = kargs.pop("troughs", False)
        poly = kargs.pop("poly", 2)
        assertion(
            poly >= 2, "poly must be at least 2nd order in peaks for checking for significance of peak or through"
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
        setas = self.setas.clone  # pylint: disable=E0203
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
        pad = np.mean(d2[index_offset:-index_offset])
        d2[:index_offset] = pad
        d2[-index_offset:] = pad

        # Set the significance from the 2nd ifferential if not already set
        significance = kargs.pop(
            "significance", np.max(np.abs(d2)) / (2 * width)
        )  # Base an apriori significance on max d2y/dx2 / 20
        if isinstance(significance, int):  # integer significance is inverse to floating
            significance = np.max(np.abs(d2)) / significance  # Base an apriori significance on max d2y/dx2 / 20

        d2_interp = interp1d(np.arange(len(d2)), d2, kind="cubic")
        # Ensure we have some X-data
        if xcol is None:
            xdata = np.arange(len(self))
        else:
            xdata = self.column(xcol)
        xdata = interp1d(np.arange(len(self)), xdata, kind="cubic")

        possible_peaks = np.array(_threshold(0, d1, rising=troughs, falling=peaks))
        curvature = np.abs(d2_interp(possible_peaks))

        # Filter just the significant peaks
        possible_peaks = np.array([p for ix, p in enumerate(possible_peaks) if abs(curvature[ix]) > significance])
        # Sort in order of significance
        if sort:
            possible_peaks = np.take(possible_peaks, np.argsort(np.abs(d2_interp(possible_peaks))))

        xdat = xdata(possible_peaks + index_offset)

        if modify:
            self.data = self.interpolate(xdat, xcol=xcol, kind="cubic")
            ret = self
        elif full_data:
            ret = self.interpolate(xdat, kind="cubic", xcol=False)
        else:
            ret = xdat
        self.setas = setas
        # Return - but remembering to add back on the offset that we took off due to differentials not working at
        # start and end
        return ret

    def find_peaks(self, **kargs):
        """Interface to :py:func:`scipy.signal.find_peaks` for locating peaks in data.

        Args:
            ycol (index):
                the column name or index of the data in which to search for peaks

        Keyword Arguments:
            xcol (index):
                the column name or index of the x data that the peaks correspond to.
            height : number or ndarray or sequence, optional
                Required height of peaks. Either a number, ``None``, an array matching
                `ycol` or a 2-element sequence of the former. The first element is
                always interpreted as the  minimal and the second, if supplied, as the
                maximal required height.
            threshold : number or ndarray or sequence, optional
                Required threshold of peaks, the vertical distance to its neighbouring
                samples. Either a number, ``None``, an array matching `ycol` or a
                2-element sequence of the former. The first element is always
                interpreted as the  minimal and the second, if supplied, as the maximal
                required threshold.
            distance : number, optional
                Required minimal horizontal distance (>= 1) in samples between
                neighbouring peaks. Smaller peaks are removed first until the condition
                is fulfilled for all remaining peaks. If this is a *float* and *xcol* is set, then
                the units are in terms of the x-data, otherwise in rwo indices.
            prominence : number or ndarray or sequence, optional
                Required prominence of peaks. Either a number, ``None``, an array
                matching `ycol` or a 2-element sequence of the former. The first
                element is always interpreted as the  minimal and the second, if
                supplied, as the maximal required prominence.
            width : number or ndarray or sequence, optional
                Required width of peaks in samples. Either a number, ``None``, an array
                matching `ycol` or a 2-element sequence of the former. The first
                element is always interpreted as the  minimal and the second, if
                supplied, as the maximal required width. If this is a *float* and *xcol* is set, then
                the units are in terms of the x-data, otherwise in rwo indices.
            wlen : int, optional
                Used for calculation of the peaks prominences, thus it is only used if
                one of the arguments `prominence` or `width` is given. See argument
                `wlen` in `peak_prominences` for a full description of its effects.
            rel_height : float, optional
                Used for calculation of the peaks width, thus it is only used if `width`
                is given. See argument  `rel_height` in `peak_widths` for a full
                description of its effects.
            plateau_size : number or ndarray or sequence, optional
                Required size of the flat top of peaks in samples. Either a number,
                ``None``, an array matching `ycol` or a 2-element sequence of the former.
                The first element is always interpreted as the minimal and the second,
                if supplied as the maximal required plateau size. If this is a *float* and *xcol* is set, then
                the units are in terms of the x-data, otherwise in rwo indices.
            prefix (str):
                If et, then the metadata keys that return information about the peaks is returned with the given
                prefix. Default is None - no prefix.
            sort (bool):
                Sor the results by prominence of peak
            modify (book):
                If true, then the returned object is a copy of self with only the peaks left in the data.
            full_data (bool):
                If True (default) then all columns of the data at which peaks in the *ycol* column are found.
                *modify* true implies *full_data* is also true. If *full_data* is False, then only the x-column
                values of the peaks are returned.

        Returns:
            (various):
                If *modify* is true, then returns a the AnalysisMixin with the data set to just the peaks/troughs.
                If *modify* is false (default), then the return value depends on *ycol* and *xcol*. If *ycol* is
                not None and *xcol* is None, then returns complete rows of data corresponding to the found
                peaks/troughs. If *xcol* is not None, or *ycol* is None and *xcol* is None, then returns a 1D array
                of the x positions of the peaks/troughs.

        See Also:
            User guide section :ref:`peak_finding`
        """
        distance = kargs.pop("distance", None)
        width = kargs.pop("width", None)
        plateau_size = kargs.pop("plateau_size", None)
        sort = kargs.pop("sort", False)
        modify = kargs.pop("modify", False)
        bounds = kargs.pop("bounds", lambda x, y: True)
        prefix = kargs.pop("prefix", None)
        full_data = kargs.pop("full_data", True)
        _ = self._col_args(scalar=False, xcol=kargs.pop("xcol", None), ycol=kargs.pop("ycol", None))
        xcol, ycol = _.xcol, _.ycol
        if isiterable(ycol):
            ycol = ycol[0]

        if isinstance(width, float):  # Convert a floating point width unto an integer.
            xmin, xmax = self.span(xcol)
            width = int(len(self) * width / (xmax - xmin))
        elif isTuple(width, float, float):
            xmin, xmax = self.span(xcol)
            width = int(len(self) * width[0] / (xmax - xmin)), int(len(self) * width[1] / (xmax - xmin))
        if width is not None:
            kargs["width"] = width

        if isinstance(distance, float):  # Convert a floating point width unto an integer.
            xmin, xmax = self.span(xcol)
            distance = int(np.ceil(len(self) * distance / (xmax - xmin)))
        if distance is not None:
            kargs["distance"] = distance

        if isinstance(plateau_size, float):  # Convert a floating point plateau_size unto an integer.
            xmin, xmax = self.span(xcol)
            plateau_size = int(len(self) * plateau_size / (xmax - xmin))
        elif isTuple(plateau_size, float, float):
            xmin, xmax = self.span(xcol)
            plateau_size = (
                int(len(self) * plateau_size[0] / (xmax - xmin)),
                int(len(self) * plateau_size[1] / (xmax - xmin)),
            )
        if plateau_size is not None:
            kargs["plateau_size"] = plateau_size

        seek = self.search(xcol, bounds)
        peaks, data = find_peaks(seek[:, ycol], **kargs)
        peaks = self.data.i[seek.i[peaks]]  # de-reference frombounded data back to main dataset

        for sort_key in ["prominences", "peak_heights", "widths"]:
            if sort_key in data:
                break
        else:
            sort_key = None
        if sort and sort_key:
            idx = np.sort(np.array(list(zip(data[sort_key], np.arange(peaks.size)))), axis=0)[:, 1].astype(int)
            peaks = peaks[idx]
            for k in data:
                data[k] = data[k][idx]
        xmin, xmax = self.span(_.xcol)
        xconv = len(self) / (xmax - xmin)
        for k, v in data.items():
            if k.startswith("left") or k.startswith("right") or k == "widths":
                data[k] = v / xconv + (xmin if k != "widths" else 0)
        peak_data = self.data[peaks, :]
        for k, v in data.items():
            if prefix is None:
                self[k] = v
            else:
                self[f"{prefix}:{k}"] = v
        if modify:
            self.data = peak_data
            return self
        if full_data:
            return peak_data, data
        return peak_data[_.xcol], peak_data[_.yxol], data

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
        _ = self._col_args(xcol=xcol, ycol=ycol, scalar=True)
        points = self.column([_.xcol, _.ycol])
        points = points[points[:, 0].argsort(), :]
        points[:, 0] += min_overlap
        otherpoints = other.column([_.xcol, _.ycol])
        otherpoints = otherpoints[otherpoints[:, 0].argsort(), :]
        self_second = np.max(points[:, 0]) > np.max(otherpoints[:, 0])
        if overlap is None:  # Calculate the overlap
            lower = max(np.min(points[:, 0]), np.min(otherpoints[:, 0]))
            upper = min(np.max(points[:, 0]), np.max(otherpoints[:, 0]))
        elif isinstance(overlap, int) and overlap > 0:
            if self_second:
                lower = points[0, 0]
                upper = points[overlap, 0]
            else:
                lower = points[-overlap - 1, 0]
                upper = points[-1, 0]
        elif (
            isinstance(overlap, tuple)
            and len(overlap) == 2
            and isinstance(overlap[0], float and isinstance(overlap[1], float))
        ):
            lower = min(overlap)
            upper = max(overlap)
        inrange = np.logical_and(points[:, 0] >= lower, points[:, 0] <= upper)
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
            A0 = np.mean(xp) - np.mean(x)
            C0 = np.mean(yp) - np.mean(y)
            B0 = (np.max(yp) - np.min(yp)) / (np.max(y) - np.min(y))
            p = np.array([0, A0, B0, C0])
            assertion(isinstance(mode, string_types), "mode keyword should be a string if func is not defined")
            mode = mode.lower()
            assertion(mode in opts, f"mode keyword should be one of {opts.keys}")
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
        set1 = np.append(x, y)
        set2 = np.append(xp, yp)
        assertion(len(set1) == len(set2), "The number of points in the overlap are different in the two data sets")

        def transform(set1, *p):
            """Construct the wrapper function to fit for transform."""
            m = int(len(set1) / 2)
            x = set1[:m]
            y = set1[m:]
            tmp = func(x, y, *p)
            out = np.append(tmp[0], tmp[1])
            return out

        popt, pcov = curve_fit(transform, set1, set2, p0=p0)  # Curve fit for optimal A,B,C
        perr = np.sqrt(np.diagonal(pcov))
        self.data[:, _.xcol], self.data[:, _.ycol] = func(self.data[:, _.xcol], self.data[:, _.ycol], *popt)
        self["Stitching Coefficients"] = list(popt)
        self["Stitching Coeffient Errors"] = list(perr)
        self["Stitching overlap"] = (lower, upper)
        self["Stitching Window"] = num_pts

        return self

    def threshold(self, threshold, **kargs):
        """Find partial indices where the data in column passes the threshold, rising or falling.

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
            There has been an API change. Versions prior to 0.1.9 placed the column before the threshold in the
            positional argument list. In order to support the use of assigned columns, this has been swapped to the
            present order.
        """
        DataArray = type(self.data)
        col = kargs.pop("col", None)
        xcol = kargs.pop("xcol", None)
        _ = self._col_args(xcol=xcol, ycol=col)

        col = _.ycol
        if xcol is None and _.has_xcol:
            xcol = _.xcol

        rising = kargs.pop("rising", True)
        falling = kargs.pop("falling", False)
        all_vals = kargs.pop("all_vals", False)

        current = self.column(col)

        # Recursively call if we've got an iterable threshold
        if isiterable(threshold):
            if isinstance(xcol, bool) and not xcol:
                ret = np.zeros((len(threshold), self.shape[1]))
            else:
                ret = np.zeros_like(threshold).view(type=DataArray)
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
                    ret = np.atleast_2d(ret)
                    ret.column_headers = [self.column_headers[self.find_col(xcol)]]
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
