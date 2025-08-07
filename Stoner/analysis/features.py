#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature Finding functions for analysis code."""

__all__ = ["FeatureOpsMixin"]

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

from Stoner.tools import isiterable, isTuple
from Stoner.core.exceptions import assertion
from .utils import threshold


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
            xdata = self.column(xcol).ravel()
        xdata = interp1d(np.arange(len(self)), xdata, kind="cubic")

        possible_peaks = np.array(threshold(0, d1, rising=troughs, falling=peaks))
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
