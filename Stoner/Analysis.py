"""Stoner .Analysis provides a subclass of :class:`.Data` that has extra analysis routines builtin."""

__all__ = ["AnalysisMixin", "GetAffineTransform", "ApplyAffineTransform"]
from inspect import getfullargspec
import numpy as np
import numpy.ma as ma

try:
    from scipy.integrate import cumtrapz
except ImportError:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from .tools import isiterable, isTuple
from .compat import string_types
from .core.exceptions import assertion
from .analysis.utils import threshold as _threshold, ApplyAffineTransform, GetAffineTransform

# from matplotlib.pylab import * #Surely not?


class AnalysisMixin:
    """A mixin calss designed to work with :py:class:`Stoner.Core.DataFile` to provide additional analysis methods."""

    def __dir__(self):
        """Handle the local attributes as well as the inherited ones."""
        attr = dir(type(self))
        attr.extend(super().__dir__())
        attr.extend(list(self.__dict__.keys()))
        attr = list(set(attr))
        return sorted(attr)

    def apply(self, func, col=None, replace=True, header=None, **kargs):
        """Apply the given function to each row in the data set and adds to the data set.

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

            and should return either a single float, in which case it will be used to repalce the specified column,
            or an array, in which case it is used to completely replace the row of data.

            If the function returns a complete row of data, then the *replace* parameter will cause the return
            value to be a new datafile, leaving the original unchanged. The *headers* parameter can give the complete
            column headers for the new data file.

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
            nc = np.zeros((len(self), len(ret)))
        else:
            nc = np.zeros(len(self))
        # Evaluate the data row by row
        for ix, r in enumerate(self.rows()):
            ret = func(r, **kargs)
            if isiterable(ret) and not isinstance(ret, np.ndarray):
                ret = np.ma.MaskedArray(ret)
            nc[ix] = ret
        # Work out how to handle the result
        if nc.ndim == 1:
            if header is None:
                header = func.__name__
            self.add_column(nc, header=header, index=col, replace=replace, setas=self.setas[col])
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
        return self.del_rows(col, lambda x, y: x < clipper[0] or x > clipper[1])

    def decompose(self, xcol=None, ycol=None, sym=None, asym=None, replace=True, hysteretic=False, **kwords):
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
            hysteretic (book)L
                Look separately for outgoing and incoming data first.

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
        if not isinstance(ycol, list):
            ycol = [ycol]

        if hysteretic:
            from .Util import split_up_down

            fldr = split_up_down(self, self.xcol)
            for grp in ["rising", "falling"]:
                for f in fldr[grp][1:]:
                    fldr[grp][0] += f
            rising = fldr["rising"][0].sort(xcol)
            falling = fldr["falling"][0].sort(xcol)
            points = fldr["rising"][0].size
        else:
            rising = self.clone.sort(xcol)
            falling = rising.clone
            points = rising.x.size

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
        rising_vals = rising_func((self // xcol).view(np.ndarray))
        falling_vals = falling_func(-(self // xcol).view(np.ndarray))

        symd = (rising_vals + falling_vals) / 2
        asymd = (rising_vals - falling_vals) / 2

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

    def integrate(
        self,
        xcol=None,
        ycol=None,
        result=None,
        header=None,
        result_name=None,
        output="data",
        bounds=lambda x, y: True,
        **kargs,
    ):
        """Integrate a column of data, optionally returning the cumulative integral.

        Args:
            xcol (index):
                The X data column index (or header)
            ycol (index)
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
            **kargs:
                Other keyword arguments are fed direct to the scipy.integrate.cumtrapz method

        Returns:
            (:py:class:`Stoner.Data`):
                The newly modified Data object.

        Note:
            This is a pass through to the scipy.integrate.cumtrapz routine which just uses trapezoidal integration.
            A better alternative would be to offer a variety of methods including simpson's rule and interpolation
            of data. If xcol or ycol are not specified then the current values from the
            :py:attr:`Stoner.Core.DataFile.setas` attribute are used.
        """
        _ = self._col_args(xcol=xcol, ycol=ycol)

        working = self.search(_.xcol, bounds)
        working = ma.mask_rowcols(working, axis=0)
        xdat = working[:, self.find_col(_.xcol)]
        ydat = working[:, self.find_col(_.ycol)]
        ydat = np.atleast_2d(ydat).T

        final = []
        for i in range(ydat.shape[1]):
            yd = ydat[:, i]
            resultdata = cumtrapz(yd, xdat, **kargs)
            resultdata = np.append(np.array([0]), resultdata)
            if result is not None:
                header = header if header is not None else f"Integral of {self.column_headers[_.ycol]}"
                if isinstance(result, bool) and result:
                    self.add_column(resultdata, header=header, replace=False)
                else:
                    result_name = self.column_headers[self.find_col(result)]
                    self.add_column(resultdata, header=header, index=result, replace=(i == 0))
            final.append(resultdata[-1])
        if len(final) == 1:
            final = final[0]
        else:
            final = np.array(final)
        result_name = result_name if result_name is not None else header if header is not None else "Area"
        self[result_name] = final
        if output.lower() == "result":
            return final
        return self

    def normalise(self, target=None, base=None, replace=True, header=None, scale=None, limits=(0.0, 1.0)):
        """Normalise data columns by dividing through by a base column value.

        Args:
            target (index):
                One or more target columns to normalise can be a string, integer or list of strings or integers.
                If None then the default 'y' column is used.

        Keyword Arguments:
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
        _ = self._col_args(scalar=True, ycol=target)

        target = _.ycol
        if not isinstance(target, list):
            target = [self.find_col(target)]
        for t in target:
            if header is None:
                header = self.column_headers[self.find_col(t)] + "(norm)"
            else:
                header = str(header)
            if not isTuple(base, float, float) and base is not None:
                self.divide(t, base, header=header, replace=replace)
            else:
                if isTuple(base, float, float):
                    scale = base
                elif scale is None:
                    scale = (-1.0, 1.0)
                if not isTuple(scale, float, float):
                    raise ValueError("limit parameter is either None, or limit or base is a tuple of two floats.")
                data = self.column(t).ravel()
                data = np.sort(data[~np.isnan(data)])
                if limits != (0.0, 1.0):
                    low, high = limits
                    low = data[int(low * data.size)]
                    high = data[int(high * data.size)]
                else:
                    high = data.max()
                    low = data.min()
                data = np.copy(self.data[:, t])
                data = np.where(data > high, high, np.where(data < low, low, data))
                scl, sch = scale
                data = (data - low) / (high - low) * (sch - scl) + scl
                setas = self.setas.clone
                self.add_column(data, index=t, replace=replace, header=header)
                self.setas = setas
        return self

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
        match overlap:
            case int() if overlap > 0:
                if self_second:
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
        self["Stitching Coefficient Errors"] = list(perr)
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
