"""Stoner .Analysis provides a subclass of :class:`.Data` that has extra analysis routines builtin."""

__all__ = ["AnalysisMixin"]
import numpy as np
import numpy.ma as ma
from scipy.integrate import cumtrapz

from .tools import isiterable, isTuple

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
        xdata = np.sort(np.append(-pxdata, pxdata))
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
            result_name (str):
                The metadata name for the final result
            header (str):
                The name of the header for the results column.
            output (Str):
                What to return - 'data' (default) - this object, 'result': final result
            bounds (callable):
                A function that evaluates for each row to determine if the data should be integrated over.
            **kargs:
                Other keyword arguements are fed direct to the scipy.integrate.cumtrapz method

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
                header = header if header is not None else f"Intergral of {self.column_headers[_.ycol]}"
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
                The column to normalise to, can be an integer or string. **Depricated** can also be a tuple (low,
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
