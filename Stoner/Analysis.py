"""Stoner .Analysis provides a subclass of :py:class:`Stoner.Core.DataFile` that has extra analysis routines builtin.

Provides  :py:class:`AnalyseFile` - DataFile with extra bells and whistles.
"""
from Stoner.compat import *
from Stoner.Core import DataFile
import Stoner.FittingFuncs
import Stoner.nlfit
import numpy as _np_
import numpy.ma as ma
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from inspect import getargspec
from collections import Iterable
from lmfit.model import Model,ModelFit

import sys

def cov2corr(M):
    """ Converts a covariance matrix to a correlation matrix. Taken from bvp.utils.misc

    Args:
        M (2D _np_.array): Co-varriance Matric

    Returns:
        Correlation Matrix.
    """
    if (not isinstance(M, _np_.ndarray)) or (not (len(M.shape) == 2)) or (not(M.shape[0] == M.shape[1])):
        raise ValueError('cov2corr expects a square ndarray, got %s' % M)

    if _np_.isnan(M).any():
        raise ValueError('Found NaNs in my covariance matrix: %s' % M)

    # TODO check Nan and positive diagonal
    d = M.diagonal()
    if (d < 0).any():
        raise ValueError('Expected positive elements for square matrix, got diag = %s' % d)

    n = M.shape[0]
    R = _np_.ndarray((n, n))
    for i in range(n):
        for j in range(n):
            d = M[i, j] / _np_.sqrt(M[i, i] * M[j, j])
            R[i, j] = d

    return R

class AnalyseFile(DataFile):
    """:py:class:`Stoner.Analysis.AnalyseFile` extends :py:class:`Stoner.Core.DataFile` with numpy and scipy passthrough functions.

    Note:
        There is no separate constructor for this class - it inherits from DataFile

    """

    def SG_Filter(self, col=None, points=15, poly=1, order=0,result=None, replace=False, header=None):
        """ Implements Savitsky-Golay filtering of data for smoothing and differentiating data.

        Args:
            col (index): Column of Data to be filtered
            prints (int): Number of data points to use in the filtering window. Should be an odd number > poly+1 (default 15)

        Keyword Arguments:
            poly (int): Order of polynomial to fit to the data. Must be equal or greater than order (default 1)
            order (int): Order of differentiation to carry out. Default=0 meaning smooth the data only.

        Returns:
            A numpy array representing the smoothed or differentiated data.

        Notes:
            If col is not specified or is None then the :py:atrt:`DataFile.setas` column assignments are used
            to set an x and y column. If col is a tuple, then it is assumed to secify and x-column and y-column
            for differentiating data. This is now a pass through to :py:func:`scipy.signal.savgol_filter`
        """
        from Stoner.Util import ordinal
        if points % 2 ==0: #Ensure window length is odd
                points+=1
        if col is None:
            cols=self.setas._get_cols()
            col=(cols["xcol"],cols["ycols"][0])
        if isinstance(col, (list,tuple)):
            data=self.column(list(col)).T
            ddata=savgol_filter(data,window_length=points,polyorder=poly,deriv=order,mode="interp")
            r=ddata[1:]/ddata[0]
        else:
            data=self.column(col)
            r=savgol_filter(data,window_length=points,polyorder=poly,deriv=order,mode="interp")
        if result is not None:
            if not isinstance(header, string_types):
                header='{} after {} order Savitsky-Golay Filter'.format(self.column_headers[self.find_col(col)],ordinal(order))
            if isinstance(result, bool) and result:
                result=self.shape[1]-1
            self.add_column(r,header,index=result,replace=replace)

        return r


    def __get_math_val(self,col):
        """Utility routine to interpret col as either a column index or value or an array of values.

        Args:
            col (various): If col can be interpreted as a column index then return the first matching column.
                If col is a 1D array of the same length as the data then just return the data. If col is a
                float then just return it as a float.

        Returns:
            The matching data.
        """
        if isinstance(col,index_types):
            col=self.find_col(col)
            if isinstance(col,list):
                col=col[0]
            data=self.column(col)
            name=self.column_headers[col]
        elif isinstance(col,_np_.ndarray) and len(col.shape)==1 and len(col)==len(self):
            data=col
            name="data"
        elif isinstance(col,float):
            data=col*_np_.ones(len(self))
            name=str(col)
        else:
            raise RuntimeError("Bad column index: {}".format(col))
        return data,name


    def __mpf_fn(self, p, **fa):
        """Internal routine for general non-linear least squeares fitting.

        Args:
            p (list or tuple): fitting parameter values for fitting function .

        Keyword Arguments:
            func (callable): fitting function
            x (array of float): X values
            y(array of float): Y data values
            err (array of float): Weightings of data values

        Note:
            All other keywords are passed to the fitting function directly.

        Returns:
            Difference between model values ad actual y values divided by weighting.
        """
        func=fa['func']
        x=fa['x']
        y=fa['y']
        err=fa['err']
        del(fa['x'])
        del(fa['y'])
        del(fa['func'])
        del(fa['fjac'])
        del(fa['err'])
        model = func(x, p, **fa)
        # stop the calculation.
        status = 0
        return [status, (y-model)/err]


    def __threshold(self, threshold, data, rising=True, falling=False):
        """ Internal function that implements the threshold method - also used in peak-finder

        Args:
            threshold (float): Threshold valuye in data to look for
            rising (bool): Find points where data is rising up past threshold
            falling (bool): Find points where data is falling below the threshold

        Returns:
            A numpy array of fractional indices where the data has crossed the threshold assuming a
            straight line interpolation between two points.
        """
        current=data
        previous=_np_.roll(current, 1)
        index=_np_.arange(len(current))
        sdat=_np_.column_stack((index, current, previous))
        if rising==True and falling==False:
            expr=lambda x:(x[1]>=threshold) & (x[2]<threshold)
        elif rising==True and falling==True:
            expr=lambda x:((x[1]>=threshold) & (x[2]<threshold)) | ((x[1]<=threshold) & (x[2]>threshold))
        elif rising==False and falling==True:
            expr=lambda x:(x[1]<=threshold) & (x[2]>threshold)
        else:
            expr=lambda x:False
        intr=lambda x:x[0]-1+(x[1]-threshold)/(x[1]-x[2])
        return _np_.array([intr(x) for x in sdat if expr(x) and intr(x)>0])


    def __dir__(self):
        """Handles the local attributes as well as the inherited ones"""
        attr=dir(type(self))
        attr.extend(super(AnalyseFile,self).__dir__())
        attr.extend(list(self.__dict__.keys()))
        attr=list(set(attr))
        return sorted(attr)


    def add(self, a, b, replace=False, header=None):
        """Add one column, number or array (b) to another column (a).

        Args:
            a (index): First column to work with
            b (index, float or 1D array):  Second column to work with.

        Keyword Arguments:
            header (string or None): new column header  (defaults to a-b
            replace (bool): Replace the a column with the new data

        Returns:
            A copy of the new data object

        If a and b are tuples of length two, then the firstelement is assumed to be the value and
        the second element an uncertainty in the value. The uncertainties will then be propagated and an
        additional column with the uncertainites will be added to the data."""
        a=self.find_col(a)
        if isinstance(a,tuple) and isinstance(b,tuple) and len(a)==2 and len(b)==2: #Error columns on
            (a,e1)=a
            (b,e2)=b
            e1data=self.__get_math_val(e1)[0]
            e2data=self.__get_math_val(e2)[0]
            err_header=None
            err_calc=lambda adata,bdata,e1data,e2data: _np_.sqrt(e1data**2+e2data**2)
        else:
            err_calc=None
        adata,aname=self.__get_math_val(a)
        bdata,bname=self.__get_math_val(b)
        if isinstance(header,tuple) and len(header)==0:
            header,err_header=header
        if header is None:
            header="{}+{}".format(aname,bname)
        if err_calc is not None and err_header is None:
            err_header="Error in "+header
        if err_calc is not None:
            err_data=err_calc(adata,bdata,e1data,e2data)
        self.add_column((adata+bdata), header, a, replace=replace)
        if err_calc is not None:
            self.add_column(err_data,err_header,a+1,replace=False)
        return self


    def apply(self, func, col=None, replace=True, header=None,**kargs):
        """Applies the given function to each row in the data set and adds to the data set.

        Args:
            func (callable): A function that takes a numpy 1D array representing each row of data
            col (index): The column in which to place the result of the function

        Keyword Arguments:
            replace (bool): Isnert a new data column (False) or replace the data column (True, default)
            header (string or None): The new column header (defaults to the name of the function func

        Returns:
            A copy of the current instance
        """

        if col is None:
           col=self.setas._get_cols()["ycol"][0]
        col=self.find_col(col)
        nc=_np_.zeros(len(self))
        for i,r in enumerate(self.rows()):
            ret=func(r)
            if isinstance(ret,Iterable):
                if len(ret)==len(r):
                    ret=ret[col]
                else:
                    ret=ret[0]
            nc[i]=ret
        if header==None:
            header=func.__name__
        if replace!=True:
            self=self.add_column(nc, header, col)
        else:
            self.data[:, col]=_np_.reshape(nc, -1)
            self.column_headers[col]=header
        return self

    def bin(self,xcol=None,ycol=None,bins=0.03,mode="log",**kargs):
        """Bin x-y data into new values of x with an error bar.

        Args:
            xcol (index): Index of column of data with X values
            ycol (index): Index of column of data with Y values
            bins (int or float): Number of bins (if integer) or size of bins (if float)
            mode (string): "log" or "lin" for logarithmic or linear binning

        Keyword Arguments:
            yerr (index): Column with y-error data if present.
            bin_start (float): Manually override the minimum bin value
            bin_stop (float): Manually override the maximum bin value

        Returns:
            n x 3 array of x-bin, y-bin, y-bin-error where n is the number of bins

        Note:
            Algorithm inspired by MatLab code wbin,    Copyright (c) 2012:
            Michael Lindholm Nielsen
        """

        if "yerr" in kargs:
            yerr=kargs["yerr"]
        else:
            yerr=None

        if None in (xcol,ycol):
            cols=self.setas._get_cols()
            if xcol is None:
                xcol=cols["xcol"]
            if ycol is None:
                ycol=cols["ycol"]
            if "yerr" not in kargs and cols["has_yerr"]:
                yerr=cols["yerr"]

        bin_left,bin_right,bin_centres=self.make_bins(xcol,bins,mode,**kargs)

        ycol=self.find_col(ycol)
        if yerr is not None:
            yerr=self.find_col(yerr)

        ybin=_np_.zeros((len(bin_left),len(ycol)))
        ebin=_np_.zeros((len(bin_left),len(ycol)))
        nbins=_np_.zeros((len(bin_left),len(ycol)))
        xcol=self.find_col(xcol)
        i=0

        for limits in zip(bin_left,bin_right):
            data=self.search(xcol,limits)
            if yerr is not None:
                w=1.0/data[:,yerr]**2
                W=_np_.sum(w,axis=0)
                e=1.0/_np_.sqrt(W)
            else:
                w=_np_.ones((data.shape[0],len(ycol)))
                W=data.shape[0]
                e=_np_.std(data[:,ycol],axis=0)/_np_.sqrt(W)
            y=_np_.sum(data[:,ycol]*(w/W),axis=0)
            ybin[i,:]=y
            ebin[i,:]=e
            nbins[i,:]=data.shape[0]
            i+=1
        return (bin_centres,ybin,ebin,nbins)

    def chi2mapping(self, ini_file, func):
        """Non-linear fitting using the :py:mod:`Stoner.nlfit` module.

        Args:
            ini_file (string): Path to ini file with model
            func_name (string or callable): Name of function to fit with (as seen in FittingFuncs.py module in Stoner)
                or the function itself.

        ReturnsL
            AnalyseFile instance, matplotlib.fig instance (or None if plotting disabled), DataFile instance of parameter steps"""
        return Stoner.nlfit.nlfit(ini_file, func, data=self, chi2mapping=True)


    def clip(self, clipper,column=None):
        """Clips the data based on the column and the clipper value.

        Args:
            column (index): Column to look for the maximum in
            clipper (tuple or array): Either a tuple of (min,max) or a numpy.ndarray -
                in which case the max and min values in that array will be
                used as the clip limits
        Returns:
            This instance."

        Note:
            If column is not defined (or is None) the :py:attr:`DataFile.setas` column
            assignments are used.
        """
        if column is None:
            col=self.setas._get_cols("ycol")
        else:
            col=self.find_col(column)
        clipper=(min(clipper), max(clipper))
        self=self.del_rows(col, lambda x, y:x<clipper[0] or x>clipper[1])
        return self


    def curve_fit(self, func,  xcol=None, ycol=None, p0=None, sigma=None,**kargs):
        """General curve fitting function passed through from scipy.

        Args:
            func (callable): The fitting function with the form def f(x,*p) where p is a list of fitting parameters
            xcol (index): The index of the x-column data to fit
            ycol (index): The index of the y-column data to fit

        Keyword Arguments:
            p0 (list, tuple or array): A vector of initial parameter values to try
            sigma (index): The index of the column with the y-error bars
            bounds (callable) A callable object that evaluates true if a row is to be included. Should be of the form f(x,y)
            result (bool): Determines whether the fitted data should be added into the DataFile object. If result is True then
                the last column will be used. If result is a string or an integer then it is used as a column index.
                Default to None for not adding fitted data
            replace (bool): Inidcatesa whether the fitted data replaces existing data or is inserted as a new column (default False)
            header (string or None): If this is a string then it is used as the name of the fitted data. (default None)
            absolute_sigma (bool, defaults to True) If False, `sigma` denotes relative weights of the data points.
                The returned covariance matrix `pcov` is based on *estimated*
                errors in the data, and is not affected by the overall
                magnitude of the values in `sigma`. Only the relative
                magnitudes of the `sigma` values matter.
                If True, `sigma` describes one standard deviation errors of
                the input data points. The estimated covariance in `pcov` is
                based on these values.
            output (str, default "fit"): Specifiy what to return.

        Returns:
            popt (array): Optimal values of the fitting parameters p
            pcov (2d array): The variance-co-variance matrix for the fitting parameters.
            The return value is determined by the *output* parameter. Options are:
                * "ffit"    (tuple of popt,pcov)
                * "row"     just a one dimensional numpy array of the fit paraeters interleaved with their uncertainties
                * "full"    a tuple of (popt,pcov,dictionary of optional outputs, message, return code, row).
        Note:
            If the columns are not specified (or set to None) then the X and Y data are taken using the
            :py:attr:`DataFile.setas` attribute.

            The fitting function should have prototype y=f(x,p[0],p[1],p[2]...)
            The x-column and y-column can be anything that :py:meth:`Stoner.Core.DataFile.find_col` can use as an index
            but typucally either strings to be matched against column headings or integers.
            The initial parameter values and weightings default to None which corresponds to all parameters starting
            at 1 and all points equally weighted. The bounds function has format b(x, y-vec) and rewturns true if the
            point is to be used in the fit and false if not.

        See Also:
            :py:meth:`Stoner.Analysis.AnalyseFile.lmfit`
        """

        bounds=kargs.pop("bounds",lambda x, y: True)
        result=kargs.pop("result",None)
        replace=kargs.pop("replace",False)
        header=kargs.pop("header",None)
        #Support either scale_covar or absolute_sigma, the latter wins if both supplied
        scale_covar=kargs.pop("scale_covar",False)
        absolute_sigma=kargs.pop("absolute_sigma",not scale_covar)
        #Support both asrow and output, the latter wins if both supplied
        asrow=kargs.pop("asrow",False)
        output=kargs.pop("output","row" if asrow else "fit")
        if output=="full":
            kargs["full_output"]=True

        if None in (xcol,ycol,sigma):
            cols=self.setas._get_cols()
            if xcol is None:
                xcol=cols["xcol"]
            if ycol is None:
                ycol=cols["ycol"][0]
            if sigma is None:
                if len(cols["yerr"])>0:
                    sigma=cols["yerr"][0]


        working=self.search(xcol, bounds)
        working=ma.mask_rowcols(working,axis=0)
        if sigma is not None:
            sigma=working[:,self.find_col(sigma)]
        xdat=working[:,self.find_col(xcol)]
        ydat=working[:,self.find_col(ycol)]
        ret=curve_fit(func,  xdat,ydat, p0=p0, sigma=sigma, absolute_sigma=absolute_sigma,**kargs)
        popt=ret[0]
        pcov=ret[1]
        if result is not None:
            args=getargspec(func)[0]
            for i in range(len(popt)):
                self['Fit '+func.__name__+'.'+str(args[i+1])]=popt[i]
            xc=self.find_col(xcol)
            if not isinstance(header, string_types):
                header='Fitted with '+func.__name__
            if isinstance(result, bool) and result:
                result=self.shape[1]-1
            self.apply(lambda x:func(x[xc], *popt), result, replace=replace, header=header)
        row=_np_.array([])
        for i in range(len(popt)):
            row=_np_.append(row,[popt[i],_np_.sqrt(pcov[i,i])])
        ret=ret+(row,)
        retval={"fit":(popt,pcov),"row":row,"full":ret}
        if output not in retval:
            raise RuntimeError("Specified output: {}, from curve_fit not recognised".format(kargs["output"]))
        return retval[output]

    def decompose(self,xcol=None,ycol=None,sym=None, asym=None,replace=True, **kwords):
        """Given (x,y) data, decomposes the y part into symmetric and antisymmetric contributions in x.

        Keyword Arguments:
            xcol (index): Index of column with x data - defaults to first x column in self.setas
            ycol (index or list of indices): indices of y column(s) data
            sym (index): Index of column to place symmetric data in default, append to end of data
            asym (index): Index of column for asymmetric part of ata. Defaults to appending to end of data
            replace (bool): Overwrite data with output (true)

        Returns:
            A copy of the newly modified AnalyseFile.
        """
        if xcol is None and ycol is None:
            if "_startx" in kwords:
                startx=kwords["_startx"]
                del kwords["_startx"]
            else:
                startx=0
            cols=self.setas._get_cols(startx=startx)
            xcol=cols["xcol"]
            ycol=cols["ycol"]
        xcol=self.find_col(xcol)
        ycol=self.find_col(ycol)
        if isinstance(ycol,list):
            ycol=ycol[0] # FIXME should work with multiple output columns
        pxdata=self.search(xcol,lambda x,r:x>0,xcol)
        xdata=_np_.sort(_np_.append(-pxdata,pxdata))
        self.data=self.interpolate(xdata,xcol=xcol)
        ydata=self.data[:,ycol]
        symd=(ydata+ydata[::-1])/2
        asymd=(ydata-ydata[::-1])/2
        if sym is None:
            self&=symd
            self.column_headers[-1]="Symmetric Data"
        else:
            self.add_column(symd,"Symmetric Data",index=sym,replace=replace)
        if asym is None:
            self&=asymd
            self.column_headers[-1]="Asymmetric Data"
        else:
            self.add_column(asymd,"Symmetric Data",index=asym,replace=replace)

        return self

    def diffsum(self, a, b, replace=False, header=None):
        """Subtract one column, number or array (b) from another column (a) and divbdatae by their sums.

        Args:
            a (index): First column to work with
            b (index, float or 1D array):  Second column to work with.

        Keyword Arguments:
            header (string or None): new column header  (defaults to a-b
            replace (bool): Replace the a column with the new data

        Returns:
            A copy of the new data object

        If a and b are tuples of length two, then the firstelement is assumed to be the value and
        the second element an uncertainty in the value. The uncertainties will then be propagated and an
        additional column with the uncertainites will be added to the data."""
        if isinstance(a,tuple) and isinstance(b,tuple) and len(a)==2 and len(b)==2: #Error columns on
            (a,e1)=a
            (b,e2)=b
            e1data=self.__get_math_val(e1)[0]
            e2data=self.__get_math_val(e2)[0]
            err_header=None
            err_calc=lambda adata,bdata,e1data,e2data: _np_.sqrt((1.0/(adata+bdata)-(adata-bdata)/(adata+bdata)**2)**2*e1data**2+(-1.0/(adata+bdata)-(adata-bdata) / (adata+bdata)**2)**2*e2data**2)
        else:
            err_calc=None
        adata,aname=self.__get_math_val(a)
        bdata,bname=self.__get_math_val(b)
        if isinstance(header,tuple) and len(header)==0:
            header,err_header=header
        if header is None:
            header="({}-{})/({}+{})".format(aname,bname,aname,bname)
        if err_calc is not None and err_header is None:
            err_header="Error in "+header
        if err_calc is not None:
            err_data=err_calc(adata,bdata,e1data,e2data)
        self.add_column((adata-bdata)/(adata+bdata), header, a, replace=replace)
        if err_calc is not None:
            self.add_column(err_data,err_header,a+1,replace=False)
        return self


    def divide(self, a, b, replace=False, header=None):
        """Divide one column (a) by  another column, number or array (b).

        Args:
            a (index): First column to work with
            b (index, float or 1D array):  Second column to work with.

        Keyword Arguments:
            header (string or None): new column header  (defaults to a-b
            replace (bool): Replace the a column with the new data

        Returns:
            A copy of the new data object

        If a and b are tuples of length two, then the firstelement is assumed to be the value and
        the second element an uncertainty in the value. The uncertainties will then be propagated and an
        additional column with the uncertainites will be added to the data."""
        if isinstance(a,tuple) and isinstance(b,tuple) and len(a)==2 and len(b)==2: #Error columns on
            (a,e1)=a
            (b,e2)=b
            e1data=self.__get_math_val(e1)[0]
            e2data=self.__get_math_val(e2)[0]
            err_header=None
            err_calc=lambda adata,bdata,e1data,e2data: _np_.sqrt((e1data/adata)**2+(e2data/bdata)**2)*adata*bdata
        else:
            err_calc=None
        adata,aname=self.__get_math_val(a)
        bdata,bname=self.__get_math_val(b)
        if isinstance(header,tuple) and len(header)==0:
            header,err_header=header
        if header is None:
            header="{}/{}".format(aname,bname)
        if err_calc is not None and err_header is None:
            err_header="Error in "+header
        if err_calc is not None:
            err_data=err_calc(adata,bdata,e1data,e2data)
        self.add_column((adata/bdata), header, a, replace=replace)
        if err_calc is not None:
            self.add_column(err_data,err_header,a+1,replace=False)
        return self


    def integrate(self,xcol=None,ycol=None,result=None,result_name=None, bounds=lambda x,y:True,**kargs):
        """Inegrate a column of data, optionally returning the cumulative integral.

        Args:
            xcol (index): The X data column index (or header)
            ycol (index) The Y data column index (or header)

        Keyword Arguments:
            result (index or None): Either a column index (or header) to overwrite with the cumulative data,
                or True to add a new column or None to not store the cumulative result.
            result_name (string): The new column header for the results column (if specified)
            bounds (callable): A function that evaluates for each row to determine if the data should be integrated over.
            kargs: Other keyword arguements are fed direct to the scipy.integrate.cumtrapz method

        Returns:
            The final integration result

        Note:
            This is a pass through to the scipy.integrate.cumtrapz routine which just uses trapezoidal integration. A better alternative would be
            to offer a variety of methods including simpson's rule and interpolation of data. If xcol or ycol are not specified then
            the current values from the :py:attr:`Stoner.Core.DataFile.setas` attribute are used.


            """
        if xcol is None or ycol is None:
            cols=self.setas._get_cols()
            if xcol is None:
                xcol=cols["xcol"]
            if ycol is None:
                ycol=cols["ycol"]
        working=self.search(xcol, bounds)
        working=ma.mask_rowcols(working,axis=0)
        xdat=working[:,self.find_col(xcol)]
        ydat=working[:,self.find_col(ycol)]
        final=[]
        for i in range(ydat.shape[1]):
            yd=ydat[:,i]
            resultdata=cumtrapz(xdat,yd,**kargs)
            resultdata=_np_.append(_np_.array([0]),resultdata)
            if result is not None:
                if isinstance(result,bool) and result:
                    self.add_column(resultdata,result_name,replace=False)
                else:
                    result_name=self.column_headers[self.find_col(result)]
                    self.add_column(resultdata,result_name,index=result,replace=(i==0))
            final.append(resultdata[-1])
        if len(final)==1:
            final=final[0]
        else:
            final=_np_.array(final)
        return final


    def interpolate(self, newX,kind='linear', xcol=None ):
        """AnalyseFile.interpolate(newX, kind='linear",xcol=None).

        Args:
            ewX (1D array): Row indices or X column values to interpolate with

        Keyword Arguments:
            kind (string): Type of interpolation function to use - does a pass through from numpy. Default is linear.
            xcol (index or None): Column index or label that contains the data to use with newX to determine which rows to return. Defaults to None.peaks

        Returns:
            2D numpy array representing a section of the current object's data.

        Note:
            Returns complete rows of data corresponding to the indices given in newX. if xcol is None, then newX is interpreted as (fractional) row indices.
            Otherwise, the column specified in xcol is thresholded with the values given in newX and the resultant row indices used to return the data.
        """
        l=_np_.shape(self.data)[0]
        index=_np_.arange(l)
        if xcol is None:
            xcol=self.setas._get_cols("xcol")
        elif isinstance(xcol,bool) and not xcol:
            xcol=None
        if xcol is not None: # We need to convert newX to row indices
            xfunc=interp1d(self.column(xcol), index, kind, 0) # xfunc(x) returns partial index
            newX=xfunc(newX)
        inter=interp1d(index, self.data, kind, 0)
        return inter(newX)

    def lmfit(self,model,xcol=None,ycol=None,p0=None, sigma=None, **kargs):
        """Wrapper around lmfit module fitting.

        Args:
            model (lmfit.Model): An instance of an lmfit.Model that represents the model to be fitted to the data
            xcol (index or None): Columns to be used for the x  data for the fitting. If not givem defaults to the :py:attr:`Stoner.Core.DataFile.setas` x column
            ycol (index or None): Columns to be used for the  y data for the fitting. If not givem defaults to the :py:attr:`Stoner.Core.DataFile.setas` y column

        Keyword Arguments:
            p0 (list, tuple or array): A vector of initial parameter values to try
            sigma (index): The index of the column with the y-error bars
            bounds (callable) A callable object that evaluates true if a row is to be included. Should be of the form f(x,y)
            result (bool): Determines whether the fitted data should be added into the DataFile object. If result is True then
                the last column will be used. If result is a string or an integer then it is used as a column index.
                Default to None for not adding fitted data
            replace (bool): Inidcatesa whether the fitted data replaces existing data or is inserted as a new column (default False)
            header (string or None): If this is a string then it is used as the name of the fitted data. (default None)
            scale_covar (bool) : whether to automatically scale covariance matrix (leastsq only)
            output (str, default "fit"): Specifiy what to return.

        Returns:
            The lmfit module will refurn an instance of the :py:class:`lmfit.models.ModelFit` class that contains all
            relevant information about the fit.
            The return value is determined by the *output* parameter. Options are
                * "ffit"    just the :py:class:`lmfit.model.ModelFit` instance
                * "row"     just a one dimensional numpy array of the fit paraeters interleaved with their uncertainties
                * "full"    a tuple of the fit instance and the row.

        See Also:
            :py:meth:`AnalyseFile.curve_fit`
        """

        bounds=kargs.pop("bounds",lambda x, y: True)
        result=kargs.pop("result",None)
        replace=kargs.pop("replace",False)
        header=kargs.pop("header",None)
        # Support both absolute_sigma and scale_covar, but scale_covar wins here (c.f.curve_fit)
        absolute_sigma=kargs.pop("absolute_sigma",True)
        scale_covar=kargs.pop("scale_covar",not absolute_sigma)
        #Support both asrow and output, the latter wins if both supplied
        asrow=kargs.pop("asrow",False)
        output=kargs.pop("output","row" if asrow else "fit")

        if not isinstance(model,Model):
            raise TypeError("model parameter must be an instance of lmfit.model/Model!")
        if xcol is None or ycol is None:
            cols=self.setas._get_cols()
            if xcol is None:
                xcol=cols["xcol"]
            if ycol is None:
                ycol=cols["ycol"][0]
        working=self.search(xcol, bounds)
        working=ma.mask_rowcols(working,axis=0)

        xdata=working[:,self.find_col(xcol)]
        ydata=working[:,self.find_col(ycol)]
        if p0 is not None:
            if isinstance(p0,(list,tuple,_np_.ndarray)):
                p0={p:pv for p,pv in zip(model.param_names,p0)}
            if not isinstance(p0,dict):
                raise RuntimeError("p0 should have been a tuple, list, ndarray or dict")
                p0.update(kargs)
        else:
            p0=kargs

        if sigma is not None:
            if isinstance(sigma,index_types):
                sigma=working[:,self.find_col(sigma)]
            elif isinstance(sigma,(list,tuple,_np_.ndarray)):
                sigma=_np_.ndarray(sigma)
            else:
                raise RuntimeError("Sigma should have been a column index or list of values")
        else:
            sigma=_np_.ones(len(xdata))
            scale_covar=True
        xvar=model.independent_vars[0]
        p0[xvar]=xdata

        fit=model.fit(ydata,None,scale_covar=scale_covar,weights=1.0/sigma,**p0)
        if fit.success:
            row=[]
            if isinstance(result,index_types) or (isinstance(result,bool) and result):
                self.add_column(fit.best_fit,column_header=header,index=result,replace=replace)
            elif result is not None:
                raise RuntimeError("Didn't recognize result as an index type or True")
            for p in fit.params:
                self[p]=fit.params[p].value
                self[p+"_err"]=fit.params[p].stderr
                row.append([fit.params[p].value,fit.params[p].stderr])
            self["chi^2"]=fit.chisqr
            self["nfev"]=fit.nfev
            retval={"fit":fit,"row":row,"full":(fit,row)}
            if output not in retval:
                raise RuntimeError("Failed to recognise output format:{}".format(output))
            else:
                return retval[output]
        else:
            raise RuntimeError("Failed to complete fit. Error was:\n{}\n{}".format(fit.lmdif_message,fit.message))

    def make_bins(self,xcol,bins,mode,**kargs):
        """Utility method to generate bin boundaries and centres along an axis.

        Args:
            xcol (index): Column of data with X values
            bins (int or float): Number of bins (int) or width of bins (if float)
            mode (string): "lin" for linear binning, "log" for logarithmic binning.

        Keyword Arguments:
            bin_start (float): Override minimum bin value
            bin_stop (float): Override the maximum bin value

        Returns:
            bin_start,bin_stop,bin_centres (1D arrays): The locations of the bin
                boundaries and centres for each bin.
        """
        (xmin,xmax)=self.span(xcol)
        if "bin_start" in kargs:
            xmin=kargs["bin_start"]
        if "bin_stop" in kargs:
            xmax=kargs["bin_stop"]
        if isinstance(bins,int): # Given a number of bins
            if mode.lower()=="lin":
                bin_width=float(xmax-xmin)/bins
                bin_start=linspace(xmin,xmax-bin_width,bins)
                bin_stop=linspace(xmin+bin_width,xmax,bins)
                bin_centres=(bin_start+bin_stop)/2.0
            elif mode.lowerlower()=="log":
                xminl=_np_.log(xmin)
                xmaxl=_np_.log(xmax)
                bin_width=float(xmaxl-xminl)/bins
                bin_start=linspace(xminl,xmaxl-bin_width,bins)
                bin_stop=linspace(xminl+bin_width,xmaxl,bins)
                bin_centres=(bin_start+bin_stop)/2.0
                bin_start=_np_.exp(bin_start)
                bin_stop=_np_.exp(bin_stop)
                bin_centres=_np_.exp(bin_centres)
        elif isinstance(bins,float): # Given a bin with as a flot
            if mode.lower()=="lin":
                bin_width=bins
                bins=_np_.ceil(float(xmin-xmax)/bins)
                bin_start=linspace(xmin,xmax-bin_width,bins)
                bin_stop=linspace(xmin+bin_width,xmax,bins)
                bin_centres=(bin_start+bin_stop)/2.0
            elif mode.lower()=="log":
                if not 0.0<bins<=1.0:
                    raise ValueError("Bin width must be between 0 ans 1 for log binning")
                xp=xmin
                splits=[]
                centers=[]
                while xp<xmax:
                    splits.append(xp)
                    centers.append(xp*(1+bins/2))
                    xp=xp*(1+bins)
                splits.append(xmax)
                bin_start=_np_.array(splits[:-1])
                bin_stop=_np_.array(splits[1:])
                bin_centres=_np_.array(centers)
        if len(bin_start)>len(self):
            raise ValueError("Attempting to bin into more bins than there is data.")
        return bin_start,bin_stop,bin_centres



    def max(self, column=None, bounds=None):
        """FInd maximum value and index in a column of data.

        Args:
            column (index): Column to look for the maximum in

        Keyword Arguments:
            bounds (callable): A callable function that takes a single argument list of
                numbers representing one row, and returns True for all rows to search in.

        Returns:
            (maximum value,row index of max value)

        Note:
            If column is not defined (or is None) the :py:attr:`DataFile.setas` column
            assignments are used.
        """
        if column is None:
            col=self.setas._get_cols("ycol")
        else:
            col=self.find_col(column)
        if bounds is not None:
            self._push_mask()
            self._set_mask(bounds, True, col)
        result=self.data[:, col].max(), self.data[:, col].argmax()
        if bounds is not None:
            self._pop_mask()
        return result


    def mean(self, column=None, bounds=None):
        """FInd mean value of a data column.

        Args:
            column (index): Column to look for the maximum in

        Keyword Arguments:
            bounds (callable): A callable function that takes a single argument list of
                numbers representing one row, and returns True for all rows to search in.

        Returns:
            The mean of the data.

        Note:
            If column is not defined (or is None) the :py:attr:`DataFile.setas` column
            assignments are used.

        .. todo::
            Fix the row index when the bounds function is used - see note of \b max
        """
        if column is None:
            col=self.setas._get_cols("ycol")
        else:
            col=self.find_col(column)
        if bounds is not None:
            self._push_mask()
            self._set_mask(bounds, True, col)
        result=self.data[:, col].mean()
        if bounds is not None:
            self._pop_mask()
        return result


    def min(self, column=None, bounds=None):
        """FInd minimum value and index in a column of data.

        Args:
            column (index): Column to look for the maximum in

        Keyword Arguments:
            bounds (callable): A callable function that takes a single argument list of
                numbers representing one row, and returns True for all rows to search in.

        Returns:
            (minimum value,row index of min value)

        Note:
            If column is not defined (or is None) the :py:attr:`DataFile.setas` column
            assignments are used.

                """
        if column is None:
            col=self.setas._get_cols("ycol")
        else:
            col=self.find_col(column)
        if bounds is not None:
            self._push_mask()
            self._set_mask(bounds, True, col)
        result=self.data[:, col].min(), self.data[:, col].argmin()
        if bounds is not None:
            self._pop_mask()
        return result


    def mpfit(self, func,  xcol, ycol, p_info,  func_args=dict(), sigma=None, bounds=lambda x, y: True, **mpfit_kargs ):
        """Runs the mpfit algorithm to do a curve fitting with constrined bounds etc.

                mpfit(func, xcol, ycol, p_info, func_args=dict(),sigma=None,bounds=labdax,y:True,**mpfit_kargs)

        Args:
            func (callable): Fitting function def func(x,parameters, **func_args)
            xcol, ycol (index): index the x and y data sets
            p_info (list of dictionaries): Defines the fitting parameters

        Keyword Arguments:
            sigma (index): weights of the data poiints. If not specified, then equal weighting assumed
            bounds (callable): function that takes x,y pairs and returns true if to be used in the fitting
            **mpfit_kargs: other lkeywords passed straight to mpfit

        Returns:
            Best fit parameters
        """
        from .mpfit import mpfit
        if sigma==None:
            working=self.search(xcol, bounds, [xcol, ycol])
            x=working[:, 0]
            y=working[:, 1]
            sigma=_np_.ones(_np_.shape(y), _np_.float64)
        else:
            working=self.search(xcol, bounds, [xcol, ycol, sigma])
            x=working[:, 0]
            y=working[:, 1]
            sigma=working[:, 2]
        func_args["x"]=x
        func_args["y"]=y
        func_args["err"]=sigma
        func_args["func"]=func
        m = mpfit(self.__mpf_fn, parinfo=p_info,functkw=func_args, **mpfit_kargs)
        return m


    def mpfit_iterfunct(self, myfunct, p, iterator, fnorm, functkw=None,parinfo=None, quiet=0, dof=None):
        """Function that is called on every iteration of the non-linerar fitting.

        Args:
            myfunct (callable): Function being modelled
            iteration (int): Iteration number
            fnorm (list): ?
            functkw (dictionary): Keywords being passed to the user function
            parinfo (list of dicts): PArameter informatuion
            quiet (int): 0 to suppress output
            dof (float): Figure of merit ?

        Note:
            This functionb just prints a full stop for every iteration.

        """
        sys.stdout.write('.')
        sys.stdout.flush()


    def multiply(self, a, b, replace=False, header=None):
        """Multiply one column (a) by  another column, number or array (b).

        Args:
            a (index): First column to work with
            b (index, float or 1D array):  Second column to work with.

        Keyword Arguments:
            header (string or None): new column header  (defaults to a-b
            replace (bool): Replace the a column with the new data

        Returns:
            A copy of the new data object

        If a and b are tuples of length two, then the firstelement is assumed to be the value and
        the second element an uncertainty in the value. The uncertainties will then be propagated and an
        additional column with the uncertainites will be added to the data."""
        a=self.find_col(a)
        if isinstance(a,tuple) and isinstance(b,tuple) and len(a)==2 and len(b)==2: #Error columns on
            (a,e1)=a
            (b,e2)=b
            e1data=self.__get_math_val(e1)[0]
            e2data=self.__get_math_val(e2)[0]
            err_header=None
            err_calc=lambda adata,bdata,e1data,e2data: _np_.sqrt((e1data/adata)**2+(e2data/bdata)**2)*adata*bdata
        else:
            err_calc=None
        adata,aname=self.__get_math_val(a)
        bdata,bname=self.__get_math_val(b)
        if isinstance(header,tuple) and len(header)==0:
            header,err_header=header
        if header is None:
            header="{}*{}".format(aname,bname)
        if err_calc is not None and err_header is None:
            err_header="Error in "+header
        if err_calc is not None:
            err_data=err_calc(adata,bdata,e1data,e2data)
        self.add_column((adata*bdata), header, a, replace=replace)
        if err_calc is not None:
            self.add_column(err_data,err_header,a+1,replace=False)
        return self


    def nlfit(self, ini_file, func):
        """Non-linear fitting using the :py:mod:`Stoner.nlfit` module.

        Args:
            ini_file (string): path to ini file with model
            func (string or callable):Name of function to fit with (as seen in FittingFuncs.py module in Stoner)
                    or function instance to fit with

        Returns:
            AnalyseFile instance, matplotlib.fig instance (or None if plotting disabled in the inifile)
        """
        return Stoner.nlfit.nlfit(ini_file, func, data=self)


    def normalise(self, target, base, replace=True, header=None):
        """Normalise data columns by dividing through by a base column value.

        Args:
            target (index): One or more target columns to normalise can be a string, integer or list of strings or integers.
            base (index): The column to normalise to, can be an integer or string

        Keyword Arguments:
            replace (bool): Set True(default) to overwrite  the target data columns
            header (string or None): The new column header - default is target name(norm)

        Returns:
            A copy of the current object

        If a and b are tuples of length two, then the firstelement is assumed to be the value and
        the second element an uncertainty in the value. The uncertainties will then be propagated and an
        additional column with the uncertainites will be added to the data."""

        if not isinstance(target, list):
            target=[self.find_col(target)]
        for t in target:
            if header is None:
                header=self.column_headers[self.find_col(t)]+"(norm)"
            else:
                header=str(header)
            self.divide(t,base,header=header,replace=replace)
        return self


    def peaks(self, ycol=None, width=None, significance=None , xcol=None, peaks=True, troughs=False, poly=2,  sort=False):
        """Locates peaks and/or troughs in a column of data by using SG-differentiation.

        Args:
            ycol (index): is the column name or index of the data in which to search for peaks
            width (float): is the expected minium halalf-width of a peak in terms of the number of data points.
                This is used in the differnetiation code to find local maxima. Bigger equals less sensitive
                to experimental noise, smaller means better eable to see sharp peaks
            poly (int): This is the order of polynomial to use when differentiating the data to locate a peak. Must >=2, higher numbers
                will find sharper peaks more accurately but at the risk of finding more false positives.

        Keyword Arguments:
            significance (float): is used to decide whether a local maxmima is a significant peak. Essentially just the curvature
                of the data. Bigger means less sensistive, smaller means more likely to detect noise.
            xcol (index or None): name or index of data column that p[rovides the x-coordinate (default None)
            peaks (bool): select whether to measure peaks in data (default True)
            troughs (bool): select whether to measure troughs in data (default False)
            sort (bool): Sor the results by significance of peak

        Returns:
            If xcol is None then returns conplete rows of data corresponding to the found peaks/troughs. If xcol is not none, returns a 1D array of the x positions of the peaks/troughs.
            """

        if ycol is None:
            ycol=self.setas._get_cols("ycol")
            xcol=self.setas._get_cols("xcol")
        if width is None: # Set Width to be length of data/20
            width=len(self)/20
        assert poly>=2,"poly must be at least 2nd order in peaks for checking for significance of peak or trough"
        if significance is None: # Guess the significance based on the range of y and width settings
            dm=self.max(ycol)[0]
            dp=self.min(ycol)[0]
            dm=dm-dp
            significance=0.2*dm/(4*width**2)
        d1=self.SG_Filter(ycol, width, poly, 1)
        i=_np_.arange(len(d1))
        d2=interp1d(i, self.SG_Filter(ycol, width, poly, 2))
        if xcol==None:
            xcol=i
        else:
            xcol=self.column(xcol)
        index=interp1d(i, xcol)
        w=abs(xcol[0]-xcol[width]) # Approximate width of our search peak in xcol
        z=_np_.array(self.__threshold(0, d1, rising=troughs, falling=peaks))
        z=[zv for zv in z if zv>w/2.0 and zv<max(xcol)-w/2.0] #Throw out peaks or troughts too near the ends
        if sort:
            z=_np_.take(z, _np_.argsort(d2(z)))
        return index([x for x in z if _np_.abs(d2(x))>significance])


    def polyfit(self,xcol=None,ycol=None,polynomial_order=2, bounds=lambda x, y:True, result=None,replace=False,header=None):
        """ Pass through to numpy.polyfit.

            Args:
                xcol (index): Index to the column in the data with the X data in it
                ycol (index): Index to the column int he data with the Y data in it
                polynomial_order: Order of polynomial to fit (default 2)
                bounds (callable): A function that evaluates True if the current row should be included in the fit
                result (index or None): Add the fitted data to the current data object in a new column (default don't add)
                replace (bool): Overwrite or insert new data if result is not None (default False)
                header (string or None): Name of column_header of replacement data. Default is construct a string from the y column headser and polynomial order.

            Returns:
                The best fit polynomial as a numpy.poly object.

            Note:
                If the x or y columns are not specified (or are None) the the setas attribute is used instead.

                This method is depricated and may be removed in a future version in favour of the more general curve_fit
            """
        from Stoner.Util import ordinal

        if None in (xcol,ycol):
            cols=self.setas._get_cols()
            if xcol is None:
                xcol=cols["xcol"]
            if ycol is None:
                ycol=cols["ycol"][0]

        working=self.search(xcol, bounds)
        p= _np_.polyfit(working[:, self.find_col(xcol)],working[:, self.find_col(ycol)],polynomial_order)
        if result is not None:
            if header is None:
                header="Fitted {} with {} order polynomial".format(self.column_headers[self.find_col(ycol)],ordinal(polynomial_order))
            self.add_column(_np_.polyval(p, self.column(xcol)), index=result, replace=replace, column_header=header)
        return p


    def span(self, column=None, bounds=None):
        """Returns a tuple of the maximum and minumum values within the given column and bounds by calling into :py:meth:`AnalyseFile.max` and :py:meth:`AnalyseFile.min`.

        Args:
            column (index): Column to look for the maximum in

        Keyword Arguments:
            bounds (callable): A callable function that takes a single argument list of
                numbers representing one row, and returns True for all rows to search in.

        Returns:
            A tuple of (min value, max value)

        Note:
            If column is not defined (or is None) the :py:attr:`DataFile.setas` column
            assignments are used.

        """
        return (self.min(column, bounds)[0], self.max(column, bounds)[0])


    def split(self,xcol=None,func=None):
        """Splits the current :py:class:`AnalyseFile` object into multiple :py:class:`AnalyseFile` objects where each one contains the rows
        from the original object which had the same value of a given column.

        Args:
            xcol (index): The index of the column to look for values in.
                This can be a list in which case a :py:class:`Stoner.Folders.DataFolder` with groups
                with subfiles is built up by applying each item in the xcol list recursively.
            func (callable):  Function that can be evaluated to find the value to determine which output object
                each row belongs in. If this is left as the default None then the column value is converted
                to a string and that is used.

        Returns:
            A :py:class:`Stoner.Folders.DataFolder` object containing the individual :py:class:`AnalyseFile` objects

        Note:
            The function to be of the form f(x,r) where x is a single float value and r is a list of floats representing
            the complete row. The return value should be a hashable value. func can also be a list if xcol is a list,
            in which the func values are used along with the @a xcol values.
        """
        from Stoner.Folders import DataFolder
        if xcol is None:
            xcol=self.setas._get_cols("xcol")
        else:
            xcol=self.find_col(xcol)
        out=DataFolder(nolist=True)
        files=dict()
        morecols=[]
        morefuncs=None
        if isinstance(xcol,list) and len(xcol)<=1:
            xcol=xcol[0]
        elif isinstance(xcol,list):
            morecols=xcol[1:]
            xcol=xcol[0]
        if isinstance(func,list) and len(func)<=1:
            func=func[0]
        elif isinstance(func,list):
            morefuncs=func[1:]
            func=func[0]
        if func is None:
            for val in _np_.unique(self.column(xcol)):
                files[str(val)]=self.clone
                files[str(val)].data=self.search(xcol,val)
        else:
            xcol=self.find_col(xcol)
            for r in self.rows():
                x=r[xcol]
                key=func(x,r)
                if key not in files:
                    files[key]=self.clone
                    files[key].data=_np_.array([r])
                else:
                    files[key]=files[key]+r
        for k in files:
            files[k].filename="{}={}".format(xcol,k)
        if len(morecols)>0:
            for k in sorted(list(files.keys())):
                out.groups[k]=files[k].split(morecols,morefuncs)
        else:
            out.files=[files[k] for k in sorted(list(files.keys()))]
        return out


    def stitch(self,other,xcol=None,ycol=None,overlap=None,min_overlap=0.0,mode="All",func=None,p0=None):
        """Apply a scaling to this data set to make it stich to another dataset.

        Args:
            other (DataFile): Another data set that is used as the base to stitch this one on to
            xcol,ycol (index or None): The x and y data columns. If left as None then the current setas attribute is used.

        Keyword Arguments:
            overlap (tuple of (lower,higher) or None): The band of x values that are used in both data sets to match, if left as None, thenthe common overlap of the x data is used.
            min_overlap (float): If you know that overlap must be bigger than a certain amount, the bounds between the two data sets needs to be adjusted. In this case min_overlap shifts the boundary of the overlap on this DataFile.
            mode (str): Unless *func* is specified, controls which parameters are actually variable, defaults to all of them.
            func (callable): a stitching function that transforms :math:`(x,y)\\rightarrow(x',y')`. Default is to use functions defined by *mode*()
            p0 (iterable): if func is not None then p0 should be the starting values for the stitching function parameters

        Returns:
            A copy of the current AnalyseFile with the x and y data columns adjusted to stitch

        To stitch the data together, the x and y data in the current data file is transforms so that
        :math:`x'=x+A` and :math:`y'=By+C` where :math:`A,B,C` are constants and :math:`(x',y')` are close matches to the
        :math:`(x,y)` data in *other*. The algorithm assumes that the overlap region contains equal
        numbers of :math:`(x,y)` points *mode controls whether A,B, and C are fixed or adjustable

        * "All" - all three parameters adjustable
        * "Scale y, shift x" - C is fixed at 0.0
        * "Scale and shift y" A is fixed at 0.0
        * "Scale y" - only B is adjustable
        * "Shift y" - Only c is adjsutable
        * "Shift x" - Only A is adjustable
        * "Shift both" - B is fixed at 1.0
        .
        """
        if xcol is None: #Sort out the xcolumn and y column indexes
            xcol=self.setas._get_cols("xcol")
        else:
            xcol=self.find_col(xcol)
        if ycol is None:
            ycol=self.setas._get_cols("ycol")
        else:
            ycol=self.find_col(ycol)
        x=self.column(xcol)+min_overlap # Get the (x,y) data from each data set to be stitched
        y=self.column(ycol)
        xp=other.column(xcol)
        yp=other.column(ycol)
        if overlap is None: # Now sort out the overlap region
            lower=max(_np_.min(x),_np_.min(xp))
            upper=min(_np_.max(x),_np_.max(xp))
        else:
            lower=min(overlap)
            upper=max(overlap)
        # Now func is a function (x,y,p1,p2,p3)
        #and p0 is vector of the correct length
        inrange=_np_.logical_and(x>=lower,x<=upper)
        inrange_other=_np_.logical_and(xp>=lower,xp<=upper)
        x=x[inrange] # And throw away data that isn't in the overlap
        y=y[inrange]
        xp=xp[inrange_other]
        yp=yp[inrange_other]
        if func is None:
            opts={"all":(lambda x,y,A,B,C:(x+A,y*B+C)),
                  "scale y and shift x":(lambda x,y,A,B:(x+A,B*y)),
                  "scale and shift y":(lambda x,y,B,C:(x,y*B+C)),
                  "scale y":(lambda x,y,B:(x,y*B)),
                  "shift y":(lambda x,y,C:(x,y+C)),
                  "shift both":(lambda x,y,A,C:(x+A,y+C))}
            defaults={"all":[1,2,3],
                  "scale y,shift x":[1,2],
                  "scale and shift y":[2,3],
                  "scale y":[2],
                  "shift y": [3],
                  "shift both": [1,3]}
            A0=_np_.mean(xp)-_np_.mean(x)
            C0=_np_.mean(yp)-_np_.mean(y)
            B0=(_np_.max(yp)-_np_.min(yp))/(_np_.max(y)-_np_.min(y))
            p=_np_.array([0,A0,B0,C0])
            assert isinstance(mode,string_types),"mode keyword should be a string if func is not defined"
            mode=mode.lower()
            assert mode in opts,"mode keyword should be one of {}".format(opts.keys)
            func=opts[mode]
            p0=p[defaults[mode]]
        else:
            assert callable(func),"Keyword func should be callable if given"
            (args,varargs,keywords,defaults)=getargspec(func)
            assert isinstance(p0,Iterable),"Keyword parameter p0 shoiuld be iterable if keyword func is given"
            assert len(p0)==len(args)-2,"Keyword p0 should be the same length as the optional arguments to func"
        # This is a bit of a hack, we turn (x,y) points into a 1D array of x and then y data
        set1=_np_.append(x,y)
        set2=_np_.append(xp,yp)
        assert len(set1)==len(set2),"The number of points in the overlap are different in the two data sets"

        def transform(set1,*p):
            m=len(set1)/2
            x=set1[:m]
            y=set1[m:]
            tmp=func(x,y,*p)
            out=_np_.append(tmp[0],tmp[1])
            return out
        popt,pcov=curve_fit(transform,set1,set2,p0=p0) # Curve fit for optimal A,B,C
        perr=_np_.sqrt(_np_.diagonal(pcov))

        (self.data[:,xcol],self.data[:,ycol])=func(self.data[:,xcol]+min_overlap,self.data[:,ycol],*popt)
        self["Stitching Coefficients"]=list(popt)
        self["Stitching Coeffient Errors"]=list(perr)
        return self

    def subtract(self, a, b, replace=False, header=None):
        """Subtract one column, number or array (b) from another column (a).

        Args:
            a (index): First column to work with
            b (index, float or 1D array):  Second column to work with.

        Keyword Arguments:
            header (string or None): new column header  (defaults to a-b
            replace (bool): Replace the a column with the new data

        Returns:
            A copy of the new data object

        If a and b are tuples of length two, then the firstelement is assumed to be the value and
        the second element an uncertainty in the value. The uncertainties will then be propagated and an
        additional column with the uncertainites will be added to the data."""

        if isinstance(a,tuple) and isinstance(b,tuple) and len(a)==2 and len(b)==2: #Error columns on
            (a,e1)=a
            (b,e2)=b
            e1data=self.__get_math_val(e1)[0]
            e2data=self.__get_math_val(e2)[0]
            err_header=None
            err_calc=lambda adata,bdata,e1data,e2data: _np_.sqrt(e1data**2+e2data**2)
        else:
            err_calc=None
        adata,aname=self.__get_math_val(a)
        bdata,bname=self.__get_math_val(b)
        if isinstance(header,tuple) and len(header)==0:
            header,err_header=header
        if header is None:
            header="{}-{}".format(aname,bname)
        if err_calc is not None and err_header is None:
            err_header="Error in "+header
        if err_calc is not None:
            err_data=err_calc(adata,bdata,e1data,e2data)
        self.add_column((adata-bdata), header, a, replace=replace)
        if err_calc is not None:
            a=self.find_col(a)
            self.add_column(err_data,err_header,a,replace=False)
        return self


    def threshold(self, threshold,col=None, rising=True, falling=False,xcol=None,all_vals=False,transpose=False):
        """Finds partial indices where the data in column passes the threshold, rising or falling.

        Args:
            col (index): Column index to look for data in
            threshold (float): Value to look for in column col

        Keyword Arguments:
            rising (bool):  look for case where the data is increasing in value (defaukt True)
            falling (bool): look for case where data is fallinh in value (default False)
            xcol (index or None): rather than returning a fractional row index, return the
                interpolated value in column xcol
                all_vals (bool): return all crossing points of the threshold or just the first. (default False)
            transpose (bbool): Swap the x and y columns around - this is most useful when the column assignments
                have been done via the setas attribute
            all_vals (bool): Return all values that match the criteria, or just the first in the file.

        Returns:
            Either a sing;le fractional row index, or an in terpolated x value

        Note:
            If you don't sepcify a col value or set it to None, then the assigned columns via the
            :py:attr:`DataFile.setas` attribute will be used.

        Warning:
            There has been an API change. Versions prior to 0.1.9 placed the column before the threshold in the positional
            argument list. In order to support the use of assigned columns, this has been swapped to the present order.
            """
        if col is None:
            col=self.setas._get_cols("ycol")
            xcol=self.setas._get_cols("xcol")

        current=self.column(col)
        ret=[]
        if isinstance(threshold, (list,_np_.ndarray)):
            if all_vals:
                ret=[self.__threshold(x, current, rising=rising, falling=falling) for x in threshold]
            else:
                ret=[self.__threshold(x, current, rising=rising, falling=falling)[0] for x in threshold]
        else:
            if all_vals:
                ret=self.__threshold(threshold, current, rising=rising, falling=falling)
            else:
                ret=[self.__threshold(threshold, current, rising=rising, falling=falling)[0]]
        if xcol is not None:
            ret=[self.interpolate(r,xcol=False)[self.find_col(xcol)] for r in ret]
        if all_vals:
            return ret
        else:
            return ret[0]
