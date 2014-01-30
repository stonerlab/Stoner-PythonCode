"""Stoner .Analysis provides a subclass of :py:class:`Stoner.Core.DataFile` that has extra analysis routines builtin.

Provides  :py:class:`AnalyseFile` - DataFile with extra bells and whistles.
"""
from Stoner.compat import *
from Stoner.Core import DataFile
import Stoner.FittingFuncs
import Stoner.nlfit
import numpy
import numpy.ma as ma
from scipy.integrate import cumtrapz
import math
import sys
import re

def cov2corr(M):
    """ Converts a covariance matrix to a correlation matrix. Taken from bvp.utils.misc

    Args:
        M (2D numpy.array): Co-varriance Matric

    Returns:
        Correlation Matrix.
    """
    if (not isinstance(M, numpy.ndarray)) or (not (len(M.shape) == 2)) or (not(M.shape[0] == M.shape[1])):
        raise ValueError('cov2corr expects a square ndarray, got %s' % M)

    if numpy.isnan(M).any():
        raise ValueError('Found NaNs in my covariance matrix: %s' % M)

    # TODO check Nan and positive diagonal
    d = M.diagonal()
    if (d < 0).any():
        raise ValueError('Expected positive elements for square matrix, got diag = %s' % d)

    n = M.shape[0]
    R = numpy.ndarray((n, n))
    for i in range(n):
        for j in range(n):
            d = M[i, j] / math.sqrt(M[i, i] * M[j, j])
            R[i, j] = d

    return R

class AnalyseFile(DataFile):
    """:py:class:`Stoner.Analysis.AnalyseFile` extends :py:class:`Stoner.Core.DataFile` with numpy and scipy passthrough functions.

    Note:
        There is no separate constructor for this class - it inherits from DataFile

    """

    def __dir__(self):
        """Handles the local attributes as well as the inherited ones"""
        attr=dir(type(self))
        attr.extend(super(AnalyseFile,self).__dir__())
        attr.extend(list(self.__dict__.keys()))
        attr=list(set(attr))
        return sorted(attr)


    def __SG_calc_coeff(self, num_points, pol_degree=1, diff_order=0):
        """ calculates filter coefficients for symmetric savitzky-golay filter.
            see: http://www.nrbook.com/a/bookcpdf/c14-8.pdf

        ArgsL
            num_points (int): Number of points to use in filter
            poll_degree (int): Order of polynomial to use in the filter - defaults to linear
            diff_order (int): Order of differential to find - defaults to 0 which is just a smoothing operation

        Returns:
            A 1D array to convolve with the data

        Note:
            num_points   means that 2*num_points+1 values contribute to the smoother.
        """

        # setup interpolation matrix
        # ... you might use other interpolation points
        # and maybe other functions than monomials ....

        x = numpy.arange(-num_points, num_points+1, dtype=int)

        A = numpy.zeros((2*num_points+1, pol_degree+1), float)
        for i in range(2*num_points+1):
            for j in range(pol_degree+1):
                A[i,j] = x[i]**j

        # calculate diff_order-th row of inv(A^T A)
        ATA = numpy.dot(A.transpose(), A)
        rhs = numpy.zeros((pol_degree+1,), float)
        rhs[diff_order] = (-1)**diff_order
        wvec = numpy.linalg.solve(ATA, rhs)

        # calculate filter-coefficients
        coeff = numpy.dot(A, wvec)

        return coeff

    def __SG_smooth(self, signal, coeff):
        """ applies coefficients calculated by calc_coeff() to signal

        Really just a pass through to numpy convolve
        """

        N = numpy.size(coeff-1)/2
        res = numpy.convolve(signal, coeff)
        return res[N:-N]

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
        previous=numpy.roll(current, 1)
        index=numpy.arange(len(current))
        sdat=numpy.column_stack((index, current, previous))
        if rising==True and falling==False:
            expr=lambda x:(x[1]>=threshold) & (x[2]<threshold)
        elif rising==True and falling==True:
            expr=lambda x:((x[1]>=threshold) & (x[2]<threshold)) | ((x[1]<=threshold) & (x[2]>threshold))
        elif rising==False and falling==True:
            expr=lambda x:(x[1]<=threshold) & (x[2]>threshold)
        else:
            expr=lambda x:False
        intr=lambda x:x[0]-1+(x[1]-threshold)/(x[1]-x[2])
        return numpy.array([intr(x) for x in sdat if expr(x) and intr(x)>0])

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
        elif isinstance(col,numpy.ndarray) and len(col.shape)==1 and len(col)==len(self):
            data=col
            name="data"
        elif isinstance(col,float):
            data=col*numpy.ones(len(self))
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

    def mpfit_iterfunct(self, myfunct, p, iterator, fnorm, functkw=None,parinfo=None, quiet=0, dof=None):
        """Function that is called on every iteration of the non-linerar fitting

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


    def polyfit(self,column_x,column_y,polynomial_order, bounds=lambda x, y:True, result=None,replace=False,header=None):
        """ Pass through to numpy.polyfit

            Args:
                column_x (index): Index to the column in the data with the X data in it
                column_y (index): Index to the column int he data with the Y data in it
                polynomial_order: Order of polynomial to fit
                bounds (callable): A function that evaluates True if the current row should be included in the fit
                result (index or None): Add the fitted data to the current data object in a new column (default don't add)
                replace (bool): Overwrite or insert new data if result is not None (default False)
                header (string or None): Name of column_header of replacement data. Default is construct a string from the y column headser and polynomial order.

            Returns:
                The best fit polynomial as a numpy.poly object.
            """
        from Stoner.Util import ordinal
        working=self.search(column_x, bounds)
        p= numpy.polyfit(working[:, self.find_col(column_x)],working[:, self.find_col(column_y)],polynomial_order)
        if result is not None:
            if header is None:
                header="Fitted {} with {} order polynomial".format(self.column_headers[self.find_col(column_y)],ordinal(polynomial_order))
            self.add_column(numpy.polyval(p, self.column(column_x)), index=result, replace=replace, column_header=header)
        return p

    def curve_fit(self, func,  xcol, ycol, p0=None, sigma=None, bounds=lambda x, y: True, result=None, replace=False, header=None ,asrow=False):
        """General curve fitting function passed through from scipy

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
            asrow (bool): Instead of returning popt,pcov, return a single array of popt, interleaved with the standard error in popt

        Returns:
            popt (array): Optimal values of the fitting parameters p
            pcov (2d array): The variance-co-variance matrix for the fitting parameters.
            If asrow is True, then return [popt[0],sqrt(pcov[0,0]),popt[1],sqrt(pcov[1,1])...popt[n],sqrt(pcov[n,n])]

        Note:
            The fitting function should have prototype y=f(x,p[0],p[1],p[2]...)
            The x-column and y-column can be anything that :py:meth:`Stoner.Core.DataFile.find_col` can use as an index
            but typucally either strings to be matched against column headings or integers.
            The initial parameter values and weightings default to None which corresponds to all parameters starting
            at 1 and all points equally weighted. The bounds function has format b(x, y-vec) and rewturns true if the
            point is to be used in the fit and false if not.
        """
        from scipy.optimize import curve_fit
        from inspect import getargspec

        working=self.search(xcol, bounds)
        working=ma.mask_rowcols(working,axis=0)
        if sigma is not None:
            sigma=working[:,self.find_col(sigma)]
        xdat=working[:,self.find_col(xcol)]
        ydat=working[:,self.find_col(ycol)]
        popt, pcov=curve_fit(func,  xdat,ydat, p0, sigma)
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
        if not asrow:
            return popt, pcov
        else:
            rc=numpy.array([])
            for i in range(len(popt)):
                rc=numpy.append(rc,[popt[i],numpy.sqrt(pcov[i,i])])
            return rc
    def max(self, column, bounds=None):
        """FInd maximum value and index in a column of data

        Args:
            column (index): Column to look for the maximum in

        Keyword Arguments:
            bounds (callable): A callable function that takes a single argument list of
                numbers representing one row, and returns True for all rows to search in.

        Returns:
            (maximum value,row index of max value)
        """
        col=self.find_col(column)
        if bounds is not None:
            self._push_mask()
            self._set_mask(bounds, True, col)
        result=self.data[:, col].max(), self.data[:, col].argmax()
        if bounds is not None:
            self._pop_mask()
        return result

    def min(self, column, bounds=None):
        """FInd minimum value and index in a column of data

        Args:
            column (index): Column to look for the maximum in

        Keyword Arguments:
            bounds (callable): A callable function that takes a single argument list of
                numbers representing one row, and returns True for all rows to search in.

        Returns:
            (minimum value,row index of min value)
                """
        col=self.find_col(column)
        if bounds is not None:
            self._push_mask()
            self._set_mask(bounds, True, col)
        result=self.data[:, col].min(), self.data[:, col].argmin()
        if bounds is not None:
            self._pop_mask()
        return result

    def span(self, column, bounds=None):
        """Returns a tuple of the maximum and minumum values within the given column and bounds by calling into \b AnalyseFile.max and \b AnalyseFile.min

        Args:
            column (index): Column to look for the maximum in

        Keyword Arguments:
            bounds (callable): A callable function that takes a single argument list of
                numbers representing one row, and returns True for all rows to search in.

        Returns:
            A tuple of (min value, max value)
        """
        return (self.min(column, bounds)[0], self.max(column, bounds)[0])

    def clip(self, column, clipper):
        """Clips the data based on the column and the clipper value

        Args:
            column (index): Column to look for the maximum in
            clipper (tuple or array): Either a tuple of (min,max) or a numpy.ndarray -
                in which case the max and min values in that array will be
                used as the clip limits
        Returns:
            This instance."
        @param column Column to look for the clipping value in
        @param clipper
        """
        clipper=(min(clipper), max(clipper))
        self=self.del_rows(column, lambda x, y:x<clipper[0] or x>clipper[1])
        return self



    def mean(self, column, bounds=None):
        """FInd mean value of a data column

        Args:
            column (index): Column to look for the maximum in

        Keyword Arguments:
            bounds (callable): A callable function that takes a single argument list of
                numbers representing one row, and returns True for all rows to search in.

        Returns:
            The mean of the data.

        .. todo::
            Fix the row index when the bounds function is used - see note of \b max
        """
        col=self.find_col(column)
        if bounds is not None:
            self._push_mask()
            self._set_mask(bounds, True, col)
        result=self.data[:, col].mean()
        if bounds is not None:
            self._pop_mask()
        return result

    def diffsum(self, a, b, replace=False, header=None):
        """Subtract one column, number or array (b) from another column (a) and divbdatae by their sums

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
            err_calc=lambda adata,bdata,e1data,e2data: numpy.sqrt((1.0/(adata+bdata)-(adata-bdata)/(adata+bdata)**2)**2*e1data**2+(-1.0/(adata+bdata)-(adata-bdata) / (adata+bdata)**2)**2*e2data**2)
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

    def subtract(self, a, b, replace=False, header=None):
        """Subtract one column, number or array (b) from another column (a)

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
            err_calc=lambda adata,bdata,e1data,e2data: numpy.sqrt(e1data**2+e2data**2)
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
            self.add_column(err_data,err_header,a+1,replace=False)
        return self

    def add(self, a, b, replace=False, header=None):
        """Add one column, number or array (b) to another column (a)

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
            err_calc=lambda adata,bdata,e1data,e2data: numpy.sqrt(e1data**2+e2data**2)
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

    def divide(self, a, b, replace=False, header=None):
        """Divide one column (a) by  another column, number or array (b)

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
            err_calc=lambda adata,bdata,e1data,e2data: numpy.sqrt((e1data/adata)**2+(e2data/bdata)**2)*adata*bdata
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

    def mulitply(self, a, b, replace=False, header=None):
        """Multiply one column (a) by  another column, number or array (b)

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
            err_calc=lambda adata,bdata,e1data,e2data: numpy.sqrt((e1data/adata)**2+(e2data/bdata)**2)*adata*bdata
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

    def apply(self, func, col, replace=True, header=None):
        """Applies the given function to each row in the data set and adds to the data set

        Args:
            func (callable): A function that takes a numpy 1D array representing each row of data
            col (index): The column in which to place the result of the function

        Keyword Arguments:
            replace (bool): Isnert a new data column (False) or replace the data column (True, default)
            header (string or None): The new column header (defaults to the name of the function func

        Returns:
            A copy of the current instance
        """
        col=self.find_col(col)
        nc=numpy.zeros(len(self))
        i=0
        for r in self.rows():
            nc[i]=func(r)
            i+=1
        if header==None:
            header=func.__name__
        if replace!=True:
            self=self.add_column(nc, header, col)
        else:
            self.data[:, col]=numpy.reshape(nc, -1)
            self.column_headers[col]=header
        return self

    def split(self,xcol,func=None):
        """Splits the current :py:calss:`AnalyseFile` object into multiple :py:class@`AnalyseFile` objects where each one contains the rows
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
            for val in numpy.unique(self.column(xcol)):
                files[str(val)]=self.clone
                files[str(val)].data=self.search(xcol,val)
        else:
            xcol=self.find_col(xcol)
            for r in self.rows():
                x=r[xcol]
                key=func(x,r)
                if key not in files:
                    files[key]=self.clone
                    files[key].data=numpy.array([r])
                else:
                    files[key]=files[key]+r
        for k in files:
            files[k].filename="{}={}".format(xcol,k)
        if len(morecols)>0:
            for k in sorted(list(files.keys())):
                out.groups[k]=files[k].split(morecols,morefuncs)
        else:
            out.files=[files[k] for k in sorted(files.keys())]
        return out

    def SG_Filter(self, col, points, poly=1, order=0,result=None, replace=False, header=None):
        """ Implements Savitsky-Golay filtering of data for smoothing and differentiating data

        Args:
            col (index): Column of Data to be filtered
            prints (int): Number of data points to use in the filtering window. Should be an odd number > poly+1

        Keyword Arguments:
            poly (int): Order of polynomial to fit to the data. Must be equal or greater than order (default 1)
            order (int): Order of differentiation to carry out. Default=0 meaning smooth the data only.

        SG_Filter(column,points, polynomial order, order of differentuation)
        or
        SG_Filter((x-col,y,col),points,polynomial order, order of differentuation)"""
        from Stoner.Util import ordinal
        p=points
        if isinstance(col, tuple):
            x=self.column(col[0])
            x=numpy.append(numpy.array([x[0]]*p), x)
            x=numpy.append(x, numpy.array([x[-1]]*p))
            y=self.column(col[1])
            y=numpy.append(numpy.array([y[0]]*p), y)
            y=numpy.append(y, numpy.array([y[-1]]*p))
            dx=self.__SG_smooth(x, self.__SG_calc_coeff(points, poly, order))
            dy=self.__SG_smooth(y, self.__SG_calc_coeff(points, poly, order))
            r=dy/dx
        else:
            d=self.column(col)
            d=numpy.append(numpy.array([d[0]]*p),d)
            d=numpy.append(d, numpy.array([d[-1]]*p))
            r=self.__SG_smooth(d, self.__SG_calc_coeff(points, poly, order))
        if result is not None:
            if not isinstance(header, string_types):
                header='{} after {} order Savitsky-Golay Filter'.format(self.column_headers[self.find_col(col)],ordinal(order))
            if isinstance(result, bool) and result:
                result=self.shape[1]-1
                self.add_column()

        return r[p:-p]


    def threshold(self, col, threshold, rising=True, falling=False,xcol=None,all_vals=False):
        """Finds partial indices where the data in column passes the threshold, rising or falling

        Args:
            col (index): Column index to look for data in
            threshold (float): Value to look for in column col

        Keyword Arguments:
            rising (bool):  look for case where the data is increasing in value (defaukt True)
            falling (bool): look for case where data is fallinh in value (default False)
            xcol (index or None): rather than returning a fractional row index, return the
                interpolated value in column xcol
                all_vals (bool): return all crossing points of the threshold or just the first. (default False)

        Returns:
            Either a sing;le fractional row index, or an in terpolated x value"""
        current=self.column(col)
        if isinstance(threshold, (list,numpy.ndarray)):
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
            ret=[self.interpolate(r)[self.find_col(xcol)] for r in ret]
        if all_vals:
            return ret
        else:
            return ret[0]

    def interpolate(self, newX,kind='linear', xcol=None ):
        """AnalyseFile.interpolate(newX, kind='linear",xcol=None)

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
        from scipy.interpolate import interp1d
        l=numpy.shape(self.data)[0]
        index=numpy.arange(l)
        if xcol is not None: # We need to convert newX to row indices
            xfunc=interp1d(self.column(xcol), index, kind, 0) # xfunc(x) returns partial index
            newX=xfunc(newX)
        inter=interp1d(index, self.data, kind, 0)
        return inter(newX)

    def peaks(self, ycol, width, significance=None , xcol=None, peaks=True, troughs=False, poly=2,  sort=False):
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
        from scipy.interpolate import interp1d
        assert poly>=2,"poly must be at least 2nd order in peaks for checking for significance of peak or trough"
        if significance is None: # Guess the significance based on the range of y and width settings
            dm=self.max(ycol)[0]
            dp=self.min(ycol)[0]
            dm=dm-dp
            significance=0.2*dm/(4*width**2)
        d1=self.SG_Filter(ycol, width, poly, 1)
        i=numpy.arange(len(d1))
        d2=interp1d(i, self.SG_Filter(ycol, width, poly, 2))
        if xcol==None:
            xcol=i
        else:
            xcol=self.column(xcol)
        index=interp1d(i, xcol)
        w=abs(xcol[0]-xcol[width]) # Approximate width of our search peak in xcol
        z=numpy.array(self.__threshold(0, d1, rising=troughs, falling=peaks))
        z=[zv for zv in z if zv>w/2.0 and zv<max(xcol)-w/2.0] #Throw out peaks or troughts too near the ends
        if sort:
            z=numpy.take(z, numpy.argsort(d2(z)))
        return index([x for x in z if numpy.abs(d2(x))>significance])

    def integrate(self,xcol,ycol,result=None,result_name=None, bounds=lambda x,y:True,**kargs):
        """Inegrate a column of data, optionally returning the cumulative integral

        Args:
            xcol (index): The X data column index (or header)
            ycol (index) The Y data column index (or header)

        Keyword Arguments:
            result (index or None): Either a column index (or header) to overwrite with the cumulative data, or True to add a new column
                or None to not store the cumulative result.
            result_name (string): The new column header for the results column (if specified)
            bounds (callable): A function that evaluates for each row to determine if the data should be integrated over.
            kargs: Other keyword arguements are fed direct to the scipy.integrate.cumtrapz method

        Returns:
            The final integration result

        Note:
            This is a pass through to the scipy.integrate.cumtrapz routine which just uses trapezoidal integration. A better alternative would be
            to offer a variety of methods including simpson's rule and interpolation of data."""
        xc=self.find_col(xcol)
        xdat=[]
        ydat=[]
        yc=self.find_col(ycol)
        for r in self.rows():
            xdat.append(r[xc])
            if bounds(r[xc],r):
                ydat.append(r[yc])
            else:
                ydat.append(0)
        xdat=numpy.array(xdat)
        ydat=numpy.array(ydat)
        resultdata=cumtrapz(xdat,ydat,**kargs)
        resultdata=numpy.append(numpy.array([0]),resultdata)
        if result is not None:
            if isinstance(result,bool) and result:
                self.add_column(resultdata,result_name)
            else:
                result_name=self.column_headers[self.find_col(result)]
                self.add_column(resultdata,result_name,index=result,replace=True)
        return resultdata[-1]

    def mpfit(self, func,  xcol, ycol, p_info,  func_args=dict(), sigma=None, bounds=lambda x, y: True, **mpfit_kargs ):
        """Runs the mpfit algorithm to do a curve fitting with constrined bounds etc

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
            sigma=numpy.ones(numpy.shape(y), numpy.float64)
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

    def nlfit(self, ini_file, func):
        """Non-linear fitting using the :py:mod:`Stoner.nlfit` module

        Args:
            ini_file (string): path to ini file with model
            func (string or callable):Name of function to fit with (as seen in FittingFuncs.py module in Stoner)
                    or function instance to fit with

        Returns:
            AnalyseFile instance, matplotlib.fig instance (or None if plotting disabled in the inifile)
        """
        return Stoner.nlfit.nlfit(ini_file, func, data=self)

    def chi2mapping(self, ini_file, func):
        """Non-linear fitting using the :py:mod:`Stoner.nlfit` module

        Args:
            ini_file (string): Path to ini file with model
            func_name (string or callable): Name of function to fit with (as seen in FittingFuncs.py module in Stoner)
                or the function itself.

        ReturnsL
            AnalyseFile instance, matplotlib.fig instance (or None if plotting disabled), DataFile instance of parameter steps"""
        return Stoner.nlfit.nlfit(ini_file, func, data=self, chi2mapping=True)


