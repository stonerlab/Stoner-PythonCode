"""Stoner .Analysis provides a subclass of DataFile that has extra analysis routines builtin.

@b AnalyseFile - @b DataFile with extra bells and whistles.
"""
from .Core import DataFile
import Stoner.FittingFuncs
import Stoner.nlfit
import numpy
from scipy.integrate import cumtrapz
import math
import sys

def cov2corr(M):
    """ Converts a covariance matrix to a correlation matrix. Taken from bvp.utils.misc"""
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
    """@b Stoner.Analysis.AnalyseFile extends DataFile with numpy passthrough functions


    @b AnalyseFile provides the mthods to manipulate and fit data in variety of ways."""

    def __SG_calc_coeff(self, num_points, pol_degree=1, diff_order=0):
        """ calculates filter coefficients for symmetric savitzky-golay filter.
            see: http://www.nrbook.com/a/bookcpdf/c14-8.pdf

            num_points   means that 2*num_points+1 values contribute to the
                     smoother.

            pol_degree   is degree of fitting polynomial

            diff_order   is degree of implicit differentiation.
                     0 means that filter results in smoothing of function
                     1 means that filter results in smoothing the first
                                                 derivative of function.
                     and so on ...

        """

        # setup interpolation matrix
        # ... you might use other interpolation points
        # and maybe other functions than monomials ....

        x = numpy.arange(-num_points, num_points+1, dtype=int)
        monom = lambda x, deg : math.pow(x, deg)

        A = numpy.zeros((2*num_points+1, pol_degree+1), float)
        for i in range(2*num_points+1):
            for j in range(pol_degree+1):
                A[i,j] = monom(x[i], j)

        # calculate diff_order-th row of inv(A^T A)
        ATA = numpy.dot(A.transpose(), A)
        rhs = numpy.zeros((pol_degree+1,), float)
        rhs[diff_order] = (-1)**diff_order
        wvec = numpy.linalg.solve(ATA, rhs)

        # calculate filter-coefficients
        coeff = numpy.dot(A, wvec)

        return coeff

    def __SG_smooth(self, signal, coeff):
        """ applies coefficients calculated by calc_coeff()
            to signal """

        N = numpy.size(coeff-1)/2
        res = numpy.convolve(signal, coeff)
        return res[N:-N]

    def __threshold(self, threshold, data, rising=True, falling=False):
        """ Internal function that implements the threshold method - also used in peak-finder"""
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
        return filter(lambda x:x>0,  map(lambda x:x[0]-1+(x[1]-threshold)/(x[1]-x[2]), filter(expr, sdat)))

    def __mpf_fn(self, p, **fa):
        # Parameter values are passed in "p"
        # If FJAC!=None then partial derivatives must be comptuer.
        # FJAC contains an array of len(p), where each entry
        # is 1 if that parameter is free and 0 if it is fixed.
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

    def mpfit_iterfunct(self, myfunct, p, iter, fnorm, functkw=None,parinfo=None, quiet=0, dof=None):
        # Prints a single . for each iteration
        sys.stdout.write('.')
        sys.stdout.flush()


    def polyfit(self,column_x,column_y,polynomial_order, bounds=lambda x, y:True, result=None):
        """ Pass through to numpy.polyfit

                AnalysisFile.polyfit(xx_column,y_column,polynomial_order,bounds function,result=None)

                x_column and y_column can be integers or strings that match the column headings
                bounds function should be a python function that takes a single paramter that represents an x value
                and returns true if the datapoint is to be retained and false if it isn't."""
        working=self.search(column_x, bounds)
        p= numpy.polyfit(working[:, self.find_col(column_x)],working[:, self.find_col(column_y)],polynomial_order)
        if result is not None:
            self.add_column(numpy.polyval(p, self.column(column_x)), index=result, replace=False, column_header='Fitter with '+str(polynomial_order)+' order poylnomial')
        return p

    def curve_fit(self, func,  xcol, ycol, p0=None, sigma=None, bounds=lambda x, y: True, result=None, replace=False, header=None ):
        """General curve fitting function passed through from numpy

                @param func A callable object that represents the fitting function with the form def f(x,*p) where p is a list of fitting parameters
                @param xcol The index of the x-column data to fit
                @param ycol The index of the y-column data to fit
                @param p0 A vector of initial parameter values to try
                @sigma See scipy documentation
                @bounds A callable object that evaluates true if a row is to be included. Should be of the form f(x,y)
                @result Determines whether the fitted data should be added into the DataFile object. If result is True then the last column
                will be used. If result is a string or an integer then it is used as a column index. Default to None for not adding fitted data
                @param replace Inidcatesa whether the fitted data replaces existing data or is inserted as a new column
                @param header If this is a string then it is used as the name of the fitted data.

                AnalysisFile.Curve_fit(fitting function, x-column,y_column, initial parameters=None, weighting=None, bounds function)

                The fitting function should have prototype y=f(x,p[0],p[1],p[2]...)
                The x-column and y-column can be either strings to be matched against column headings or integers.
                The initial parameter values and weightings default to None which corresponds to all parameters starting
                at 1 and all points equally weighted. The bounds function has format b(x, y-vec) and rewturns true if the
                point is to be used and false if not.
        """
        from scipy.optimize import curve_fit
        from inspect import getargspec

        working=self.search(xcol, bounds, [xcol, ycol])
        working=numpy.reshape(working[numpy.logical_not(working.mask)],(-1,2))
        popt, pcov=curve_fit(func,  working[:, 0], working[:, 1], p0, sigma)
        if result is not None:
            (args, varargs, keywords, defaults)=getargspec(func)
            for i in range(len(popt)):
                self['Fit '+func.__name__+'.'+str(args[i+1])]=popt[i]
            xc=self.find_col(xcol)
            if not isinstance(header, str):
                header='Fitted with '+func.__name__
            if isinstance(result, bool) and result:
                result=self.shape[1]-1
            self.apply(lambda x:func(x[xc], *popt), result, replace=replace, header=header)
        return popt, pcov

    def max(self, column, bounds=None):
        """FInd maximum value and index in a column of data
        @param column Column to look for the maximum in
        @param bounds A callable function that takes a single argument list of numbers representing one row, and returns True for all rows to search in.
        @return (maximum value,row index of max value)


                AnalysisFile.max(column)
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

        @param column Column to look for the minimum in
        @param bounds A callable function that takes a single argument list of numbers representing one row, and returns True for all rows to search in.
        @return (minimum value,row index of min value)

                 AnalysisFile.min(column)
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
        @param column Column to look for the max and min values in
        @param bounds A callable function that takes a single argument list of numbers representing one row, and returns True for all rows to search in.
        @return A tuple of (min value, max value)"""

        return (self.min(column, bounds)[0], self.max(column, bounds)[0])

    def clip(self, column, clipper):
        """Clips the data based on the column and the clipper value
        @param column Column to look for the clipping value in
        @param clipper Either a tuple of (min,max) or a numpy.ndarray - in which case the max and min values in that array will be used as the clip limits"""

        clipper=(min(clipper), max(clipper))
        self=self.del_rows(column, lambda x, y:x<clipper[0] or x>clipper[1])
        return self



    def mean(self, column, bounds=None):
        """FInd mean value of a data column

        @param column Column to look for the minimum in
        @param bounds A callable function that takes a single argument list of numbers representing one row, and returns True for all rows to search in.
        @return mean value of data column

        @todo Fix the row index when the bounds function is used - see note of \b max
                AnalysisFile.min(column)"""

        col=self.find_col(column)
        if bounds is not None:
            self._push_mask()
            self._set_mask(bounds, True, col)
        result=self.data[:, col].mean()
        if bounds is not None:
            self._pop_mask()
        return result


    def normalise(self, target, base, replace=True, header=None):
        """Normalise data columns by dividing through by a base column value.

        @param target One or more target columns to normalise can be a string, integer or list of strings or integers.
        @param base The column to normalise to, can be an integer or string
        @param replace Set True(default) to overwrite  the target data columns
        @param header The new column header - default is target name(norm)
        @return A copy of the current object"""

        if isinstance(base, float):
            base=base*numpy.ones(len(self))
        elif isinstance(base, numpy.ndarray) and len(base.shape)==1 and len(base)==len(self):
            pass
        else:
            base=self.column(base)
        if not isinstance(target, list):
            target=[self.find_col(target)]
        else:
            target=[self.find_col(t) for t in target]
        for t in target:
            if header is None:
                header=self.column_headers[t]+"(norm)"
            else:
                header=str(header)
            self.add_column(self.column(t)/numpy.array(base), header, t, replace=replace)
        return self

    def subtract(self, a, b, replace=False, header=None):
        """Subtract one column from another column
        @param a First column to subtract from
        @param b Second column to subtract from a may be a column index, floating point number or a 1D array of numbers
        @param header new column header  (defaults to a-b
        @param replace Replace the a column with the a-b data
        @return A copy of the new data object"""
        a=self.find_col(a)
        if isinstance(b, float):
            if header is None:
                header=self.column_headers[a]+"- "+str(b)
            self.add_column(self.column(a)-b, header, a, replace=replace)
        elif isinstance(b, numpy.ndarray) and len(b.shape)==1 and len(b)==len(self):
            if header is None:
                header=self.column_headers[a]+"- data"
            self.add_column(self.column(a)-numpy.array(b), header, a, replace=replace)
        else:
            b=self.find_col(b)
            if header is None:
                header=self.column_headers[a]+"-"+self.column_headers[b]
            self.add_column(self.column(a)-self.column(b), header, a, replace=replace)
        return self

    def add(self, a, b, replace=False, header=None):
        """Subtract one column from another column
        @param a First column to add to
        @param b Second column to add to a, may be a column index, floating point number or 1D array of numbers
        @param header new column header  (defaults to a-b
        @param replace Replace the a column with the a-b data
        @return A copy of the new data object"""
        a=self.find_col(a)
        if isinstance(b, float):
            if header is None:
                header=self.column_headers[a]+"+ "+str(b)
            self.add_column(self.column(a)+b, header, a, replace=replace)
        elif isinstance(b, numpy.ndarray) and len(b.shape)==1 and len(b)==len(self):
            if header is None:
                header=self.column_headers[a]+"+ data"
            self.add_column(self.column(a)+numpy.array(b), header, a, replace=replace)
        else:
            b=self.find_col(b)
            if header is None:
                header=self.column_headers[a]+"+"+self.column_headers[b]
            self.add_column(self.column(a)+self.column(b), header, a, replace=replace)
        return self

    def divide(self, a, b, replace=False, header=None):
        """Divide data columns by dividing through by a base column value. synonym of normalise, but note the opposite default to replace.

        @param target One or more target columns to normalise can be a string, integer or list of strings or integers.
        @param base The column to normalise to, can be an integer or string
        @param replace Set True(default) to overwrite  the target data columns
        @param header The new column header - default is target name(norm)
        @return A copy of the current object"""
        a=self.find_col(a)
        if isinstance(b, float):
            if header is None:
                header=self.column_headers[a]+"/ "+str(b)
            self.add_column(self.column(a)/b, header, a, replace=replace)
        elif isinstance(b, numpy.ndarray) and len(b.shape)==1 and len(b)==len(self):
            if header is None:
                header=self.column_headers[a]+"/ data"
            self.add_column(self.column(a)/numpy.array(b), header, a, replace=replace)
        else:
            b=self.find_col(b)
            if header is None:
                header=self.column_headers[a]+"/"+self.column_headers[b]
            self.add_column(self.column(a)/self.column(b), header, a, replace=replace)
        return self


    def mulitply(self, a, b, replace=False, header=None):
        """Subtract one column from another column
        @param a First column to multiply
        @param b Second column to multiply a with, may be a column index, floating point number or 1D array of numbers
        @param header new column header  (defaults to a*b
        @param replace Replace the a column with the a*b data
        @return A copy of the new data object"""
        a=self.find_col(a)
        if isinstance(b, float):
            if header is None:
                header=self.column_headers[a]+"* "+str(b)
            self.add_column(self.column(a)*b, header, a, replace=replace)
        elif isinstance(b, numpy.ndarray) and len(b.shape)==1 and len(b)==len(self):
            if header is None:
                header=self.column_headers[a]+"* data"
            self.add_column(self.column(a)*numpy.array(b), header, a, replace=replace)
        else:
            b=self.find_col(b)
            if header is None:
                header=self.column_headers[a]+"*"+self.column_headers[b]
            self.add_column(self.column(a)*self.column(b), header, a, replace=replace)
        return self


    def apply(self, func, col, replace=True, header=None):
        """Applies the given function to each row in the data set and adds to the data set

            @param func A function that takes a numpy 1D array representing each row of data
            @param col The column in which to place the result of the function
            @param replace Keyword argument indicating to isnert a new data column (False) or replace the data column (True)
            @param header The new column header (defaults to the name of the function func"""
        col=self.find_col(col)
        nc=numpy.array([func(row) for row in self.rows()])
        if header==None:
            header=func.__name__
        if replace!=True:
            self=self.add_column(nc, header, col)
        else:
            self.data[:, col]=numpy.reshape(nc, -1)
            self.column_headers[col]=header
        return self
        
    def split(self,xcol,func=None):
        """Splits the current AnalyseFile object into multiple AnalyseFile objects where each one contains the rows
        from the original object which had the same value of a given column.
        
        @param xcol The index of the column to look for values in. This can be a list in which case a DataFolder with groups
        with subfiles is built up by applying each item in the xcol list recursively.
        @param func A callable function that can be evaluated to find the value to determine which output object
        each row belongs in. If this is left as the default None then the column value is converted to a string and that is used.
        The function to be of the form f(x,r) where x is a single float value and r is a list of floats representing the complete row.
        The return value should be a hashable value. @a func can also be a list if @A xcol is a list, in which the @a func values are used along
        with the @a xcol values.
        @return A DataFolder object containing the individual AnalyseFile objects
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
            for k in sorted(files.keys()):
                out.groups[k]=files[k].split(morecols,morefuncs)
        else:
            out.files=[files[k] for k in sorted(files.keys())]
        return out

    def SG_Filter(self, col, points, poly=1, order=0):
        """ Implements Savitsky-Golay filtering of data for smoothing and differentiating data

        SG_Filter(column,points, polynomial order, order of differentuation)
        or
        SG_Filter((x-col,y,col),points,polynomial order, order of differentuation)"""
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
            return r[p:-p]
        else:
            d=self.column(col)
            d=numpy.append(numpy.array([d[0]]*p),d)
            d=numpy.append(d, numpy.array([d[-1]]*p))
            r=self.__SG_smooth(d, self.__SG_calc_coeff(points, poly, order))
            return r[p:-p]
    def threshold(self, col, threshold, rising=True, falling=False,xcol=None,all=False):
        """Finds partial indices where the data in column passes the threshold, rising or falling
        @param col Column index to look for data in
        @param threshold Value to look for in column col
        @param rising (defaukt True) look for case where the data is increasing in value
        @param falling (default False) look for case where data is fallinh in value
        @param xcol (default None) rather than returning a fractional row index, return the
        interpola,ted value in column xcol
        @param all (default False) return all crossing points of the threshold or just the first.
        @return Either a sing;le fractional row index, or an in terpolated x value"""
        current=self.column(col)
        if isinstance(threshold, list) or isinstance(threshold, numpy.ndarray):
            ret=[self.__threshold(x, current, rising=rising, falling=falling,all=all) for x in threshold]
        else:
            if all:
                ret=self.__threshold(threshold, current, rising=rising, falling=falling)
            else:
                ret=[self.__threshold(threshold, current, rising=rising, falling=falling)[0]]               
        if xcol is not None:
            ret=[self.interpolate(r)[self.find_col(xcol)] for r in ret]
        if all:
            return ret
        else:
            return ret[0]

    def interpolate(self, newX,kind='linear', xcol=None ):
        """AnalyseFile.interpolate(newX, kind='linear",xcol=None)

        @param newX Row indices or X column values to interpolate with
        @param kind Type of interpolation function to use - does a pass through from numpy. Default is linear.
        @param xcol Column index or label that contains the data to use with newX to determine which rows to return. Defaults to None.peaks
        @return Returns a 2D numpy array representing a section of the current object's data.

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

    def peaks(self, ycol, width, significance=None , xcol=None, peaks=True, troughs=False, poly=2,  sorted=False):
        """AnalysisFile.peaks(ycol,width,signficance, xcol=None.peaks=True, troughs=False)

        Locates peaks and/or troughs in a column of data by using SG-differentiation.

        @param ycol is the column name or index of the data in which to search for peaks
        @param width is the expected minium halalf-width of a peak in terms of the number of data points.
                This is used in the differnetiation code to find local maxima. Bigger equals less sensitive
                to experimental noise, smaller means better eable to see sharp peaks
            @param poly This is the order of polynomial to use when differentiating the data to locate a peak. Must >=2, higher numbers
            will find sharper peaks more accurately but at the risk of finding more false positives.
            @param significance is used to decide whether a local maxmima is a significant peak. Essentially just the curvature
                of the data. Bigger means less sensistive, smaller means more likely to detect noise.
            @param xcol name or index of data column that p[rovides the x-coordinate (default None)
            @param peaks select whether to measure peaks in data (default True)
            @param troughs select whether to measure troughs in data (default False)
            @return If xcol is None then returns conplete rows of data corresponding to the found peaks/troughs. If xcol is not none, returns a 1D array of the x positions of the peaks/troughs.
            """
        from scipy.interpolate import interp1d
        assert poly>=2,"poly must be at least 2nd order in peaks for checking for significance of peak or trough"
        if significance is None: # Guess the significance based on the range of y and width settings
            dm, p=self.max(ycol)
            dp, p=self.min(ycol)
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
        if sorted:
            z=numpy.take(z, numpy.argsort(d2(z)))
        return index(filter(lambda x: numpy.abs(d2(x))>significance, z))

    def integrate(self,xcol,ycol,result=None,result_name=None, bounds=lambda x,y:True,**kargs):
        """Inegrate a column of data, optionally returning the cumulative integral
        @param xcol The X data column index (or header)
        @para ycol The Y data column index (or header)
        @param result Either a column index (or header) to overwrite with the cumulative data, or True to add a new column
        or None to not store the cumulative result.
        @param result_name The new column header for the results column (if specified)
        @bounds A function that evaluates for each row to determine if the data should be integrated over.
        @param kargs Other keyword arguements are fed direct to the scipy.integrate.cumtrapz method
        @return The final integration result"""
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
            elif isinstance(result,str) or isinstance(result,int):
                print type(result)
                print result
                self.add_column(resultdata,result_name,index=result,replace=True)
        return resultdata[-1]

    def mpfit(self, func,  xcol, ycol, p_info,  func_args=dict(), sigma=None, bounds=lambda x, y: True, **mpfit_kargs ):
        """Runs the mpfit algorithm to do a curve fitting with constrined bounds etc

                mpfit(func, xcol, ycol, p_info, func_args=dict(),sigma=None,bounds=labdax,y:True,**mpfit_kargs)

                func: Fitting function def func(x,parameters, **func_args)
                xcol, ycol: index the x and y data sets
                p_info: array of dictionaries that define the fitting parameters
                sigma: weights of the data poiints. If not specified, then equal weighting assumed
                bounds: function that takes x,y pairs and returns true if to be used in the fitting
                **mpfit_kargs: other lkeywords passed straight to mpfit"""
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
        """Non-linear fitting using the nlfit module
        @param ini_file: string giving path to ini file with model
        @param func: string giving name of function to fit with (as seen in FittingFuncs.py module in Stoner)
                or function instance to fit with
        @return AnalyseFile instance, matplotlib.fig instance (or None if plotting disabled)"""
        return Stoner.nlfit.nlfit(ini_file, func, data=self)

    def chi2mapping(self, ini_file, func):
        """Non-linear fitting using the nlfit module
        @param ini_file: string giving path to ini file with model
        @param func_name: string giving name of function to fit with (as seen in FittingFuncs.py module in Stoner)
        @return AnalyseFile instance, matplotlib.fig instance (or None if plotting disabled), DataFile instance of parameter steps"""
        return Stoner.nlfit.nlfit(ini_file, func, data=self, chi2mapping=True)


