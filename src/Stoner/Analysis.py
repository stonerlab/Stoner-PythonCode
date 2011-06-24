
#############################################
#
# AnalysisFile object of the Stoner Package
#
# $Id: Analysis.py,v 1.7 2011/06/24 16:23:58 cvs Exp $
#
# $Log: Analysis.py,v $
# Revision 1.7  2011/06/24 16:23:58  cvs
# Update API documentation. Minor improvement to save method to force a dialog box.
#
# Revision 1.6  2011/05/10 22:10:31  cvs
# Workaround new behaviou of deepcopy() in Python 2.7 and improve handling when a typehint for the metadata doesn't exist (printing the DataFile will fix the typehinting).
#
# Revision 1.5  2011/05/09 18:34:48  cvs
# Minor changes to AnalyseFile
#
# Revision 1.4  2011/03/09 11:02:38  cvs
# Fix bug in polyfit
#
# Revision 1.3  2011/01/11 18:55:57  cvs
# Move mpfit into a method of AnalyseFile and make the API like AnalyseFile.curvefit
#
# Revision 1.2  2011/01/10 23:11:21  cvs
# Switch to using GLC's version of the mpit module
# Made PlotFile.plot_xy take keyword arguments and return the figure
# Fixed a missing import math in AnalyseFile
# Major rewrite of CSA's PCAR fitting code to use mpfit and all the glory of the Stoner module - GB
#
# Revision 1.1  2011/01/08 20:30:02  cvs
# Complete splitting Stoner into a package with sub-packages - Core, Analysis and Plot.
# Setup some imports in __init__ so that import Stoner still gets all the subclasses - Gavin
#
#
#############################################

from .Core import DataFile
import scipy
import numpy
import math
import sys

class AnalyseFile(DataFile):
    """Extends DataFile with numpy passthrough functions"""

#Private Helper Functions
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

    
    def polyfit(self,column_x,column_y,polynomial_order, bounds=lambda x, y:True):
        """ Pass through to numpy.polyfit
        
                AnalysisFile.polyfit(xx_column,y_column,polynomial_order,bounds function)
                
                x_column and y_column can be integers or strings that match the column headings
                bounds function should be a python function that takes a single paramter that represents an x value
                and returns true if the datapoint is to be retained and false if it isn't."""
        working=self.search(column_x, bounds)
        return numpy.polyfit(working[self.find_col(column_x)],working[self.find_col(column_y)],polynomial_order)
        
    def curve_fit(self, func,  xcol, ycol, p0=None, sigma=None, bounds=lambda x, y: True ):
        """General curve fitting function passed through from numpy
        
                AnalysisFile.Curve_fit(fitting function, x-column,y_column, initial parameters=None, weighting=None, bounds function)
                
                The fitting function should have prototype y=f(x,p[0],p[1],p[2]...)
                The x-column and y-column can be either strings to be matched against column headings or integers. 
                The initial parameter values and weightings default to None which corresponds to all parameters starting 
                at 1 and all points equally weighted. The bounds function has format b(x, y-vec) and rewturns true if the 
                point is to be used and false if not.
        """
        from scipy.optimize import curve_fit
        working=self.search(xcol, bounds, [xcol, ycol])
        popt, pcov=curve_fit(func,  working[:, 0], working[:, 1], p0, sigma)
        return popt, pcov
        
    def max(self, column):
        """FInd maximum value and index in a column of data
                
                AnalysisFile.max(column)
                """
        col=self.find_col(column)
        return self.data[:, col].max(), self.data[:, col].argmax()
        
    def min(self, column):
        """FInd minimum value and index in a column of data
                
                AnalysisFile.min(column)
                """
        col=self.find_col(column)
        return self.data[:, col].min(), self.data[:, col].argmin()
    
    def apply(self, func, col, insert=False, header=None):
        """Applies the given function to each row in the data set and adds to the data set
        
            AnalysisFile.apply(func,column,insert=False)"""
        col=self.find_col(col)
        nc=numpy.array([func(row) for row in self.rows()])
        if insert==True:
            if header==None:
                header=func.__name__
            self=self.add_column(nc, header, col)
        else:
            self.data[:, col]=nc
        return self

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
            y=elf.column(col[1])
            y=anumpy.append(numpy.array([y[0]]*p), y)
            y=anumpy.append(y, numpy.array([y[-1]]*p))
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
    def threshold(self, col, threshold, rising=True, falling=False):
        """AnalysisFile.threshold(column, threshold, rising=True,falling=False)
        
        Finds partial indices where the data in column passes the threshold, rising or falling"""
        current=self.column(col)
        if isinstance(threshold, list) or isinstance(threshold, numpy.ndarray):
            ret=[self.__threshold(x, current, rising=rising, falling=falling)[0] for x in threshold]
        else:
            ret=self.__threshold(threshold, current, rising=rising, falling=falling)[0]
        return ret
        
    def interpolate(self, newX,kind='linear', xcol=None ):
        """AnalyseFile.interpolate(newX, kind='linear",xcol=None)
        
        @param newX Row indices or X column values to interpolate with
        @param kind Type of interpolation function to use - does a pass through from numpy. Default is linear.
        @param xcol Column index or label that contains the data to use with newX to determine which rows to return. Defaults to None.peaks
        @return Returns a 2D numpy array representing a section of the current object's data.
        
        Returns complete rows of data corresponding to the indices given in newX. if xcol is None, then newX is interpreted as (fractional) row indices. 
        Otherwise, the column specified in xcol is thresholded with the values given in newX and the resultant row indices used to return the data.
        """
        if not xcol is None:
            newX=self.threshold(xcol, newX, True, True)
        from scipy.interpolate import interp1d
        l=numpy.shape(self.data)[0]
        index=numpy.arange(l)
        inter=interp1d(index, self.data, kind, 0)
        return inter(newX)
    
    def peaks(self, ycol, width, significance , xcol=None, peaks=True, troughs=False, poly=2):
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
        d1=self.SG_Filter(ycol, width, poly, 1)
        i=numpy.arange(len(d1))
        d2=interp1d(i, self.SG_Filter(ycol, width, poly, 2))
        if xcol==None:
            xcol=i
        else:
            xcol=self.column(xcol)
        index=interp1d(i, xcol)
        z=self.__threshold(0, d1, rising=troughs, falling=peaks)
        return index(filter(lambda x: numpy.abs(d2(x))>significance, z))       
        
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

        
