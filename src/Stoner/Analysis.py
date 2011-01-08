
#############################################
#
# AnalysisFile object of the Stoner Package
#
# $Id: Analysis.py,v 1.1 2011/01/08 20:30:02 cvs Exp $
#
# $Log: Analysis.py,v $
# Revision 1.1  2011/01/08 20:30:02  cvs
# Complete splitting Stoner into a package with sub-packages - Core, Analysis and Plot.
# Setup some imports in __init__ so that import Stoner still gets all the subclasses - Gavin
#
#
#############################################

from .Core import DataFile
import scipy
import numpy

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
        
    
    def polyfit(self,column_x,column_y,polynomial_order, bounds=lambda x, y:True):
        """ Pass through to numpy.polyfit
        
                AnalysisFile.polyfit(xx_column,y_column,polynomial_order,bounds function)
                
                x_column and y_column can be integers or strings that match the column headings
                bounds function should be a python function that takes a single paramter that represents an x value
                and returns true if the datapoint is to be retained and false if it isn't."""
        working=self.search(column_x, bounds, column_y)
        return numpy.polyfit(working[0],working[1],polynomial_order)
        
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
    
    def apply(self, func, col, insert=False):
        """Applies the given function to each row in the data set and adds to the data set
        
            AnalysisFile.apply(func,column,insert=False)"""
        col=self.find_col(col)
        nc=numpy.array([func(row) for row in self.rows()])
        if insert==True:
            self=self.add_column(nc, func.__name__, col)
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
        return self.__threshold(threshold, current, rising=rising, falling=falling)
        
    def interpolate(self, newX,kind='linear' ):
        from scipy.interpolate import interp1d
        l=numpy.shape(self.data)[0]
        index=numpy.arange(l)
        inter=interp1d(index, self.data, kind, 0)
        return inter(newX)
    
    def peaks(self, ycol, width, significance , xcol=None, peaks=True, troughs=False, poly=2):
        """AnalysisFile.peaks(ycol,width,signficance, xcol=None.peaks=True, troughs=False)
        
        Locates peaks and/or troughs in a column of data by using SG-differentiation.
        
        ycol is the column name or index of the data in which to search for peaks
        width is the expected minium halalf-width of a peak in terms of the number of data points. 
                This is used in the differnetiation code to find local maxima. Bigger equals less sensitive
                to experimental noise, smaller means better eable to see sharp peaks
            sensitivity is used to decide whether a local maxmima is a significant peak. Essentially just the curvature
                of the data. Bigger means less sensistive, smaller means more likely to detect noise.
            xcol name or index of data column that p[rovides the x-coordinate
            peaks,troughs select whether to measure peaks.troughs in data"""
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
        
