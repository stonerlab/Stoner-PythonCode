#############################################
#
#PlotFile object of the Stoner Package
#
# $Id: Plot.py,v 1.7 2011/06/14 08:14:38 cvs Exp $
#
# $Log: Plot.py,v $
# Revision 1.7  2011/06/14 08:14:38  cvs
# Ammended platform.system()=Darwin for correct MacOSx implementation. - CSA
#
# Revision 1.6  2011/05/09 18:47:32  cvs
# Try to improve handling of Mac OSX detection
#
# Revision 1.5  2011/05/08 18:25:00  cvs
# Correct the Raman load to include the last point in the Xdata
#
# Revision 1.4  2011/02/18 22:37:13  cvs
# Make PlotFile return the matplotlib figure when plotting and allow it to handle multiple figures. Add methofs to force a redraw and also to access the Axes object. -Gavin
#
# Revision 1.3  2011/02/13 15:51:08  cvs
# Merge in ma gui branch back to HEAD
#
# Revision 1.2.2.1  2011/01/19 16:43:50  cvs
# Added OSX specific backend for matplotlib graphics
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
import matplotlib
import os
import platform
if os.name=="posix" and platform.system()=="Darwin":
    matplotlib.use('MacOSX')
from matplotlib import pyplot

class PlotFile(DataFile):
    """Extends DataFile with plotting functions"""
    
    __figure=None
    
    def __init__(self, *args, **kargs): #Do the import of pylab here to speed module load
        global pylab
        import pylab
        super(PlotFile, self).__init__(*args, **kargs)
    def plot_xy(self,column_x, column_y, format=None,show_plot=True,  title='', save_filename='', figure=None, **kwords):
        """plot_xy(x column, y column/s, title,save filename, show plot=True)
        
                Makes and X-Y plot of the specified data."""
        column_x=self.find_col(column_x)
        column_y=self.find_col(column_y)
        x=self.column(column_x)
        y=self.column(column_y)
        if isinstance(figure, int):
            figure=pylab.figure(figure)
        elif isinstance(figure, matplotlib.figure.Figure):
            pylab.figure(figure.number)
        elif isinstance(self.__figure,  matplotlib.figure.Figure):
            figure=self.__figure
        else:
            figure=pylab.figure()
        if show_plot == True:
            pylab.ion()
        if format==None:
            pylab.plot(x,y, **kwords)
        else:
            figure=pylab.plot(x,y, format, **kwords)        
        pyplot.draw()
        pylab.xlabel(str(self.column_headers[column_x]))
        if isinstance(column_y, list):
            ylabel=column_y
            ylabel[0]=self.column_headers[column_y[0]]
            ylabel=reduce(lambda x, y: x+","+self.column_headers[y],  ylabel)
        else:
            ylabel=self.column_headers[column_y]
        pylab.ylabel(str(ylabel))
        if title=='':
            title=self.filename
        pylab.title(title)
        pylab.grid(True)
        if save_filename != '':
            pylab.savefig(str(save_filename))
        self.__figure=figure
        return figure
    
    def draw(self):
        """Pass through to pylab to force figure redraw"""
        pylab.figure(self.__figure.number)
        pylab.draw()
        
    def axes(self):
        """Get the axes object of the current figure"""
        if isinstance(self.__figure, matplotlib.figure.Figure):
            return self.__figure.axes
        else:
            return None
        
    def figure(figure):
        """Set the figure used by \b Stoner.PlotFile
         @param figure A matplotlib figure or figure number
         @return The current \b Stoner.PlotFile instance"""
        if isinstance(figure, int):
            figure=pylab.figure(figure)
        elif isinstance(figure, matplotlib.figure.Figure):
            pylab.figure(figure.number)
        self.__figure=figure
        return self
         
 
