#############################################
#
#PlotFile object of the Stoner Package
#
# $Id: Plot.py,v 1.9 2011/06/24 16:23:58 cvs Exp $
#
# $Log: Plot.py,v $
# Revision 1.9  2011/06/24 16:23:58  cvs
# Update API documentation. Minor improvement to save method to force a dialog box.
#
# Revision 1.8  2011/06/22 22:54:29  cvs
# Look at getting PlotFile.plot_xy to plot to the same window and update API documentation
#
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
        """Constructor of \b PlotFile class. Imports pylab and then calls the parent constructor
                @param args Arguements to pass to \b DataFile.__init__
                @param kargs Dictionary of keyword arguments to pass to \b DataFile.__init__
                
                @return This instance
        
        """
        global pylab
        import pylab
        super(PlotFile, self).__init__(*args, **kargs)
    
    def __getattr__(self, name):
        """Attribute accessor
        
                @param name Name of attribute: only "fig" is supported here to return the current figure refernce
                
                All other attrbiutes are passed over to the parent class
                """
        if name=="fig":
            return self.__figure
        else:
            super(PlotFile, self).__getattr__(name)
            
    def __setattr__(self, name, value):
        """Sets the specified attribute
                
                @param name The name of the attribute to set. Only "fig" is supported in this class - everything else drops through to the parent class
                @param value The value of the attribute to set.
    """
        if name=="fig":
            self.figure(value)
        else:
            super(PlotFile, self).__setattr__(name, value)
    
    def plot_xy(self,column_x, column_y, format=None,show_plot=True,  title='', save_filename='', figure=None, **kwords):
        """Makes a simple X-Y plot of the specified data.
                
                @param column_x An integer or string that indexes the relevant column for the x data
                @param column_y An integer go string that indexes the y-data column
                @param format Optional string parameter that specifies the format for the plot - see matplotlib documentation for details
                @param show_plot = True Turns on interactive plot control
                @param title Optional parameter that specfies the plot title - otherwise the current DataFile filename is used
                @param save_filename Filename used to save the plot
                @param figure Optional argument that controls what matplotlib figure to use. Can be an integer, or a matplotlib.figure or False. If False then a new figure is always used, otherwise it will default to using the last figure used by this DataFile object.
                @param kwords A dictionary of other keyword arguments to pass into the plot function.
                
                @return a matplotlib.figure isntance
                

        
        """
        column_x=self.find_col(column_x)
        column_y=self.find_col(column_y)
        x=self.column(column_x)
        y=self.column(column_y)
        if isinstance(figure, int):
            figure=pylab.figure(figure)
        elif isinstance(figure, bool) and not figure:
            figure=pylab.figure()
        elif isinstance(figure, matplotlib.figure.Figure):
            figure=pylab.figure(figure.number)
        elif isinstance(self.__figure,  matplotlib.figure.Figure):
            figure=self.__figure
        else:
            figure=pylab.figure()
        if show_plot == True:
            pylab.ion()
        if format==None:
            pylab.plot(x,y, figure=figure, **kwords)
        else:
            figure=pylab.plot(x,y, format, figure=figure, **kwords)        
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
        
    def figure(self, figure):
        """Set the figure used by \b Stoner.PlotFile
         @param figure A matplotlib figure or figure number
         @return The current \b Stoner.PlotFile instance"""
        if isinstance(figure, int):
            figure=pylab.figure(figure)
        elif isinstance(figure, matplotlib.figure.Figure):
            pylab.figure(figure.number)
        self.__figure=figure
        return self
         
 
