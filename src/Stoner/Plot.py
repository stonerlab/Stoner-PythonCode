#############################################
#
#PlotFile object of the Stoner Package
#
# $Id: Plot.py,v 1.12 2011/12/17 20:41:39 cvs Exp $
#
# $Log: Plot.py,v $
# Revision 1.12  2011/12/17 20:41:39  cvs
# Implement a PlotFile.plot_xyz method, make plot_xy work with multiple y columns and formats a little better. Update documentation. - Gavin
#
# Revision 1.11  2011/12/05 21:56:26  cvs
# Add in DataFile methods swap_column and reorder_columns and update API documentation. Fix some Doxygen problems.
#
# Revision 1.10  2011/12/04 23:09:16  cvs
# Fixes to Keissig and plotting code
#
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
from matplotlib import pyplot as pyplot

class PlotFile(DataFile):
    """Extends DataFile with plotting functions"""
    
    __figure=None
    
    def __init__(self, *args, **kargs): #Do the import of pyplot here to speed module load
        """Constructor of \b PlotFile class. Imports pyplot and then calls the parent constructor
                @param args Arguements to pass to \b DataFile.__init__
                @param kargs Dictionary of keyword arguments to pass to \b DataFile.__init__
                
                @return This instance
        
        """
        global pyplot
        super(PlotFile, self).__init__(*args, **kargs)
    
    def __getattr__(self, name):
        """Attribute accessor
        
                @param name Name of attribute: only "fig" is supported here to return the current figure refernce
                
                All other attrbiutes are passed over to the parent class
                """
        if name=="fig":
            return self.__figure
        elif name=="axes":
            if isinstance(self.__figure, matplotlib.figure.Figure):
                return self.__figure.axes
            else:
                return None
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
    
    def plot_xy(self,column_x, column_y, format=None,show_plot=True,  title='', save_filename='', figure=None, plotter=pyplot.plot,  **kwords):
        """Makes a simple X-Y plot of the specified data.
                
                @param column_x An integer or string that indexes the relevant column for the x data
                @param column_y An integer go string that indexes the y-data column
                @param format Optional string parameter that specifies the format for the plot - see matplotlib documentation for details
                @param show_plot = True Turns on interactive plot control
                @param title Optional parameter that specfies the plot title - otherwise the current DataFile filename is used
                @param save_filename Filename used to save the plot
                @param figure Optional argument that controls what matplotlib figure to use. Can be an integer, or a matplotlib.figure or False. If False then a new figure is always used, otherwise it will default to using the last figure used by this DataFile object.
                @param plotter Optional arguement that passes a plotting function into the routine. Sensible choices might be pyplot.plot (default), py.semilogy, pyplot.semilogx
                @param kwords A dictionary of other keyword arguments to pass into the plot function.
                
                @return a matplotlib.figure isntance
                

        
        """
        column_x=self.find_col(column_x)
        column_y=self.find_col(column_y)
        x=self.column(column_x)
        y=self.column(column_y)
        if isinstance(figure, int):
            figure=pyplot.figure(figure)
        elif isinstance(figure, bool) and not figure:
            figure=pyplot.figure()
        elif isinstance(figure, matplotlib.figure.Figure):
            figure=pyplot.figure(figure.number)
        elif isinstance(self.__figure,  matplotlib.figure.Figure):
            figure=self.__figure
        else:
            figure=pyplot.figure()
        if show_plot == True:
            pyplot.ion()
        if isinstance(column_y, list):
            for ix in range(len(column_y)):
                yt=y[:, ix]
                if isinstance(format, list):
                    plotter(x,yt, format[ix], figure=figure, **kwords)
                elif format==None:
                    plotter(x,y, figure=figure, **kwords)
                else:
                    plotter(x,y, format, figure=figure, **kwords)
        else:
            if format==None:
                plotter(x,y, figure=figure, **kwords)
            else:
                plotter(x,y, format, figure=figure, **kwords)
            
        pyplot.xlabel(str(self.column_headers[column_x]))
        if isinstance(column_y, list):
            ylabel=column_y
            ylabel[0]=self.column_headers[column_y[0]]
            ylabel=reduce(lambda x, y: x+","+self.column_headers[y],  ylabel)
        else:
            ylabel=self.column_headers[column_y]
        pyplot.ylabel(str(ylabel))
        if title=='':
            title=self.filename
        pyplot.title(title)
        pyplot.grid(True)
        if save_filename != '':
            pyplot.savefig(str(save_filename))
        pyplot.draw()
        self.__figure=figure
        return self.__figure
        
    def plot_xyz(self, xcol, ycol, zcol, shape=None, cmap=pyplot.cm.jet,show_plot=True,  title='', figure=None, plotter=None,  **kwords):
        """Plots a surface plot based on rows of X,Y,Z data using matplotlib.pcolor()
        
            @param xcol Xcolumn index or label
            @param ycol Y column index or label
            @param zcol Z column index or label
            @param shape A tuple that defines the shape of the surface (i.e. the number of X and Y value. If not procided or None, then the routine will attempt to calculate these from the data provided
            @param cmap A matplotlib colour map - defaults to the jet colour map
            @param show_plot Interactive plotting on
            @param title Text to use as the title - defaults to the filename of the dataset
            @param figure Controls what figure to use for the plot. If an integer, use that figure number; if a boolean and false, create a new figuire; if a matplotlib figure instance, use that figure; otherwisem reuse the existing figure for this dataset
            @param plotter A function to use for plotting the data - defaults to matplotlib.pcolor()
            @param kwords Other keywords to pass into the plot function.  
            
            @return The matplotib figure with the data plotted"""
            
        if shape is None:
            shape=(len(self.unique(xcol)), len(self.unique(ycol)))
        xdata=numpy.reshape(self.column(xcol), shape)
        ydata=numpy.reshape(self.column(ycol), shape)
        zdata=numpy.reshape(self.column(zcol), shape)
        if isinstance(figure, int):
            figure=pyplot.figure(figure)
        elif isinstance(figure, bool) and not figure:
            figure=pyplot.figure()
        elif isinstance(figure, matplotlib.figure.Figure):
            figure=pyplot.figure(figure.number)
        elif isinstance(self.__figure,  matplotlib.figure.Figure):
            figure=self.__figure
        else:
            figure=pyplot.figure()
        self.__figure=figure
        if show_plot == True:
            pyplot.ion()
        if plotter is None:
            plotter=self.__SurfPlotter
        plotter(xdata, ydata, zdata, cmap=cmap, **kwords)
        pyplot.xlabel(str(self.column_headers[self.find_col(xcol)]))
        pyplot.ylabel(str(self.column_headers[self.find_col(ycol)]))
        if plotter==self.__SurfPlotter:
            self.axes()[0].set_zlabel(str(self.column_headers[self.find_col(zcol)]))
        if title=='':
            title=self.filename
        pyplot.title(title)
        pyplot.draw()

        return self.__figure
        
    def __SurfPlotter(self, X, Y, Z, **kargs):
        """Utility private function to plot a 3D color mapped surface
        @param X X dataFile
        @param Y Y data
        @param Z Z data
        @param kargs Other keywords to pass through
        
        @return A matplotib Figure
        
        This function attempts to work the same as the 2D surface plotter pcolor, but draws a 3D axes set"""
        from mpl_toolkits.mplot3d import Axes3D
        ax = self.fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, **kargs)
        self.fig.colorbar(surf, shrink=0.5, aspect=5)
        return surf
   
    def draw(self):
        """Pass through to pyplot to force figure redraw"""
        pyplot.figure(self.__figure.number)
        pyplot.draw()
        
    def show(self):
        """Pass through for pyplot Figure.show()"""
        self.fig.show()
        
    def figure(self, figure):
        """Set the figure used by \b Stoner.PlotFile
         @param figure A matplotlib figure or figure number
         @return The current \b Stoner.PlotFile instance"""
        if isinstance(figure, int):
            figure=pyplot.figure(figure)
        elif isinstance(figure, matplotlib.figure.Figure):
            pyplot.figure(figure.number)
        self.__figure=figure
        return self
         
 
