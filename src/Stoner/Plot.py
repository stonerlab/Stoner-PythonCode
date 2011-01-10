#############################################
#
#PlotFile object of the Stoner Package
#
# $Id: Plot.py,v 1.2 2011/01/10 23:11:21 cvs Exp $
#
# $Log: Plot.py,v $
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


class PlotFile(DataFile):
    """Extends DataFile with plotting functions"""
    def __init__(self, *args, **kargs): #Do the import of pylab here to speed module load
        global pylab
        import pylab
        super(PlotFile, self).__init__(*args, **kargs)
    def plot_xy(self,column_x, column_y, format=None,show_plot=True,  title='', save_filename='', **kwords):
        """plot_xy(x column, y column/s, title,save filename, show plot=True)
        
                Makes and X-Y plot of the specified data."""
        column_x=self.find_col(column_x)
        column_y=self.find_col(column_y)
        x=self.column(column_x)
        y=self.column(column_y)
        if show_plot == True:
            pylab.ion()
        if format==None:
            figure=pylab.plot(x,y, **kwords)
        else:
            figure=pylab.plot(x,y, format, **kwords)
        pylab.draw()
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
     
 
