"""#############################################
Stoner.Plot provides the a class to facilitate easier plotting of Stoner Data:

* @b PlotFile - A class that uses matplotlib to plot data


#############################################
"""
from .Core import DataFile
import numpy
import matplotlib
import os
import platform
if os.name=="posix" and platform.system()=="Darwin":
    matplotlib.use('MacOSX')
from matplotlib import pyplot as pyplot
from scipy.interpolate import griddata


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

    def __dir__(self):
        """Handles the local attributes as well as the inherited ones"""
        attr=self.__dict__.keys()
        attr2=[a for a in super(PlotFile, self).__dir__() if a not in attr]
        attr.extend(attr2)
        attr.extend(["fig", "axes"])
        return attr

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
            return super(PlotFile, self).__getattr__(name)

    def __setattr__(self, name, value):
        """Sets the specified attribute

                @param name The name of the attribute to set. Only "fig" is supported in this class - everything else drops through to the parent class
                @param value The value of the attribute to set.
    """
        if name=="fig":
            self.figure(value)
        else:
            super(PlotFile, self).__setattr__(name, value)

    def plot_xy(self,column_x, column_y, format=None,show_plot=True,  title='', save_filename='', figure=None, plotter=None,  **kwords):
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
        for err in ["xerr", "yerr"]:  # Check for x and y error keywords
            if err in kwords:
                if plotter is None:
                    plotter=pyplot.errorbar
                # If the keyword exists and is either an int or a string, then
                # it will be a column index, so get the matching data
                if type(kwords[err]) in [int,str]:
                    kwords[err]=self.column(kwords[err])
                elif isinstance(kwords[err], list):
                # Ok, so it's a list, so redo the check for each  item.
                    for i in range(len(kwords[err])):
                        if type(kwords[err][i]) in  [int, str]:
                            kwords[err][i]=self.column(kwords[err])
        # Now try to process the figure parameter
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
        if plotter is None: #Nothing has defined the plotter to use yet
            plotter=pyplot.plot
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
            print kwords
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

    def plot_xyz(self, xcol, ycol, zcol, shape=None, xlim=None, ylim=None,cmap=pyplot.cm.jet,show_plot=True,  title='', figure=None, plotter=None,  **kwords):
        """Plots a surface plot based on rows of X,Y,Z data using matplotlib.pcolor()

            @param xcol Xcolumn index or label
            @param ycol Y column index or label
            @param zcol Z column index or label
            @param shape A tuple that defines the shape of the surface (i.e. the number of X and Y value. If not procided or None, then the routine will attempt to calculate these from the data provided
            @param xlim A tuple that defines the x-axis limits and grid of the data to be plotted
            @param ylim A tuple that defines the Y-axis limits and grid of the data data to be plotted            
            @param cmap A matplotlib colour map - defaults to the jet colour map
            @param show_plot Interactive plotting on
            @param title Text to use as the title - defaults to the filename of the dataset
            @param figure Controls what figure to use for the plot. If an integer, use that figure number; if a boolean and false, create a new figuire; if a matplotlib figure instance, use that figure; otherwisem reuse the existing figure for this dataset
            @param plotter A function to use for plotting the data - defaults to matplotlib.pcolor()
            @param kwords Other keywords to pass into the plot function.

            @return The matplotib figure with the data plotted"""

        xdata,ydata,zdata=self.griddata(xcol,ycol,zcol,shape=shape,xlim=xlim,ylim=ylim)
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
            self.axes[0].set_zlabel(str(self.column_headers[self.find_col(zcol)]))
        if title=='':
            title=self.filename
        pyplot.title(title)
        pyplot.draw()

        return self.__figure

    def griddata(self,xc,yc=None,zc=None,shape=None,xlim=None,ylim=None,method="linear"):
        """Function to convert xyz data onto a regular grid
        
        @param xc Column to be used for the X-Data
        @param yc column to be used for Y-Data - default value is column to the right of the x-data column
        @param zc column to be used for the Z-data - default value is the column to the right of the y-data column
        @param shaoe two-tuple of the number of points along x and y in the grid - defaults to a square of sidelength = square root of the length of the data.
        @param xlim tuple of the xlimits
        @param ylim tuple of the ylimits
        @param method type of interploation to use, default is linear
        @return X,Y,Z three two dimensional arrays of the co-ordinates of the interpolated data
        """
        
        xc=self.find_col(xc)
        if yc is None:
            yc=xc+1
        else:
            yc=self.find_col(yc)
        if zc is None:
            zc=yc+1
        else:
            zc=self.find_col(zc)
        if shape is None or not(isinstance(shape,tuple) and len(shape)==2):
            shape=(numpy.floor(numpy.sqrt(len(self))),numpy.floor(numpy.sqrt(len(self))))
        if xlim is None:
            xlim=(numpy.min(self.column(xc)),numpy.max(self.column(xc)))
        if isinstance(xlim,tuple) and len(xlim)==2:
            xlim=(xlim[0],xlim[1],(xlim[1]-xlim[0])/shape[0])
        elif isinstance(xlim,tuple) and len(xlim)==3:
            xlim[2]=len(range(*xlim))
        else:
            raise RuntimeError("X limit specification not good.")
        if ylim is None:
            ylim=(numpy.min(self.column(yc)),numpy.max(self.column(yc)))
        if isinstance(ylim,tuple) and len(ylim)==2:
            ylim=(ylim[0],ylim[1],(ylim[1]-ylim[0])/shape[1])
        elif isinstance(ylim,tuple) and len(ylim)==3:
            ylim[2]=len(range(*ylim))
        else:
            raise RuntimeError("Y limit specification not good.")
            
        np=numpy.mgrid[slice(*xlim),slice(*ylim)].T
        
        points=numpy.array([self.column(xc),self.column(yc)]).T     
        Z=griddata(points,self.column(zc),np,method=method)
        
        return np[:,:,0],np[:,:,1],Z
        
    def contour_xyz(self,xc,yc,zc,shape=None,xlim=None, ylim=None, plotter=None,**kargs):
        """Grid up the three columns of data and plot
         @param xc X column
         @param yc Y column
         @param zc Z cilumn
         @param shape Tuple containing shape of the data
         @param xlim X limits to plot over
         @param ylim Y Limits to plot over
         @return A matplotlib figure
         """
         
        X,Y,Z=self.griddata(xc,yc,zc,shape,xlim,ylim)
        if plotter is None:
            plotter=pyplot.contour
             
        fig=plotter(X,Y,Z,**kargs)
        pyplot.xlabel(self.column_headers[self.find_col(xc)])
        pyplot.ylabel(self.column_headers[self.find_col(yc)])
        pyplot.title(self.filename + " "+ self.column_headers[zc])
        
        return fig
        
        
            

    def plot_matrix(self, xvals=None, yvals=None, rectang=None, cmap=pyplot.cm.jet,show_plot=True,  title='',xlabel=None, ylabel=None, zlabel=None,  figure=None, plotter=None,  **kwords):
        """Plots a surface plot by assuming that the current dataset represents a regular matrix of points.

            @param xvals Either a column index or name or a list or numpytarray of column values. The default (None) uses the first column of data
            @param yvals Either a row index or a list or numpy array of row values. The default (None) uses the column_headings interpreted as floats
            @param rectang a tuple of either 2 or 4 elements representing either the origin (row,column) or size (origin, number of rows, number of columns) of data to be used for the z0data matrix
            @param cmap A matplotlib colour map - defaults to the jet colour map
            @param show_plot Interactive plotting on
            @param title Text to use as the title - defaults to the filename of the dataset
            @param xlabel X axes label. Deafult is None - guess from xvals or metadata
            @param ylabel Y axes label, Default is None - guess from metadata
            @param zlabel Z axis label, Default is None - guess from metadata
            @param figure Controls what figure to use for the plot. If an integer, use that figure number; if a boolean and false, create a new figuire; if a matplotlib figure instance, use that figure; otherwisem reuse the existing figure for this dataset
            @param plotter A function to use for plotting the data - defaults to matplotlib.pcolor()
            @param kwords Other keywords to pass into the plot function.

            @return The matplotib figure with the data plotted"""
        # Sortout yvals values
        if isinstance(yvals, int): #  Int means we're sepcifying a data row
            if rectang is None: # we need to intitialise the rectang
                rectang=(yvals+1, 0) # We'll sort the column origin later
            elif isinstance(rectang, tuple) and rectang[1]<=yvals: # We have a rectang, but we need to adjust the row origin
                rectang[0]=yvals+1
            yvals=self[yvals] # change the yvals into a numpy array
        elif isinstance(yvals, list) or isinstance(yvals, tuple): # We're given the yvals as a list already
            yvals=numpy.array(yvals)
        elif yvals is None: # No yvals, so we'l try column headings
            if isinstance(xvals, int) or isinstance(xvals, str): # Do we have an xcolumn header to take away ?
                xvals=self.find_col(xvals)
                headers=self.column_headers[xvals+1:]
            elif xvals is None: # No xvals so we're going to be using the first column
                xvals=0
                headers=self.column_headers[1:]
            else:
                headers=self.column_headers
            yvals=numpy.array([float(x) for x in headers]) #Ok try to construct yvals aray
        else:
            raise RuntimeError("uvals must be either an integer, list, tuple, numpy array or None")
        #Sort out xvls values
        if isinstance(xvals, int) or isinstance(xvals, str): # String or int means using a column index
            if xlabel is None:
                xlabel=self.column_headers[self.find_col(xvals)]
            if rectang is None: # Do we need to init the rectan ?
                rectang=(0, xvals+1)
            elif isinstance(rectang, tuple): # Do we need to adjust the rectan column origin ?
                rectang[1]=xvals+1
            xvals=self.column(xvals)
        elif isinstance(xvals, list) or isinstance(xvals, tuple): # Xvals as a data item
            xvals=numpy.array(xvals)
        elif isinstance(xvals, numpy.ndarray):
            pass
        elif xvals is None: # xvals from column 0
            xvals=self.column(0)
            if rectang is None: # and fix up rectang
                rectang=(0, 1)
        else:
            raise RuntimeError("xvals must be a string, integer, list, tuple or numpy array or None")

        if isinstance(rectang, tuple) and len(rectang)==2: # Sort the rectang value
            rectang=(rectang[0], rectang[1], numpy.shape(self.data)[0]-rectang[0], numpy.shape(self.data)[1]-rectang[1])
        elif rectang is None:
            rectang=(0, 0, numpy.shape(self.data)[0], numpy.shape(self.data)[1])
        elif isinstance(rectang, tuple) and len(rectang)==4: # Ok, just make sure we have enough data points left.
            rectang=(rectang[0], rectang[1], min(rectang[2], numpy.shape(self.data)[0]-rectang[0]), min(rectang[3], numpy.shape(self.data)[1]-rectang[1]))
        else:
            raise RuntimeError("rectang should either be a 2 or 4 tuple or None")

        #Now we can create X,Y and Z 2D arrays
        print rectang
        zdata=self.data[rectang[0]:rectang[0]+rectang[2], rectang[1]:rectang[1]+rectang[3]]
        xvals=xvals[0:rectang[2]]
        yvals=yvals[0:rectang[3]]
        xdata, ydata=numpy.meshgrid(xvals, yvals)

        #This is the same as for the plot_xyz routine'
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
        labels={"xlabel":(xlabel, "X Data"), "ylabel":(ylabel, "Y Data"), "zlabel":(zlabel, "Z Data")}
        for label in labels:
            (v, default)=labels[label]
            if v is None:
                if label in self.metadata:
                    labels[label]=self[label]
                else:
                    labels[label]=default
            else:
                labels[label]=v

        pyplot.xlabel(str(labels["xlabel"]))
        pyplot.ylabel(str(labels["ylabel"]))
        if plotter==self.__SurfPlotter:
            self.axes[0].set_zlabel(str(labels["zlabel"]))
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


