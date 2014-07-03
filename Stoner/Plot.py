"""         Stoner.Plot
            ============

Provides the a class to facilitate easier plotting of Stoner Data:

Classes:
    PlotFile - A class that uses matplotlib to plot data
"""
from Stoner.compat import *
from Stoner.Core import DataFile
from Stoner.PlotFormats import DefaultPlotStyle
from Stoner.plotutils import errorfill
import numpy as _np_
import matplotlib
import os
import platform
import re
if os.name=="posix" and platform.system()=="Darwin":
    matplotlib.use('MacOSX')
from matplotlib import pyplot as pyplot
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as colors
from colorsys import hls_to_rgb

class PlotFile(DataFile):
    """Extends DataFile with plotting functions

    Args:
        args(tuple): Arguements to pass to :py:meth:`Stoner.Core.DataFile.__init__`
        kargs (dict):  keyword arguments to pass to \b DataFile.__init__

    Methods:
        plot_xy: Basic 2D plotting function
        plot_xyz: 3D plotting function
        griddata: Method to transform xyz data to a matrix
        contour_xyz: Plots x,y,z points as a contour plot
        plot_matrix: Plots a matrix as a 2D colour image plot
        draw: Pass throuygh to matplotlib draw
        show: Pass through to matploitlib show
        figure: Pass through to maplotlib figure.

    Attributes:
        fig (matplotlib.figure): The current figure object being worked with
        labels (list of string): List of axis labels as aternates to the column_headers

    """

    positional_fmt=[pyplot.plot,pyplot.semilogx,pyplot.semilogy,pyplot.loglog]
    no_fmt=[errorfill]

    def __init__(self, *args, **kargs): #Do the import of pyplot here to speed module load
        """Constructor of \b PlotFile class. Imports pyplot and then calls the parent constructor

        """
        if "template" in kargs: #Setup the template
            self.template=kargs["template"]
            del(kargs["template"])
        else:
            self.template=DefaultPlotStyle
        super(PlotFile, self).__init__(*args, **kargs)
        self.__figure=None
        self._labels=self.column_headers
        self.legend=True
        self._subplots=[]

    def __dir__(self):
        """Handles the local attributes as well as the inherited ones"""
        attr=dir(type(self))
        attr.extend(super(PlotFile,self).__dir__())
        attr.extend(list(self.__dict__.keys()))
        attr.extend(["fig", "axes","labels","subplots","template"])
        attr.extend(('xlabel','ylabel','title','xlim','ylim'))
        attr=list(set(attr))
        return sorted(attr)

    def __getattr__(self, name):
        """Attribute accessor

        Args:
            name (string):  Name of attribute the following attributes are supported:
                * fig - the current pyplot figure reference
                * axes - the pyplot axes object for the current plot
                * xlim - the X axis limits
                * ylim - the Y axis limits

                All other attrbiutes are passed over to the parent class
                """
        if name=="fig":
            return self.__figure
        elif name=="template":
            return self._template
        elif name=="labels":
            if len(self._labels)<len(self.column_headers):
                self._labels.extend(self.column_headers[len(self._labels):])
            return self._labels
        elif name=="subplots":
            if self.__figure is not None and len(self.__figure.axes)>len(self._subplots):
                self._subplots=self.__figure.axes
            return self._subplots
        elif name=="axes":
            if isinstance(self.__figure, matplotlib.figure.Figure):
                return self.__figure.axes
            else:
                return None
        elif name in ('xlim','ylim'):
            return pyplot.__dict__[name]()
        else:
            return super(PlotFile, self).__getattr__(name)

    def __setattr__(self, name, value):
        """Sets the specified attribute

        Args:
            name (string): The name of the attribute to set. The cuirrent attributes are supported:
                * fig - set the pyplot figure isntance to use
                * xlabel - set the X axis label text
                * ylabel - set the Y axis label text
                * title - set the plot title
                * subtitle - set the plot subtitle
                * xlim - set the x-axis limits
                * ylim - set the y-axis limits

            Only "fig" is supported in this class - everything else drops through to the parent class
            value (any): The value of the attribute to set.
    """
        if name=="fig":
            self.__figure=value
            self.template.new_figure(value.number)
        elif name=="labels":
            self._labels=value
        elif name=="template":
            if isinstance(value,DefaultPlotStyle):
                self._template=value
            elif type(value)==type(object) and issubclass(value,DefaultPlotStyle):
                self._template=value()
            else:
                raise ValueError("Template is not of the right class")
            self._template.apply()
        elif name in ('xlabel','ylabel','title','subtitle','xlim','ylim'):
            if isinstance(value,tuple):
                pyplot.__dict__[name](*value)
            elif isinstance(value,dict):
                pyplot.__dict__[name](**value)
            else:
                pyplot.__dict__[name](value)
        else:
            super(PlotFile, self).__setattr__(name, value)

    def _col_label(self,index):
        """Look up a column and see if it exists in self._lables, otherwise get from self.column_headers.

        Args:
            index (column index type): Column to return label for

        Returns:
            String type representing the column label.
        """
        ix=self.find_col(index)
        if isinstance(ix,list):
            return [self._col_label(i) for i in ix]
        else:
            if isinstance(self._labels,list) and len(self._labels)>ix:
                return self._labels[ix]
            else:
                return self.column_headers[ix]


    def _plot(self,ix,iy,fmt,plotter,figure,**kwords):
        """Private method for plotting a single plot to a figure.

        Args:
            ix (int): COlumn index of x data
            iy (int): Column index of y data
            fmt (str): Format of this plot
            plotter (callable): Routine used to plot this data
            figure (matplotlib.figure): Figure to plot data
            **kwords (dict): Other keyword arguments to pass through

        """

        if "label" not in kwords:
            kwords["label"]=self._col_label(iy)
        x=self.column(ix)
        y=self.column(iy)
        if plotter in self.positional_fmt: #plots with positional fmt
            if fmt is None:
                plotter(x,y, figure=figure, **kwords)
            else:
                plotter(x,y, fmt, figure=figure, **kwords)
        elif plotter in self.no_fmt:
            plotter(x,y,figure=figure, **kwords)
        else:
            if fmt is None:
                fmt="-"
            plotter(x,y, fmt=fmt,figure=figure, **kwords)
        for ax in figure.axes:
            self._template.customise_axes(ax)


    def plot(self,**kargs):
        """Try to make an appropriate plot based on the defined column assignments.

        The column assignments are examined to determine whether to plot and x,y plot or an x,y,z plot
        and whether to plot error bars (for an x,y plot). All keyword argume nts are passed through to
        the selected plotting routine.
        """
        for x in [i for i in range(len(self.setas)) if self.setas[i]=="x"]:
            cols=self._get_cols(startx=x)
            kargs["_startx"]=x
            if cols["axes"]==2:
                ret=self.plot_xy(**kargs)
            elif cols["axes"]==3:
                ret=self.plot_xyz(**kargs)
            elif cols["axes"]==6:
                ret=self.plot_xyzuvw(**kargs)

            else:
                raise RuntimeError("Unable to work out plot type !")
        return ret

    def plot_xy(self,column_x=None, column_y=None, fmt=None,show_plot=True,  title='', save_filename='', figure=None, plotter=None,  **kwords):
        """Makes a simple X-Y plot of the specified data.

            Args:
                column_x (index): Which column has the X-Data
                column_y (index): Which column(s) has(have) the y-data to plot

            Keyword Arguments:
                fmt (strong or sequence of strings): Specifies the format for the plot - see matplotlib documentation for details
                show_plot (bool): True Turns on interactive plot control
                title (string): Optional parameter that specfies the plot title - otherwise the current DataFile filename is used
                save_filename (string): Filename used to save the plot
                figure (matplotlib figure): Controls what matplotlib figure to use. Can be an integer, or a matplotlib.figure or False. If False then a new figure is always used, otherwise it will default to using the last figure used by this DataFile object.
                plotter (callable): Optional arguement that passes a plotting function into the routine. Sensible choices might be pyplot.plot (default), py.semilogy, pyplot.semilogx
                kwords (dict): A dictionary of other keyword arguments to pass into the plot function.

            Returns:
                A matplotlib.figure isntance

        """
        #This block sorts out the column_x and column_y where they are not defined in the signature
        # We return the first column marked as an 'X' column and then all the 'Y' columns between there
        # and the next 'X' column or the end of the file. If the yerr keyword is specified and is Ture
        # then  we look for an equal number of matching 'e' columns for the error bars.
        if column_x is None and column_y is None:
            if "_startx" in kwords:
                startx=kwords["_startx"]
                del kwords["_startx"]
            else:
                startx=0
            cols=self._get_cols(startx=startx)
            column_x=cols["xcol"]
            column_y=cols["ycol"]
            if "xerr" not in kwords and cols["has_xerr"]:
                kwords["xerr"]=cols["xerr"]
            if "yerr" not in kwords and cols["has_yerr"]:
                kwords["yerr"]=cols["yerr"]
        column_x=self.find_col(column_x)
        column_y=self.find_col(column_y)
        if "xerr" in kwords or "yerr" in kwords and plotter is None: # USe and errorbar blotter by default for errors
            plotter=pyplot.errorbar
        for err in ["xerr", "yerr"]:  # Check for x and y error keywords
            if err in kwords:
                if isinstance(kwords[err],index_types):
                    kwords[err]=self.column(kwords[err])
                elif isinstance(kwords[err], list) and isinstance(column_y,list) and len(kwords[err])==len(column_y):
                # Ok, so it's a list, so redo the check for each  item.
                    for i in range(len(kwords[err])):
                        if isinstance(kwords[err][i],index_types):
                            kwords[err][i]=self.column(kwords[err][i])
                        else:
                            kwords[err][i]=_np_.zeros(len(self))
                else:
                    kwords[err]=_np_.zeros(len(self))


        # Now try to process the figure parameter
        if isinstance(figure, int):
            figure=self.template.new_figure(figure)
        elif isinstance(figure, bool) and not figure:
            figure=self.template.new_figure(None)
        elif isinstance(figure, matplotlib.figure.Figure):
            figure=self.template.new_figure(figure.number)
        elif isinstance(self.__figure,  matplotlib.figure.Figure):
            figure=self.__figure
        else:
            figure=self.template.new_figure(None)

        self.__figure=figure
        if show_plot == True:
            pyplot.ion()
        if plotter is None: #Nothing has defined the plotter to use yet
            plotter=pyplot.plot
        if not isinstance(column_y, list):
            ylabel=self.labels[column_y]
            column_y=[column_y]
        else:
            ylabel=",".join([self.labels[ix] for ix in column_y])
        temp_kwords=kwords
        for ix in range(len(column_y)):
            if isinstance(fmt,list): # Fix up the format
                fmt_t=fmt[ix]
            else:
                fmt_t=fmt
            if "label" in kwords and isinstance(kwords["label"],list): # Fix label keywords
                temp_kwords["label"]=kwords["label"][ix]
            if "yerr" in kwords and isinstance(kwords["yerr"],list): # Fix yerr keywords
                temp_kwords["yerr"]=kwords["yerr"][ix]
            # Call plot
            self._plot(column_x,column_y[ix],fmt_t,plotter,figure,**temp_kwords)

        xlabel=str(self._col_label(column_x))
        if title=='':
            title=self.filename
        self._template.annotate(self,xlabel=xlabel,ylabel=ylabel,title=title)
        if save_filename != '':
            pyplot.savefig(str(save_filename))
        pyplot.draw()
        pyplot.show()
        return self.__figure

    def plot_xyz(self, xcol=None, ycol=None, zcol=None, shape=None, xlim=None, ylim=None,show_plot=True,  title='', figure=None, plotter=None,  **kwords):
        """Plots a surface plot based on rows of X,Y,Z data using matplotlib.pcolor()

            Args:
                xcol (index): Xcolumn index or label
                ycol (index): Y column index or label
                zcol (index): Z column index or label

            Keyword Arguments:
                shape (tuple): Defines the shape of the surface (i.e. the number of X and Y value. If not procided or None, then the routine will attempt to calculate these from the data provided
                xlim (tuple): Defines the x-axis limits and grid of the data to be plotted
                ylim (tuple) Defines the Y-axis limits and grid of the data data to be plotted
                cmap (matplotlib colour map): Surface colour map - defaults to the jet colour map
                show_plot (bool): True Turns on interactive plot control
                title (string): Optional parameter that specfies the plot title - otherwise the current DataFile filename is used
                save_filename (string): Filename used to save the plot
                figure (matplotlib figure): Controls what matplotlib figure to use. Can be an integer, or a matplotlib.figure or False. If False then a new figure is always used, otherwise it will default to using the last figure used by this DataFile object.
                plotter (callable): Optional arguement that passes a plotting function into the routine. Sensible choices might be pyplot.plot (default), py.semilogy, pyplot.semilogx
                kwords (dict): A dictionary of other keyword arguments to pass into the plot function.

            Returns:
                A matplotlib.figure isntance
        """
        if None in (xcol,ycol,zcol):
            if "_startx" in kwords:
                startx=kwords["_startx"]
                del kwords["_startx"]
            else:
                startx=0
            cols=self._get_cols(startx=startx)
            if xcol is None:
                xcol=cols["xcol"]
            if ycol is None:
                ycol=cols["ycol"][0]
            if zcol is None:
                zcol=cols["zcol"][0]
        xdata,ydata,zdata=self.griddata(xcol,ycol,zcol,shape=shape,xlim=xlim,ylim=ylim)
        if isinstance(figure, int):
            figure=self.template.new_figure(figure)
        elif isinstance(figure, bool) and not figure:
            figure=self.template.new_figure(None)
        elif isinstance(figure, matplotlib.figure.Figure):
            figure=self.template.new_figure(figure.number)
        elif isinstance(self.__figure,  matplotlib.figure.Figure):
            figure=self.__figure
        else:
            figure=self.template.new_figure(None)
        self.__figure=figure
        if show_plot == True:
            pyplot.ion()
        if plotter is None:
            plotter=self.__SurfPlotter
        if "cmap" not in kwords:
            kwords["cmap"]=cm.jet
        plotter(xdata, ydata, zdata, **kwords)
        params={}
        params["xlabel"]=str(self._col_label(xcol))
        params["ylabel"]=str(self._col_label(ycol))
        params["zlabel"]=str(self._col_label(zcol))
        if title=='':
            title=self.filename
        params["title"]=title
        if plotter is not self.__SurfPlotter:
            del(params["zlabel"])
        self._template.annotate(self,**params)
        pyplot.draw()
        pyplot.show()

        return self.__figure

    def plot_xyzuvw(self, xcol=None, ycol=None, zcol=None, ucol=None,vcol=None,wcol=None, figure=None, plotter=None,  **kwords):
        """Plots a vector field plot based on rows of X,Y,Z (U,V,W) data using ,ayavi

            Args:
                xcol (index): Xcolumn index or label
                ycol (index): Y column index or label
                zcol (index): Z column index or label
                ucol (index): U column index or label
                vcol (index): V column index or label
                wcol (index): W column index or label

            Keyword Arguments:
                colormap (string): Vector field colour map - defaults to the jet colour map
                colors (column index or numpy array): Values used to map the colors of the resultant file.
                figure (mlab figure): Controls what mlab figure to use. Can be an integer, or a mlab.figure or False. If False then a new figure is always used, otherwise it will default to using the last figure used by this DataFile object.
                plotter (callable): Optional arguement that passes a plotting function into the routine. Sensible choices might be pyplot.plot (default), py.semilogy, pyplot.semilogx
                kwords (dict): A dictionary of other keyword arguments to pass into the plot function.

            Returns:
                A mayavi scene instance
        """
        try:
            from mayavi import mlab,core
        except ImportError:
            return None
        if None in (xcol,ycol,zcol):
            if "_startx" in kwords:
                startx=kwords["_startx"]
                del kwords["_startx"]
            else:
                startx=0
            cols=self._get_cols(startx=startx)
            if xcol is None:
                xcol=cols["xcol"]
            if ycol is None:
                ycol=cols["ycol"][0]
            if zcol is None:
                zcol=cols["zcol"][0]
            if ucol is None:
                ucol=cols["ucol"][0]
            if vcol is None:
                vcol=cols["vcol"][0]
            if wcol is None:
                wcol=cols["wcol"][0]
            if "colors" in kwords:
                colors=kwords["colors"]
                del kwords["colors"]
                if isinstance(colors,bool) and colors:
                    colors=colors
                elif isinstance(colors,index_types):
                    colors=self.column(colors)
                elif isinstance(colors,_np_.ndarray):
                    colors=colors
                elif callable(colors):
                    colors=_np_.array([colors(x) for x in self.rows()])
                else:
                    raise RuntimeError("Do not recognise what to do with the colors keyword.")
                kwords["scalars"]=colors
        if isinstance(figure, int):
            figure=mlab.figure(figure)
        elif isinstance(figure, bool) and not figure:
            figure=mlab.figure(bgcolor=(1,1,1))
        elif isinstance(figure, core.scene.Scene):
            pass
        elif isinstance(self.__figure,  core.scene.Scene):
            figure=self.__figure
        else:
            figure=mlab.figure(bgcolor=(1,1,1))
        self.__figure=figure
        if plotter is None:
            plotter=self._VectorFieldPlot
        plotter(self.column(xcol), self.column(ycol), self.column(zcol),self.column(ucol),self.column(vcol),self.column(wcol), **kwords)
        mlab.show()
        return self.__figure


    def griddata(self,xc,yc=None,zc=None,shape=None,xlim=None,ylim=None,method="linear"):
        """Function to convert xyz data onto a regular grid

            Args:
                xc (index): Column to be used for the X-Data
                yc (index): column to be used for Y-Data - default value is column to the right of the x-data column
                zc (index): column to be used for the Z-data - default value is the column to the right of the y-data column

            Keyword Arguments:
                shaoe (two-tuple): Number of points along x and y in the grid - defaults to a square of sidelength = square root of the length of the data.
                xlim (tuple): The xlimits
                ylim (tuple) The ylimits
                method (string): Type of interploation to use, default is linear

            ReturnsL
                X,Y,Z three two dimensional arrays of the co-ordinates of the interpolated data
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
            shape=(_np_.floor(_np_.sqrt(len(self))),_np_.floor(_np_.sqrt(len(self))))
        if xlim is None:
            xlim=(_np_.min(self.column(xc))*(shape[0]-1)/shape[0],_np_.max(self.column(xc))*(shape[0]-1)/shape[0])
        if isinstance(xlim,tuple) and len(xlim)==2:
            xlim=(xlim[0],xlim[1],(xlim[1]-xlim[0])/shape[0])
        elif isinstance(xlim,tuple) and len(xlim)==3:
            xlim[2]=len(range(*xlim))
        else:
            raise RuntimeError("X limit specification not good.")
        if ylim is None:
            ylim=(_np_.min(self.column(yc))*(shape[1]-1)/shape[1],_np_.max(self.column(yc))*(shape[0]-1)/shape[0])
        if isinstance(ylim,tuple) and len(ylim)==2:
            ylim=(ylim[0],ylim[1],(ylim[1]-ylim[0])/shape[1])
        elif isinstance(ylim,tuple) and len(ylim)==3:
            ylim[2]=len(range(*ylim))
        else:
            raise RuntimeError("Y limit specification not good.")

        np=_np_.mgrid[slice(*xlim),slice(*ylim)].T

        points=_np_.array([self.column(xc),self.column(yc)]).T
        Z=griddata(points,self.column(zc),np,method=method)
        return np[:,:,0],np[:,:,1],Z

    def contour_xyz(self,xc,yc,zc,shape=None,xlim=None, ylim=None, plotter=None,**kargs):
        """Grid up the three columns of data and plot

        Args:
            xc (index): Column to be used for the X-Data
            yc (index): column to be used for Y-Data - default value is column to the right of the x-data column
            zc (index): column to be used for the Z-data - default value is the column to the right of the y-data column

        Keyword Arguments:
            shaoe (two-tuple): Number of points along x and y in the grid - defaults to a square of sidelength = square root of the length of the data.
            xlim (tuple): The xlimits
            ylim (tuple) The ylimits

        Returns:
            A matplotlib figure
         """

        X,Y,Z=self.griddata(xc,yc,zc,shape,xlim,ylim)
        if plotter is None:
            plotter=pyplot.contour

        fig=plotter(X,Y,Z,**kargs)
        pyplot.xlabel(self._col_label(xc))
        pyplot.ylabel(self._col_label(yc))
        pyplot.title(self.filename + " "+ self._col_label(zc))

        return fig




    def plot_matrix(self, xvals=None, yvals=None, rectang=None, cmap=pyplot.cm.jet,show_plot=True,  title='',xlabel=None, ylabel=None, zlabel=None,  figure=None, plotter=None,  **kwords):
        """Plots a surface plot by assuming that the current dataset represents a regular matrix of points.

            Args:
                xvals (index, list or numpy.array): Either a column index or name or a list or numpytarray of column values. The default (None) uses the first column of data
                yvals (int or list): Either a row index or a list or numpy array of row values. The default (None) uses the column_headings interpreted as floats
                rectang (tuple):  a tuple of either 2 or 4 elements representing either the origin (row,column) or size (origin, number of rows, number of columns) of data to be used for the z0data matrix

            Keyword Arguments:
                cmap (matplotlib colour map): Surface colour map - defaults to the jet colour map
                show_plot (bool): True Turns on interactive plot control
                title (string): Optional parameter that specfies the plot title - otherwise the current DataFile filename is used
                xlabel (string) X axes label. Deafult is None - guess from xvals or metadata
                ylabel (string): Y axes label, Default is None - guess from metadata
                zlabel (string): Z axis label, Default is None - guess from metadata
                figure (matplotlib figure): Controls what matplotlib figure to use. Can be an integer, or a matplotlib.figure or False. If False then a new figure is always used, otherwise it will default to using the last figure used by this DataFile object.
                plotter (callable): Optional arguement that passes a plotting function into the routine. Sensible choices might be pyplot.plot (default), py.semilogy, pyplot.semilogx
                kwords (dict): A dictionary of other keyword arguments to pass into the plot function.

            Returns:
                The matplotib figure with the data plotted"""
        # Sortout yvals values
        if isinstance(yvals, int): #  Int means we're sepcifying a data row
            if rectang is None: # we need to intitialise the rectang
                rectang=(yvals+1, 0) # We'll sort the column origin later
            elif isinstance(rectang, tuple) and rectang[1]<=yvals: # We have a rectang, but we need to adjust the row origin
                rectang[0]=yvals+1
            yvals=self[yvals] # change the yvals into a numpy array
        elif isinstance(yvals,(list, tuple)): # We're given the yvals as a list already
            yvals=_np_.array(yvals)
        elif yvals is None: # No yvals, so we'l try column headings
            if isinstance(xvals, index_types): # Do we have an xcolumn header to take away ?
                xvals=self.find_col(xvals)
                headers=self.column_headers[xvals+1:]
            elif xvals is None: # No xvals so we're going to be using the first column
                xvals=0
                headers=self.column_headers[1:]
            else:
                headers=self.column_headers
            yvals=_np_.array([float(x) for x in headers]) #Ok try to construct yvals aray
        else:
            raise RuntimeError("uvals must be either an integer, list, tuple, numpy array or None")
        #Sort out xvls values
        if isinstance(xvals, index_types): # String or int means using a column index
            if xlabel is None:
                xlabel=self._col_clabel(xvals)
            if rectang is None: # Do we need to init the rectan ?
                rectang=(0, xvals+1)
            elif isinstance(rectang, tuple): # Do we need to adjust the rectan column origin ?
                rectang[1]=xvals+1
            xvals=self.column(xvals)
        elif isinstance(xvals, (list,tuple)): # Xvals as a data item
            xvals=_np_.array(xvals)
        elif isinstance(xvals, _np_.ndarray):
            pass
        elif xvals is None: # xvals from column 0
            xvals=self.column(0)
            if rectang is None: # and fix up rectang
                rectang=(0, 1)
        else:
            raise RuntimeError("xvals must be a string, integer, list, tuple or numpy array or None")

        if isinstance(rectang, tuple) and len(rectang)==2: # Sort the rectang value
            rectang=(rectang[0], rectang[1], _np_.shape(self.data)[0]-rectang[0], _np_.shape(self.data)[1]-rectang[1])
        elif rectang is None:
            rectang=(0, 0, _np_.shape(self.data)[0], _np_.shape(self.data)[1])
        elif isinstance(rectang, tuple) and len(rectang)==4: # Ok, just make sure we have enough data points left.
            rectang=(rectang[0], rectang[1], min(rectang[2], _np_.shape(self.data)[0]-rectang[0]), min(rectang[3], _np_.shape(self.data)[1]-rectang[1]))
        else:
            raise RuntimeError("rectang should either be a 2 or 4 tuple or None")

        #Now we can create X,Y and Z 2D arrays
        zdata=self.data[rectang[0]:rectang[0]+rectang[2], rectang[1]:rectang[1]+rectang[3]]
        xvals=xvals[0:rectang[2]]
        yvals=yvals[0:rectang[3]]
        xdata, ydata=_np_.meshgrid(xvals, yvals)

        #This is the same as for the plot_xyz routine'
        if isinstance(figure, int):
            figure=self.template.new_figure(figure)
        elif isinstance(figure, bool) and not figure:
            figure=self.template.new_figure(None)
        elif isinstance(figure, matplotlib.figure.Figure):
            figure=self.template.new_figure(figure.number)
        elif isinstance(self.__figure,  matplotlib.figure.Figure):
            figure=self.__figure
        else:
            figure=self.template.new_figure(None)
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

        Args:
            X data
            Y Y data
            Z data
            kargs (dict): Other keywords to pass through

        ReturnsL
            A matplotib Figure

        This function attempts to work the same as the 2D surface plotter pcolor, but draws a 3D axes set"""
        Z=_np_.nan_to_num(Z)
        ax = self.fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, **kargs)
        self.fig.colorbar(surf, shrink=0.5, aspect=5,extend="both")

        return surf


    def _VectorFieldPlot(self,X,Y,Z,U,V,W,**kargs):
        """Helper function to plot vector fields using mayavi.mlab

        Args:
            X (array): X data co-ordinates
            Y (array): Y data co-ordinates
            Z (array): Z data co-ordinates
            U (array): U data vector field component
            V (array): V data vector field component
            W (array): W data vector field component

        Returns:
            An mlab figure reference.
            """
        try:
            from mayavi import mlab # might not work !
            from tvtk.api import tvtk
        except ImportError:
            return None
        if "scalars" in kargs:
            col_mode="color_by_scalar"
        else:
            col_mode="color_by_vector"
        if isinstance(kargs["scalars"],bool) and kargs["scalars"]: # fancy mode on
            del kargs["scalars"]
            pi=_np_.pi
            colors=hsl2rgb((1+self.q/pi)/2,self.r/_np_.max(self.r),(1+self.w)/2)
            quiv=mlab.quiver3d(X,Y,Z,U,V,W,scalars=_np_.ones(len(self)),**kargs)
            quiv.glyph.color_mode=col_mode
            sc=tvtk.UnsignedCharArray()
            sc.from_array(colors)
            quiv.mlab_source.dataset.point_data.scalars=sc
            quiv.mlab_source.dataset.modified()
        else:
            quiv=mlab.quiver3d(X,Y,Z,U,V,W,**kargs)
            quiv.glyph.color_mode=col_mode
        return quiv

    def draw(self):
        """Pass through to pyplot to force figure redraw"""
        self.template.new_figure(self.__figure.number)
        pyplot.draw()

    def show(self):
        """Pass through for pyplot Figure.show()"""
        self.fig.show()

    def figure(self, figure=None):
        """Set the figure used by :py:class:`Stoner.Plot.PlotFile`

        Args:
            figure A matplotlib figure or figure number

        Returns:
            The current \b Stoner.PlotFile instance"""
        if figure is None:
            figure=self.template.new_figure(None)
        elif isinstance(figure, int):
            figure=self.template.new_figure(figure)
        elif isinstance(figure, matplotlib.figure.Figure):
            figure=self.template.new_figure(figure.number)
        self.__figure=figure
        return self

    def subplot(self,*args,**kargs):
        """Pass throuygh for pyplot.subplot()

        Args:
            rows (int): If this is the only argument, then a three digit number representing
                the rows,columns,index arguments. If seperate rows, column and index are provided,
                then this is the number of rows of sub-plots in one figure.
            columns (int): The number of columns of sub-plots in one figure.
            index (int): Index (1 based) of the current sub-plot.

        Returns:
            A matplotlib.Axes instance representing the current sub-plot

        As well as passing through to the plyplot routine of the same name, this
        function maintains a list of the current sub-plot axes via the subplots attribute.
        """
        self.template.new_figure(self.__figure.number)
        sp=pyplot.subplot(*args,**kargs)
        if len(args)==1:
            rows=args[0]//100
            cols=(args[0]//10)%10
            index=args[0]%10
        else:
            rows=args[0]
            cols=args[1]
            index=args[2]
        if len(self._subplots)<rows*cols:
            self.subplots.extend([None for i in range(rows*cols-len(self._subplots))])
        self._subplots[index-1]=sp
        return sp

def hsl2rgb(h,s,l):
    """Converts from hsl colourspace to rgb colour space with numpy arrays for speed

    Args:
        h (array): Hue value
        s (array): Saturation value
        l (array): Luminence value

    Returns:
        2D array (Mx3) of unsigned 8bit integers
        """
    w=_np_.where
    if isinstance(h,float):
        h=_np_.array([h])
    if isinstance(s,float):
        s=_np_.array([s])
    if isinstance(l,float):
        l=_np_.array([l])

    if h.shape!=l.shape or h.shape!=s.shape:
        raise RuntimeError("Must have equal shaped arrays for h, s and l")


    rgb=_np_.zeros((len(h),3))
    hls=_np_.column_stack([h,l,s])
    for i in range(len(h)):
        rgb[i,:]=_np_.array(hls_to_rgb(*hls[i]))
    return (255*rgb).astype('u1')

