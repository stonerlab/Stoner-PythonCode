"""         Stoner.Plot 
            ============

Provides the a class to facilitate easier plotting of Stoner Data:

Classes:
    PlotFile - A class that uses matplotlib to plot data
"""
from Stoner.compat import *
from Stoner.Core import DataFile
from Stoner.PlotFormats import DefaultPlotStyle
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
        _setas (list): Used for automatically doing plots
        labels (list of string): List of axis labels as aternates to the column_headers
    
    """

    __figure=None
    _labels=None
    
    _template=None

    def __init__(self, *args, **kargs): #Do the import of pyplot here to speed module load
        """Constructor of \b PlotFile class. Imports pyplot and then calls the parent constructor

        """
        if "template" in kargs: #Setup the template
            self.template=kargs["template"]
            del(kargs["template"])
        else:
            self.template=DefaultPlotStyle
        super(PlotFile, self).__init__(*args, **kargs)
        self._setas=[]
        self._labels=self.column_headers
        self.legend=True
        self._subplots=[]

    def __dir__(self):
        """Handles the local attributes as well as the inherited ones"""
        attr=dir(type(self))
        attr.extend(super(PlotFile,self).__dir__())
        attr.extend(list(self.__dict__.keys()))
        attr.extend(["fig", "axes","labels","subplots","setas"])
        attr.extend(('xlabel','ylabel','title','subtitle','xlim','ylim'))
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
        elif name=="setas":
            return self._setas
        elif name=="labels":
            if len(self._labels)<len(self.column_headers):
                self._labels.extend(self.column_headers[len(self._labels):])
            return self._labels
        elif name=="subplots":
            if len(self.__figure.axes)>len(self._subplots):
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
            pyplot.figure(value.number)
        elif name=="setas":
            self._set_setas(value)
        elif name=="labels":
            self._labels=value
        elif name=="template":
            if isinstance(value,DefaultPlotStyle):
                self._template=value
            elif type(value)==type(object) and issubclass(value,DefaultPlotStyle):
                self._template=value()
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

    def _set_setas(self,value):
        """Handle the interpretation of the setas attribute. This includes parsing a string or a list
        that describes if the columns are to be used for x-y plotting."""
        if len(self._setas)<len(self.column_headers):
            self._setas.extend(list("."*(len(self.column_headers)-len(self._setas))))
        if isinstance(value,list):
            if len(value)> len(self.column_headers):
                value=value[:len(self.column_headers)]
            for v in value.lower():
                if v not in "xyzedf.":
                    raise ValueError("Set as column element is invalid: {}".format(v))
            self._setas[:len(value)]=[v.lower() for v in value]
        elif isinstance(value,string_types):
            value=value.lower()
            pattern=re.compile("^([0-9]*?)(x|y|z|d|e|f|\.)")
            i=0
            while pattern.match(value):
                res=pattern.match(value)
                (count,code)=res.groups()
                value=value[res.end():]
                if count=="":
                    count=1
                else:
                    count=int(count)
                for j in range(count):
                    self._setas[i]=code
                    i+=1
                    if i>=len(self.column_headers):
                        break
                if i>=len(self.column_headers):
                    break
        else:
            raise ValueError("Set as column string ended with a number")

    def _get_cols(self,startx=0):
        """Uses the setas attribute to work out which columns to use for x,y,z etc."""
        
        if len(self._setas)<len(self.column_headers):
            self._setas.extend(list("."*(len(self.column_headers)-len(self._setas))))
            
        xcol=self._setas[startx:].index("x")+startx
        try:
            maxcol=self._setas[xcol+1:].index("x")+xcol+1
        except ValueError:
            maxcol=len(self.column_headers)
        try:
            xerr=self._setas[startx:maxcol-startx].index("d")+startx
        except ValueError:
            xerr=None
        ycol=list()
        yerr=list()
        starty=startx
        has_yerr=False
        while True:
            try:
                ycol.append(self._setas[starty:maxcol-starty].index("y")+starty)
            except ValueError:
                break
            try:
                yerr.append(self._setas[starty:maxcol-starty].index("e")+starty)
                has_yerr=True
            except ValueError:
                yerr.append(None)
            starty=ycol[-1]+1
        zcol=list()
        zerr=list()
        startz=startx
        has_zerr=False
        while True:
            try:
                zcol.append(self._setas[startz:maxcol-startz].index("z")+startz)
            except ValueError:
                break
            startz=zcol[-1]+1
            try:
                zerr.append(self._setas[startz:maxcol-startz].index("g")+startz)
                has_zerr=True
            except ValueError:
                zerr.append(None)
        if len(zcol)==0:
            axes=2
        else:
            axes=3
        ret={"xcol":xcol,"xerr":xerr,"ycol":ycol,"yerr":yerr,"zcol":zcol,"zerr":zerr,"axes":axes}
        ret["has_xerr"]=xerr is not None
        ret["has_yerr"]=has_yerr
        ret["has_zerr"]=has_zerr
        return ret
        
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
        if plotter in (pyplot.plot,pyplot.semilogx,pyplot.semilogy,pyplot.loglog): #plots with positional fmt
            if fmt is None:
                plotter(x,y, figure=figure, **kwords)
            else:
                plotter(x,y, fmt, figure=figure, **kwords)
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
        cols=self._get_cols()
        if cols["axes"]==2:
            return self.plot_xy(**kargs)
        elif cols["axes"]==3:
            return self.plot_xyz(**kargs)
        else:
            raise RuntimeError("Unable to work out plot type !")


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
            cols=self._get_cols()
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
            figure=pyplot.figure(figure)
        elif isinstance(figure, bool) and not figure:
            figure=pyplot.figure()
        elif isinstance(figure, matplotlib.figure.Figure):
            figure=pyplot.figure(figure.number)
        elif isinstance(self.__figure,  matplotlib.figure.Figure):
            figure=self.__figure
        else:
            figure=pyplot.figure()
        
        pyplot.figure(figure.number)
        
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


        pyplot.xlabel(str(self._col_label(column_x)))
        pyplot.ylabel(str(ylabel))
        if title=='':
            title=self.filename
        pyplot.title(title)
        if self.legend:
            pyplot.legend()
        if save_filename != '':
            pyplot.savefig(str(save_filename))
        pyplot.draw()
        pyplot.show()
        self.__figure=figure
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
            cols=self._get_cols()
            if xcol is None:
                xcol=cols["xcol"]
            if ycol is None:
                ycol=cols["ycol"][0]
            if zcol is None:
                zcol=cols["zcol"][0]
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
        if "cmap" not in kwords:
            kwords["cmap"]=cm.jet
        plotter(xdata, ydata, zdata, **kwords)
        pyplot.xlabel(str(self._col_label(xcol)))
        pyplot.ylabel(str(self._col_label(ycol)))
        if plotter==self.__SurfPlotter:
            self.fig.axes[0].set_zlabel(str(self._col_label(zcol)))
        if title=='':
            title=self.filename
        pyplot.title(title)
        pyplot.draw()
        pyplot.show()

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

    def draw(self):
        """Pass through to pyplot to force figure redraw"""
        pyplot.figure(self.__figure.number)
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
            figure=pyplot.figure()
        elif isinstance(figure, int):
            figure=pyplot.figure(figure)
        elif isinstance(figure, matplotlib.figure.Figure):
            figure=pyplot.figure(figure.number)        
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
        pyplot.figure(self.__figure.number)
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
        


