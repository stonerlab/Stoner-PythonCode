"""         Stoner.Plot
            ============

Provides the a class to facilitate easier plotting of Stoner Data:

Classes:
    PlotFile - A class that uses matplotlib to plot data
"""
from Stoner.compat import *
from Stoner.Core import DataFile, _attribute_store, copy_into,isNone,all_type
from Stoner.PlotFormats import DefaultPlotStyle
from Stoner.plotutils import errorfill
import numpy as _np_
import os
import platform
from inspect import getargspec
if os.name == "posix" and platform.system() == "Darwin":
    import matplotlib
    matplotlib.use('MacOSX')
from matplotlib import pyplot as plt
from matplotlib import figure as mplfig
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid import inset_locator
from matplotlib import cm
import matplotlib.colors as colors
from colorsys import hls_to_rgb
import copy
from collections import Iterable
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

class PlotFile(DataFile):
    """Extends DataFile with plotting functions.

    Args:
        args(tuple): Arguements to pass to :py:meth:`Stoner.Core.DataFile.__init__`
        kargs (dict):  keyword arguments to pass to \b DataFile.__init__

    Attributes:
        fig (matplotlib.figure): The current figure object being worked with
        labels (list of string): List of axis labels as aternates to the column_headers

    """

    positional_fmt = [plt.plot, plt.semilogx, plt.semilogy, plt.loglog]
    no_fmt = [errorfill]

    def __init__(self, *args, **kargs):  #Do the import of plt here to speed module load
        """Constructor of \b PlotFile class. Imports plt and then calls the parent constructor.

        """
        if "template" in kargs:  #Setup the template
            self.template = kargs["template"]
            del (kargs["template"])
        else:
            self.template = DefaultPlotStyle
        super(PlotFile, self).__init__(*args, **kargs)
        self.__figure = None
        self._labels = copy.deepcopy(self.column_headers)
        self.legend = True
        self._subplots = []
        self.multiple = "common"

#============================================================================================================================
#Properties of PlotFile
#============================================================================================================================
    @property
    def _public_attrs(self):
        ret = super(PlotFile, self)._public_attrs
        ret.update({
            "fig": (int, mplfig.Figure),
            "labels": list,
            "template": DefaultPlotStyle,
            "xlim": tuple,
            "ylim": tuple,
            "title": string_types,
            "xlabel": string_types,
            "ylabel": string_types
        })
        return ret

    @property
    def ax(self):
        return self.axes.index(self.fig.gca())

    @ax.setter
    def ax(self,value):
        if isinstance(value, int) and 0 <= value < len(self.axes):
            self.fig.sca(self.axes[value])
        else:
            raise IndexError("Figure doesn't have enough axes")

    @property
    def axes(self):
        if isinstance(self.__figure, mplfig.Figure):
            ret = self.__figure.axes
        else:
            ret = None
        return ret

    @DataFile.clone.getter
    def clone(self):
        c = self.__class__()
        ret = copy_into(self,c)
        ret.template = copy.deepcopy(self.template)
        ret.labels=self.labels
        return ret

    @property
    def column_headers(self):
        return DataFile.column_headers.fget(self)

    @column_headers.setter
    def column_headers(self,value):
        if all_type(value,string_types):
            DataFile.column_headers.fset(self,value)
            self.labels = value
        else:
            raise NotImplementedError("Column headers should be an iterable of strings.")

    @property
    def fig(self):
        return self.__figure

    @fig.setter
    def fig(self,value):
        if isinstance(value,plt.Figure):
            self.__figure = value
            self.__figure, ax = self.template.new_figure(value.number)
        elif isinstance(value.int):
            value=plt.Figure(value)
            self.fig=value
        else:
            raise NotImplementedError("fig should be a number of matplotlib figure")

    @property
    def fignum(self):
        return self.__figure.number

    @property
    def labels(self):
        if len(self._labels) < len(self.column_headers):
            self._labels.extend(copy.deepcopy(self.column_headers[len(self._labels):]))
        return self._labels

    @labels.setter
    def labels(self,value):
        self._labels = value

    @property
    def subplots(self):
        if self.__figure is not None and len(self.__figure.axes) > len(self._subplots):
            self._subplots = self.__figure.axes
        return self._subplots


    @property
    def template(self):
        return self._template

    @template.setter
    def template(self,value):
        if type(value) == type(object) and issubclass(value, DefaultPlotStyle):
            value = value()
        if isinstance(value, DefaultPlotStyle):
            self._template = value
        else:
            raise ValueError("Template is not of the right class")
        self._template.apply()




    def _Plot(self, ix, iy, fmt, plotter, figure, **kwords):
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
            kwords["label"] = self._col_label(iy)
        x = self.column(ix)
        y = self.column(iy)
        if plotter in self.positional_fmt:  #plots with positional fmt
            if fmt is None:
                plotter(x, y, figure=figure, **kwords)
            else:
                plotter(x, y, fmt, figure=figure, **kwords)
        elif plotter in self.no_fmt:
            plotter(x, y, figure=figure, **kwords)
        else:
            if fmt is None:
                fmt = "-"
            plotter(x, y, fmt=fmt, figure=figure, **kwords)
        for ax in figure.axes:
            self._template.customise_axes(ax, self)

    def __SurfPlotter(self, X, Y, Z, **kargs):
        """Utility private function to plot a 3D color mapped surface.

        Args:
            X data
            Y Y data
            Z data
            kargs (dict): Other keywords to pass through

        ReturnsL
            A matplotib Figure

        This function attempts to work the same as the 2D surface plotter pcolor, but draws a 3D axes set"""
        ax=plt.gca(projection="3d")
        Z = _np_.nan_to_num(Z)
        surf = ax.plot_surface(X, Y, Z, **kargs)
        self.fig.colorbar(surf, shrink=0.5, aspect=5, extend="both")

        return surf

    def _VectorFieldPlot(self, X, Y, Z, U, V, W, **kargs):
        """Helper function to plot vector fields using mayavi.mlab.

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
            from mayavi import mlab  # might not work !
            from tvtk.api import tvtk
        except ImportError:
            return None
        if "scalars" in kargs:
            col_mode = "color_by_scalar"
        else:
            col_mode = "color_by_vector"
        if "scalars" in kargs and isinstance(kargs["scalars"], bool) and kargs["scalars"]:  # fancy mode on
            del kargs["scalars"]
            pi = _np_.pi
            colors = hsl2rgb((1 + self.q / pi) / 2, self.r / _np_.max(self.r), (1 + self.w) / 2)
            quiv = mlab.quiver3d(X, Y, Z, U, V, W, scalars=_np_.ones(len(self)), **kargs)
            quiv.glyph.color_mode = col_mode
            sc = tvtk.UnsignedCharArray()
            sc.from_array(colors)
            quiv.mlab_source.dataset.point_data.scalars = sc
            quiv.mlab_source.dataset.modified()
        else:
            quiv = mlab.quiver3d(X, Y, Z, U, V, W, **kargs)
            quiv.glyph.color_mode = col_mode
        return quiv

    def _col_label(self, index, join=False):
        """Look up a column and see if it exists in self._lables, otherwise get from self.column_headers.

        Args:
            index (column index type): Column to return label for

        Returns:
            String type representing the column label.
        """
        ix = self.find_col(index)
        if isinstance(ix, list):
            if join:
                return ",".join([self._col_label(i) for i in ix])
            else:
                return [self._col_label(i) for i in ix]
        else:
            if isinstance(self._labels, list) and len(self._labels) > ix:
                return self._labels[ix]
            else:
                return self.column_headers[ix]

    def __dir__(self):
        """Handles the local attributes as well as the inherited ones."""
        attr = dir(type(self))
        attr.extend(super(PlotFile, self).__dir__())
        attr.extend(list(self.__dict__.keys()))
        attr.extend(["fig", "axes", "labels", "subplots", "template"])
        attr.extend(('xlabel', 'ylabel', 'title', 'xlim', 'ylim'))
        attr = list(set(attr))
        return sorted(attr)

    def _fix_cols(self, **kargs):
        """Sorts out axis specs, replacing with contents from setas as necessary."""
        if "startx" in kargs:
            startx = kargs["startx"]
            del kargs["startx"]
        else:
            startx = 0
        if "multi_y" not in kargs:
            kargs["multi_y"] = False
        c = self.setas._get_cols(startx=startx)
        for k in ["xcol", "ycol", "zcol", "ucol", "vcol", "wcol", "xerr", "yerr", "zerr"]:
            if k in kargs and k == "xcol" and kargs["xcol"] is None:
                kargs["xcol"] = c.xcol
            elif k in kargs and k == "xerr" and kargs["xerr"] is None:
                kargs["xerr"] = c.xerr
            elif k in kargs and k != "xcol" and kargs[k] is None and len(c[k]) > 0:
                if kargs["multi_y"]:
                    kargs[k] = c[k]
                else:
                    kargs[k] = c[k][0]
        for k in list(kargs.keys()):
            if k not in ["xcol", "ycol", "zcol", "ucol", "vcol", "wcol", "xerr", "yerr", "zerr"]:
                del kargs[k]
        return _attribute_store(kargs)

    def _fix_fig(self, figure, **kargs):
        """Sorts out the matplotlib figure handling."""
        if isinstance(figure, int):
            figure, ax = self.template.new_figure(figure, **kargs)
        elif isinstance(figure, bool) and not figure:
            figure, ax = self.template.new_figure(None, **kargs)
        elif isinstance(figure, mplfig.Figure):
            figure, ax = self.template.new_figure(figure.number, **kargs)
        elif isinstance(self.__figure, mplfig.Figure):
            figure = self.__figure
            ax = self.__figure.gca(**kargs)
        else:
            figure, ax = self.template.new_figure(None, **kargs)
        return figure, ax

    def _fix_kargs(self, function=None, defaults=None, otherkargs=None, **kargs):
        """Fix parameters to the plotting function to provide defaults and no extransous arguments.

        Returns:
            dictionary of correct arguments, dictionary of all arguments"""
        if defaults is None:
            defaults = dict()
        defaults.update(kargs)
        # Defaults now a dictionary of default arugments overlaid with keyword argument values
        # Now inspect the plotting function to see what it takes.
        if function is None:
            function = defaults["plotter"]
            if isinstance(function,string_types):
                if "projection" in kargs:
                    projection=kargs["projection"]
                else:
                    projection="rectilinear"
                function=plt.gca(projection=projection).__getattribute__(function)
                if self.__figure is not plt.gcf():
                    plt.close(plt.gcf())

        (args, vargs, kwargs, defs) = getargspec(function)
        # Manually overide the list of arguments that the plotting function takes if it takes keyword dictionary
        if isinstance(otherkargs, (list, tuple)) and kwargs is not None:
            args.extend(otherkargs)
        nonkargs = dict()
        for k in list(defaults.keys()):
            nonkargs[k] = defaults[k]
            if k not in args:
                del defaults[k]
        return defaults, nonkargs

    def _fix_titles(self, ix, multiple, **kargs):
        """Does the titling and labelling for a matplotlib plot."""
        self._template.annotate(ix, multiple, self, **kargs)
        if "show_plot" in kargs and kargs["show_plot"]:
            plt.ion()
            plt.draw()
            plt.show()
        if "save_filename" in kargs and kargs["save_filename"] is not None:
            plt.savefig(str(nonkargs["save_filename"]))

    def __getattr__(self, name):
        """Attribute accessor.

        Args:
            name (string):  Name of attribute the following attributes are supported:
                * fig - the current plt figure reference
                * axes - the plt axes object for the current plot
                * xlim - the X axis limits
                * ylim - the Y axis limits

                All other attrbiutes are passed over to the parent class
                """
        try:
            return super(PlotFile, self).__getattr__(name)
        except AttributeError:
            if not isinstance(self.__figure, mplfig.Figure):
                raise AttributeError("Unknown attribute {}".format(name))
            ax = self.__figure.axes
            if "get_{}".format(name) in dir(ax):
                func = ax.__getattribute__("get_{}".format(name))
                ret = func()
            elif name in plt.__dict__:  # Sort of a universal pass through to plt
                ret = plt.__dict__[name]
            elif name in dir(ax):  # Sort of a universal pass through to plt
                ret = ax.__getattribute__(name)
            else:
                raise AttributeError("Unknown attribute {}".format(name))
        return ret

    def __setattr__(self, name, value):
        """Sets the specified attribute.

        Args:
            name (string): The name of the attribute to set. The cuirrent attributes are supported:
                * fig - set the plt figure isntance to use
                * xlabel - set the X axis label text
                * ylabel - set the Y axis label text
                * title - set the plot title
                * subtitle - set the plot subtitle
                * xlim - set the x-axis limits
                * ylim - set the y-axis limits

            Only "fig" is supported in this class - everything else drops through to the parent class
            value (any): The value of the attribute to set.
    """
        if hasattr(type(self),name) and isinstance(getattr(type(self),name),property):
            object.__setattr__(self,name, value)
        elif name in dir(super(PlotFile, self)):
            super(PlotFile, self).__setattr__(name, value)
        elif "set_{}".format(name) in dir(plt.Axes):
            tfig = plt.gcf()
            tax = tfig.gca()  # protect the current axes and figure
            ax = self.fig.gca()
            if not isinstance(value, Iterable) or isinstance(value, string_types):
                value = (value, )
            func = ax.__getattribute__("set_{}".format(name))
            if isinstance(value, dict):
                func(**value)
            else:
                func(*value)
            plt.figure(tfig.number)
            plt.sca(tax)
        else:
            super(PlotFile, self).__setattr__(name, value)

    def add_column(self, column_data, header=None, index=None, func_args=None, replace=False):
        """Appends a column of data or inserts a column to a datafile instance.

        Args:
            column_data (:py:class:`numpy.array` or list or callable): Data to append or insert or a callable function that will generate new data

        Keyword Arguments:
            column_header (string): The text to set the column header to,
                if not supplied then defaults to 'col#'
            index (int or string): The  index (numeric or string) to insert (or replace) the data
            func_args (dict): If column_data is a callable object, then this argument
                can be used to supply a dictionary of function arguments to the callable object.
            replace (bool): Replace the data or insert the data (default)

        Returns:
            A :py:class:`DataFile` instance with the additonal column inserted.

        Note:
            Like most :py:class:`DataFile` methods, this method operates in-place in that it also modifies
            the original DataFile Instance as well as returning it."""

        # Call the parent method and then update this label
        super(PlotFile,self).add_column(column_data,header=header,index=index,func_args=func_args,replace=replace)
        #Mostly this is duplicating the parent method
        if index is None:
            index = len(self.column_headers)-1
        else:
            index = self.find_col(index)

        self.labels[index]=header
        return self


    def colormap_xyz(self, xcol=None, ycol=None, zcol=None, shape=None, xlim=None, ylim=None, plotter=None, **kargs):
        """An xyz plot that forces the use of plt.contour.

        Args:
            xcol (index): Xcolumn index or label
            ycol (index): Y column index or label
            zcol (index): Z column index or label

        Keyword Arguments:
            shape (two-tuple): Number of points along x and y in the grid - defaults to a square of sidelength = square root of the length of the data.
            xlim (tuple): The xlimits, defaults to automatically determined from data
            ylim (tuple): The ylimits, defaults to automatically determined from data
            plotter (function): Function to use to plot data. Defaults to plt.contour
            show_plot (bool): Turn on interfactive plotting and show plot when drawn
            save_filename (string or None): If set to a string, save the plot with this filename
            figure (integer or matplotlib.figure or boolean): Controls which figure is used for the plot, or if a new figure is opened.
            **kargs (dict): Other arguments are passed on to the plotter.

        Returns:
            A matplotlib figure
         """
        if plotter is None:
            plotter = plt.pcolor
        kargs["plotter"] = plotter
        kargs["projection"] = "rectilinear"
        return self.plot_xyz(xcol, ycol, zcol, shape, xlim, ylim, **kargs)

    def contour_xyz(self, xcol=None, ycol=None, zcol=None, shape=None, xlim=None, ylim=None, plotter=None, **kargs):
        """An xyz plot that forces the use of plt.contour.

        Args:
            xcol (index): Xcolumn index or label
            ycol (index): Y column index or label
            zcol (index): Z column index or label

        Keyword Arguments:
            shape (two-tuple): Number of points along x and y in the grid - defaults to a square of sidelength = square root of the length of the data.
            xlim (tuple): The xlimits, defaults to automatically determined from data
            ylim (tuple): The ylimits, defaults to automatically determined from data
            plotter (function): Function to use to plot data. Defaults to plt.contour
            show_plot (bool): Turn on interfactive plotting and show plot when drawn
            save_filename (string or None): If set to a string, save the plot with this filename
            figure (integer or matplotlib.figure or boolean): Controls which figure is used for the plot, or if a new figure is opened.
            **kargs (dict): Other arguments are passed on to the plotter.

        Returns:
            A matplotlib figure
         """
        if plotter is None:
            plotter = plt.contour
        kargs["plotter"] = plotter
        return self.plot_xyz(xcol, ycol, zcol, shape, xlim, ylim, **kargs)

    def figure(self, figure=None,projection="rectilinear",**kargs):
        """Set the figure used by :py:class:`Stoner.Plot.PlotFile`.

        Args:
            figure A matplotlib figure or figure number

        Returns:
            The current \b Stoner.PlotFile instance"""
        if figure is None:
            figure, ax = self.template.new_figure(None,projection=projection,**kargs)
        elif isinstance(figure, int):
            figure, ax = self.template.new_figure(figure,projection=projection,**kargs)
        elif isinstance(figure, mplfig.Figure):
            figure, ax = self.template.new_figure(figure.number,projection=projection,**kargs)
        self.__figure = figure
        return self

    def griddata(self, xcol=None, ycol=None, zcol=None, shape=None, xlim=None, ylim=None, method="linear", **kargs):
        """Function to convert xyz data onto a regular grid.

            Args:
            xcol (index): Column to be used for the X-Data
            ycol (index): column to be used for Y-Data - default value is column to the right of the x-data column
            zcol (index): column to be used for the Z-data - default value is the column to the right of the y-data column

            Keyword Arguments:
                shape (two-tuple): Number of points along x and y in the grid - defaults to a square of sidelength = square root of the length of the data.
                xlim (tuple): The xlimits
                ylim (tuple) The ylimits
                method (string): Type of interploation to use, default is linear

            ReturnsL
                X,Y,Z three two dimensional arrays of the co-ordinates of the interpolated data
        """

        if None in (xcol, ycol, zcol):
            if "_startx" in kargs:
                startx = kargs["_startx"]
                del kargs["_startx"]
            else:
                startx = 0
            cols = self.setas._get_cols(startx=startx)
            if xcol is None:
                xcol = cols["xcol"]
            if ycol is None:
                ycol = cols["ycol"][0]
            if zcol is None:
                if len(cols["zcol"]) > 0:
                    zcol = cols["zcol"][0]
        if shape is None or not (isinstance(shape, tuple) and len(shape) == 2):
            shape = (_np_.floor(_np_.sqrt(len(self))), _np_.floor(_np_.sqrt(len(self))))
        if xlim is None:
            xlim = (_np_.min(self.column(xcol)) * (shape[0] - 1) / shape[0], _np_.max(self.column(xcol)) *
                    (shape[0] - 1) / shape[0])
        if isinstance(xlim, tuple) and len(xlim) == 2:
            xlim = (xlim[0], xlim[1], (xlim[1] - xlim[0]) / shape[0])
        elif isinstance(xlim, tuple) and len(xlim) == 3:
            xlim[2] = len(range(*xlim))
        else:
            raise RuntimeError("X limit specification not good.")
        if ylim is None:
            ylim = (_np_.min(self.column(ycol)) * (shape[1] - 1) / shape[1], _np_.max(self.column(ycol)) *
                    (shape[0] - 1) / shape[0])
        if isinstance(ylim, tuple) and len(ylim) == 2:
            ylim = (ylim[0], ylim[1], (ylim[1] - ylim[0]) / shape[1])
        elif isinstance(ylim, tuple) and len(ylim) == 3:
            ylim[2] = len(range(*ylim))
        else:
            raise RuntimeError("Y limit specification not good.")

        np = _np_.mgrid[slice(*xlim), slice(*ylim)].T

        points = _np_.array([self.column(xcol), self.column(ycol)]).T
        if zcol is None:
            zdata = _np_.zeros(len(self))
        elif isinstance(zcol, _np_.ndarray) and zcol.shape[0] == len(self):  # zcol is some data
            zdata = zcol
        else:
            zdata = self.column(zcol)
        if len(zdata.shape) == 1:
            Z = griddata(points, zdata, np, method=method)
        elif len(zdata.shape) == 2:
            Z = _np_.zeros((np.shape[0], np.shape[1], zdata.shape[1]))
            for i in range(zdata.shape[1]):
                Z[:,:, i] = griddata(points, zdata[:, i], np, method=method)
        return np[:,:, 0], np[:,:, 1], Z

    def image_plot(self, xcol=None, ycol=None, zcol=None, shape=None, xlim=None, ylim=None, **kargs):
        """Grid up the three columns of data and plot.

        Args:
            xcol (index): Column to be used for the X-Data
            ycol (index): column to be used for Y-Data - default value is column to the right of the x-data column
            zcol (index): column to be used for the Z-data - default value is the column to the right of the y-data column

        Keyword Arguments:
            shape (two-tuple): Number of points along x and y in the grid - defaults to a square of sidelength = square root of the length of the data.
            xlim (tuple): The xlimits, defaults to automatically determined from data
            ylim (tuple): The ylimits, defaults to automatically determined from data
            xlabel (string) X axes label. Deafult is None - guess from xvals or metadata
            ylabel (string): Y axes label, Default is None - guess from metadata
            zlabel (string): Z axis label, Default is None - guess from metadata
            plotter (function): Function to use to plot data. Defaults to plt.contour
            show_plot (bool): Turn on interfactive plotting and show plot when drawn
            save_filename (string or None): If set to a string, save the plot with this filename
            figure (integer or matplotlib.figure or boolean): Controls which figure is used for the plot, or if a new figure is opened.
            **kargs (dict): Other arguments are passed on to the plotter.

        Returns:
            A matplotlib figure
         """
        locals().update(self._fix_cols(xcol=xcol, ycol=ycol, **kargs))

        X, Y, Z = self.griddata(xcol, ycol, zcol, shape, xlim, ylim)
        defaults = {
            "origin": "lower",
            "interpolation": "bilinear",
            "plotter": plt.imshow,
            "title": self.filename,
            "cmap": cm.jet,
            "figure": self.__figure,
            "xlabel": self._col_label(self.find_col(xcol)),
            "ylabel": self._col_label(self.find_col(ycol))
        }
        kargs, nonkargs = self._fix_kargs(None, defaults, **kargs)
        plotter = nonkargs["plotter"]
        self.__figure, ax = self._fix_fig(nonkargs["figure"])
        if "cmap" in kargs:
            cmap = cm.get_cmap(kargs["cmap"])
        elif "cmap" in nonkargs:
            cmap = cm.get_cmap(nonkargs["cmap"])
        if len(Z.shape) == 2:
            Z = cmap(Z)
        elif len(Z.shape) != 3:
            raise RunetimeError("Z Data has a bad shape: {}".format(Z.shape))
        xmin = _np_.min(X.ravel())
        xmax = _np_.max(X.ravel())
        ymin = _np_.min(Y.ravel())
        ymax = _np_.max(Y.ravel())
        aspect = (xmax - xmin) / (ymax - ymin)
        extent = [xmin, xmax, ymin, ymax]
        fig = plotter(Z, extent=extent, aspect=aspect, **kargs)
        self._fix_titles(0, "none", **nonkargs)
        return fig

    def inset(self, parent=None, loc=None, width=0.35, height=0.30, **kargs):
        """Add a new set of axes as an inset to the current plot.

        Keyword Arguments:
            parent (matplotlib axes): Which set of axes to add inset to, defaults to the current set
            loc (int or string): Inset location - can be a string like *top right* or *upper right* or a number.
            width,height (int,float or string) the dimensions of the inset specified as a integer %, or floating point fraction of the parent axes, or as a string measurement.
            kargs (dictionary): all other keywords are passed through to inset_locator.inset_axes

        Returns:
            A new set of axes"""

        locations = ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left',
                     'center right', 'lower center', 'upper center', 'center']
        locations2 = ['best', 'top right', 'top left', 'bottom left', 'bottom right', 'outside', 'leftside',
                      'rightside', 'bottom', 'top', 'middle']
        if isinstance(loc, string_types):
            if loc in locations:
                loc = locations.index(loc)
            elif loc in locations2:
                loc = locations2.index(loc)
            else:
                raise RuntimeError("Couldn't work out where {} was supposed to be".format(loc))
        if isinstance(width, int):
            width = "{}%".format(width)
        elif isinstance(width, float) and 0 < width <= 1:
            width = "{}%".format(width * 100)
        elif not isinstance(width, string_types):
            raise RuntimeErroror("didn't Recognize width specification {}".format(width))
        if isinstance(height, int):
            height = "{}%".format(height)
        elif isinstance(height, float) and 0 < height <= 1:
            height = "{}%".format(height * 100)
        elif not isinstance(height, string_types):
            raise RuntimeErroror("didn't Recognize width specification {}".format(width))
        if parent is None:
            parent = plt.gca()
        return inset_locator.inset_axes(parent, width, height, loc, **kargs)

    def plot(self, *args, **kargs):
        """Try to make an appropriate plot based on the defined column assignments.

        The column assignments are examined to determine whether to plot and x,y plot or an x,y,z plot
        and whether to plot error bars (for an x,y plot). All keyword argume nts are passed through to
        the selected plotting routine.
        """
        if len(args) != 0:
            axes = len(args)
        else:
            for x in [i for i in range(len(self.setas)) if self.setas[i] == "x"]:
                cols = self.setas._get_cols(startx=x)
                kargs["_startx"] = x
                axes = cols["axes"]

        if "template" in kargs:
            self.template=kargs.pop("template")

        plotters = [None, None, self.plot_xy, self.plot_xyz, self.plot_xyuv, self.plot_xyuv, self.plot_xyzuvw]
        if 2 <= axes <= 6:
            plotter = plotters[axes]
            ret = plotter(*args, **kargs)
            plt.show()
        else:
            raise RuntimeError("Unable to work out plot type !")
        return ret

    def plot_matrix(self,
                    xvals=None,
                    yvals=None,
                    rectang=None,
                    cmap=plt.cm.jet,
                    show_plot=True,
                    title='',
                    xlabel=None,
                    ylabel=None,
                    zlabel=None,
                    figure=None,
                    plotter=None, **kwords):
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
                plotter (callable): Optional arguement that passes a plotting function into the routine. Sensible choices might be plt.plot (default), py.semilogy, plt.semilogx
                kwords (dict): A dictionary of other keyword arguments to pass into the plot function.

            Returns:
                The matplotib figure with the data plotted"""
        # Sortout yvals values
        if isinstance(yvals, int):  #  Int means we're sepcifying a data row
            if rectang is None:  # we need to intitialise the rectang
                rectang = (yvals + 1, 0)  # We'll sort the column origin later
            elif isinstance(rectang,
                            tuple) and rectang[1] <= yvals:  # We have a rectang, but we need to adjust the row origin
                rectang[0] = yvals + 1
            yvals = self[yvals]  # change the yvals into a numpy array
        elif isinstance(yvals, (list, tuple)):  # We're given the yvals as a list already
            yvals = _np_.array(yvals)
        elif yvals is None:  # No yvals, so we'l try column headings
            if isinstance(xvals, index_types):  # Do we have an xcolumn header to take away ?
                xvals = self.find_col(xvals)
                headers = self.column_headers[xvals + 1:]
            elif xvals is None:  # No xvals so we're going to be using the first column
                xvals = 0
                headers = self.column_headers[1:]
            else:
                headers = self.column_headers
            yvals = _np_.array([float(x) for x in headers])  #Ok try to construct yvals aray
        else:
            raise RuntimeError("uvals must be either an integer, list, tuple, numpy array or None")
        #Sort out xvls values
        if isinstance(xvals, index_types):  # String or int means using a column index
            if xlabel is None:
                xlabel = self._col_label(xvals)
            if rectang is None:  # Do we need to init the rectan ?
                rectang = (0, xvals + 1)
            elif isinstance(rectang, tuple):  # Do we need to adjust the rectan column origin ?
                rectang[1] = xvals + 1
            xvals = self.column(xvals)
        elif isinstance(xvals, (list, tuple)):  # Xvals as a data item
            xvals = _np_.array(xvals)
        elif isinstance(xvals, _np_.ndarray):
            pass
        elif xvals is None:  # xvals from column 0
            xvals = self.column(0)
            if rectang is None:  # and fix up rectang
                rectang = (0, 1)
        else:
            raise RuntimeError("xvals must be a string, integer, list, tuple or numpy array or None")

        if isinstance(rectang, tuple) and len(rectang) == 2:  # Sort the rectang value
            rectang = (rectang[0], rectang[1], _np_.shape(self.data)[0] - rectang[0],
                       _np_.shape(self.data)[1] - rectang[1])
        elif rectang is None:
            rectang = (0, 0, _np_.shape(self.data)[0], _np_.shape(self.data)[1])
        elif isinstance(rectang, tuple) and len(rectang) == 4:  # Ok, just make sure we have enough data points left.
            rectang = (rectang[0], rectang[1], min(rectang[2], _np_.shape(self.data)[0] - rectang[0]),
                       min(rectang[3], _np_.shape(self.data)[1] - rectang[1]))
        else:
            raise RuntimeError("rectang should either be a 2 or 4 tuple or None")

#Now we can create X,Y and Z 2D arrays
        zdata = self.data[rectang[0]:rectang[0] + rectang[2], rectang[1]:rectang[1] + rectang[3]]
        xvals = xvals[0:rectang[2]]
        yvals = yvals[0:rectang[3]]
        xdata, ydata = _np_.meshgrid(xvals, yvals)

        #This is the same as for the plot_xyz routine'
        if isinstance(figure, int):
            figure, ax = self.template.new_figure(figure)
        elif isinstance(figure, bool) and not figure:
            figure, ax = self.template.new_figure(None)
        elif isinstance(figure, mplfig.Figure):
            figure, ax = self.template.new_figure(figure.number)
        elif isinstance(self.__figure, mplfig.Figure):
            figure = self.__figure
        else:
            figure, ax = self.template.new_figure(None, projection="3d")
        self.__figure = figure
        if show_plot == True:
            plt.ion()
        if plotter is None:
            plotter = self.__SurfPlotter
        plotter(xdata, ydata, zdata, cmap=cmap, **kwords)
        labels = {"xlabel": (xlabel, "X Data"), "ylabel": (ylabel, "Y Data"), "zlabel": (zlabel, "Z Data")}
        for label in labels:
            (v, default) = labels[label]
            if v is None:
                if label in self.metadata:
                    labels[label] = self[label]
                else:
                    labels[label] = default
            else:
                labels[label] = v

        plt.xlabel(str(labels["xlabel"]))
        plt.ylabel(str(labels["ylabel"]))
        if plotter == self.__SurfPlotter:
            self.axes[0].set_zlabel(str(labels["zlabel"]))
        if title == '':
            title = self.filename
        plt.title(title)
        plt.show()
        plt.draw()

        return self.__figure

    def plot_xy(self, xcol=None, ycol=None, fmt=None, xerr=None, yerr=None, multiple=None, **kargs):
        """Makes a simple X-Y plot of the specified data.

        Args:
            xcol (index): Column to be used for the X-Data
            ycol (index): column to be used for Y-Data - default value is column to the right of the x-data column

        Keyword Arguments:
            fmt (strong or sequence of strings): Specifies the format for the plot - see matplotlib documentation for details
            xerr,yerr (index): Columns of data to get x and y errorbars from. Setting these turns the default plotter to plt.errorbar
            xlabel (string) X axes label. Deafult is None - guess from xvals or metadata
            ylabel (string): Y axes label, Default is None - guess from metadata
            zlabel (string): Z axis label, Default is None - guess from metadata
            title (string): Optional parameter that specfies the plot title - otherwise the current DataFile filename is used
            plotter (function): Function to use to plot data. Defaults to plt.plot unless error bars are set
            show_plot (bool): Turn on interfactive plotting and show plot when drawn
            save_filename (string or None): If set to a string, save the plot with this filename
            figure (integer or matplotlib.figure or boolean): Controls which figure is used for the plot, or if a new figure is opened.
            multiple (string): how to handle multiple y-axes with a common x axis. Options are:

                - *common* single y-axis (default)
                - *panels* panels sharing common x axis
                - *sub plots* sub plots
                - *y2* single axes with 2 y scales

            **kargs (dict): Other arguments are passed on to the plotter.

        Returns:
            A matplotlib.figure isntance

        """
        c = self._fix_cols(xcol=xcol, ycol=ycol, xerr=xerr, yerr=yerr, multi_y=True, **kargs)
        (kargs["xerr"], kargs["yerr"]) = (c.xerr, c.yerr)

        if "template" in kargs: #Catch template in kargs
            self.template=kargs.pop("template")

        defaults = {
            "plotter": plt.plot,
            "show_plot": True,
            "figure": self.__figure,
            "title": self.filename,
            "save_filename": None,
            "xlabel": self._col_label(self.find_col(c.xcol)),
            "ylabel": self._col_label(self.find_col(c.ycol), True)
        }
        otherargs = []
        if "plotter" not in kargs and (c.xerr is not None or c.yerr is not
                                       None):  # USe and errorbar blotter by default for errors
            kargs["plotter"] = plt.errorbar
            otherargs = ["agg_filter", "alpha", "animated", "antialiased", "aa", "axes", "clip_box", "clip_on",
                         "clip_path", "color", "c", "contains", "dash_capstyle", "dash_joinstyle", "dashes",
                         "drawstyle", "fillstyle", "gid", "label", "linestyle", "ls", "linewidth", "lw", "lod",
                         "marker", "markeredgecolor", "mec", "markeredgewidth", "mew", "markerfacecolor", "mfc",
                         "markerfacecoloralt", "mfcalt", "markersize", "ms", "markevery", "path_effects", "picker",
                         "pickradius", "rasterized", "sketch_params", "snap", "solid_capstyle", "solid_joinstyle",
                         "transform", "url", "visible", "xdata", "ydata", "zorder"]
        elif "plotter" not in kargs:
            kargs["plotter"] = plt.plot
            otherargs = ["agg_filter", "alpha", "animated", "antialiased", "aa", "axes", "clip_box", "clip_on",
                         "clip_path", "color", "c", "contains", "dash_capstyle", "dash_joinstyle", "dashes",
                         "drawstyle", "fillstyle", "gid", "label", "linestyle", "ls", "linewidth", "lw", "lod",
                         "marker", "markeredgecolor", "mec", "markeredgewidth", "mew", "markerfacecolor", "mfc",
                         "markerfacecoloralt", "mfcalt", "markersize", "ms", "markevery", "path_effects", "picker",
                         "pickradius", "rasterized", "sketch_params", "snap", "solid_capstyle", "solid_joinstyle",
                         "transform", "url", "visible", "xdata", "ydata", "zorder"]

        if multiple == None:
            multiple = self.multiple
        else:
            self.multiple = multiple
        kargs, nonkargs = self._fix_kargs(None, defaults, otherargs, **kargs)

        for err in ["xerr", "yerr"]:  # Check for x and y error keywords
            if err in kargs and not isNone(kargs[err]):
                if isinstance(kargs[err], index_types):
                    kargs[err] = self.column(kargs[err])
                elif isinstance(kargs[err], list) and isinstance(c.ycol, list) and len(kargs[err]) == len(c.ycol):
                    # Ok, so it's a list, so redo the check for each  item.
                    for i in range(len(kargs[err])):
                        if isinstance(kargs[err][i], index_types):
                            kargs[err][i] = self.column(kargs[err][i])
                        else:
                            kargs[err][i] = _np_.zeros(len(self))
                else:
                    kargs[err] = _np_.zeros(len(self))
            else:
                kargs.pop(err,None)


        temp_kwords = copy.copy(kargs)
        if isinstance(c.ycol, (index_types)):
            c.ycol = [c.ycol]
        if len(c.ycol) > 1:
            if multiple == "panels":
                self.__figure, ax = plt.subplots(nrows=len(c.ycol), sharex=True, gridspec_kw={"hspace": 0})
            elif multiple == "subplots":
                m = int(_np_.floor(_np_.sqrt(len(c.ycol))))
                n = int(_np_.ceil(len(c.ycol) / m))
                self.__figure, ax = plt.subplots(nrows=m, ncols=n)
            else:
                self.__figure, ax = self._fix_fig(self.__figure)
        else:
            self.__figure, ax = self._fix_fig(nonkargs["figure"])
        for ix in range(len(c.ycol)):
            if multiple != "common":
                nonkargs["ylabel"] = self._col_label(self.find_col(c.ycol[ix]))
            if ix > 0:
                if multiple == "y2" and ix == 1:
                    self.y2()
                    lines=plt.gca()._get_lines
                    if hasattr(lines,"color_cylce"): #mpl<1.5
                        cc=lines.color_cycle
                    else: #MPL >=1.5
                        cc=lines.prop_cycler
                    for i in range(ix):
                        next(cc)
            if len(c.ycol) > 1 and multiple in ["y2", "panels", "subplots"]:
                self.ax = ix  # We're manipulating the plotting here
            if isinstance(fmt, list):  # Fix up the format
                fmt_t = fmt[ix]
            else:
                fmt_t = fmt
            if "label" in kargs and isinstance(kargs["label"], list):  # Fix label keywords
                temp_kwords["label"] = kargs["label"][ix]
            if "yerr" in kargs and isinstance(kargs["yerr"], list):  # Fix yerr keywords
                temp_kwords["yerr"] = kargs["yerr"][ix]
            # Call plot

            if fmt_t is None:
                self._Plot(c.xcol, c.ycol[ix], fmt_t, nonkargs["plotter"], self.__figure, **temp_kwords)
            else:
                self._Plot(c.xcol, c.ycol[ix], fmt_t, nonkargs["plotter"], self.__figure, **temp_kwords)
            self._fix_titles(ix, multiple, **nonkargs)
            if ix > 0:  # Hooks for multiple subplots
                if multiple == "panels":
                    loc, lab = plt.yticks()
                    lab = [l.get_text() for l in lab]
                    plt.yticks(loc[:-1], lab[:-1])
        return self.__figure

    def plot_xyz(self, xcol=None, ycol=None, zcol=None, shape=None, xlim=None, ylim=None, projection="3d", **kargs):
        """Plots a surface plot based on rows of X,Y,Z data using matplotlib.pcolor().

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
                plotter (callable): Optional arguement that passes a plotting function into the routine. Default is a 3d surface plotter, but contour plot and pcolormesh also work.
                projection (string or None): Whether to use a 3D projection or regular 2D axes (deault is 3D)
                **kargs (dict): A dictionary of other keyword arguments to pass into the plot function.

            Returns:
                A matplotlib.figure isntance
        """
        c = self._fix_cols(xcol=xcol, ycol=ycol, zcol=zcol, multi_y=False, **kargs)
        xdata, ydata, zdata = self.griddata(c.xcol, c.ycol, c.zcol, shape=shape, xlim=xlim, ylim=ylim)

        if "template" in kargs: #Catch template in kargs
            self.template=kargs.pop("template")

        defaults = {
            "plotter": self.__SurfPlotter,
            "show_plot": True,
            "figure": self.__figure,
            "title": self.filename,
            "save_filename": None,
            "cmap": cm.jet,
            "rstride": max(1,zdata.shape[0] / 50),
            "cstride": max(1,zdata.shape[1] / 50)
        }
        coltypes = {"xlabel": c.xcol, "ylabel": c.ycol, "zlabel": c.zcol}
        for k in coltypes:
            if isinstance(coltypes[k], index_types):
                label = self._col_label(coltypes[k])
                if isinstance(label, list):
                    label = ",".join(label)
                defaults[k] = label
        if "plotter" not in kargs or ("plotter" in kargs and kargs["plotter"] == self.__SurfPlotter):
            otherkargs = ["rstride", "cstride", "color", "cmap", "facecolors", "norm", "vmin", "vmax", "shade","linewidth","ax"]
        else:
            otherkargs = ["vmin", "vmax","shade","color","linewidth"]
        kargs, nonkargs = self._fix_kargs(None, defaults, otherkargs=otherkargs, projection=projection,**kargs)
        plotter = nonkargs["plotter"]
        self.__figure, ax = self._fix_fig(nonkargs["figure"], projection=projection)
        if isinstance(plotter,string_types):
            plotter=ax.__getattribute__(plotter)
        self.plot3d=plotter(xdata, ydata, zdata, **kargs)
        if plotter != self.__SurfPlotter:
            del (nonkargs["zlabel"])
        self._fix_titles(0, "none", **nonkargs)
        return self.__figure

    def plot_xyuv(self, xcol=None, ycol=None, ucol=None, vcol=None, wcol=None, **kargs):
        """Makes an overlaid image and quiver plot.
        Args:
                xcol (index): Xcolumn index or label
                ycol (index): Y column index or label
                zcol (index): Z column index or label
                ucol (index): U column index or label
                vcol (index): V column index or label
                wcol (index): W column index or label

            Keyword Arguments:
                show_plot (bool): True Turns on interactive plot control
                title (string): Optional parameter that specfies the plot title - otherwise the current DataFile filename is used
                save_filename (string): Filename used to save the plot
                figure (matplotlib figure): Controls what matplotlib figure to use. Can be an integer, or a matplotlib.figure or False. If False then a new figure is always used, otherwise it will default to using the last figure used by this DataFile object.
                plotter (callable): Optional arguement that passes a plotting function into the routine. Default is a 3d surface plotter, but contour plot and pcolormesh also work.
                **kargs (dict): A dictionary of other keyword arguments to pass into the plot function.
                """
        c = self._fix_cols(xcol=xcol, ycol=ycol, ucol=ucol, vcol=vcol, wcol=wcol, **kargs)

        if "template" in kargs: #Catch template in kargs
            self.template=kargs.pop("template")

        if isinstance(c.wcol, index_types):
            wdata = self.column(c.wcol)
            phidata = (wdata - _np_.min(wdata)) / (_np_.max(wdata) - _np_.min(wdata))
        else:
            phidata = _np_.ones(len(self)) * 0.5
            wdata = phidata - 0.5
        qdata = 0.5 + (_np_.arctan2(self.column(c.ucol), self.column(c.vcol)) / (2 * _np_.pi))
        rdata = _np_.sqrt(self.column(c.ucol) ** 2 + self.column(c.vcol) ** 2 + wdata ** 2)
        rdata = rdata / max(rdata)
        Z = hsl2rgb(qdata, rdata, phidata).astype('f') / 255.0
        if "save_filename" in kargs:
            save = kargs["save_filename"]
            del kargs["save_filename"]
        else:
            save = None
        fig = self.image_plot(c.xcol, c.ycol, Z, **kargs)
        if save is not None:  # stop saving file twice
            kargs["save_filename"] = save
        fig = self.quiver_plot(c.xcol, c.ycol, c.ucol, c.vcol, **kargs)

        return fig

    def plot_xyzuvw(self, xcol=None, ycol=None, zcol=None, ucol=None, vcol=None, wcol=None, **kargs):
        """Plots a vector field plot based on rows of X,Y,Z (U,V,W) data using ,ayavi.

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
                mode (string): glyph type, default is "cone"
                scale_factor(float): Scale-size of glyphs.
                figure (mlab figure): Controls what mlab figure to use. Can be an integer, or a mlab.figure or False. If False then a new figure is always used, otherwise it will default to using the last figure used by this DataFile object.
                plotter (callable): Optional arguement that passes a plotting function into the routine. Sensible choices might be plt.plot (default), py.semilogy, plt.semilogx
                kargs (dict): A dictionary of other keyword arguments to pass into the plot function.

            Returns:
                A mayavi scene instance
        """
        try:
            from mayavi import mlab, core
        except ImportError:
            return None
        c = self._fix_cols(xcol=xcol, ycol=ycol, zcol=zcol, ucol=ucol, vcol=vcol, wcol=wcol, multi_y=False, **kargs)

        if "template" in kargs: #Catch template in kargs
            self.template=kargs.pop("template")

        defaults = {
            "figure": self.__figure,
            "plotter": self._VectorFieldPlot,
            "show_plot": True,
            "mode": "cone",
            "scale_factor": 1.0,
            "colors": True
        }
        otherkargs = ["color", "colormap", "extent", "figure", "line_width", "mask_points", "mode", "name", "opacity",
                      "reset_zoom", "resolution", "scalars", "scale_factor", "scale_mode", "transparent", "vmax",
                      "vmin"]
        kargs, nonkargs = self._fix_kargs(None, defaults, otherkargs=otherkargs, **kargs)
        colors = nonkargs["colors"]
        if isinstance(colors, bool) and colors:
            pass
        elif isinstance(colors, index_types):
            colors = self.column(colors)
        elif isinstance(colors, _np_.ndarray):
            pass
        elif callable(colors):
            colors = _np_.array([colors(x) for x in self.rows()])
        else:
            raise RuntimeError("Do not recognise what to do with the colors keyword.")
        kargs["scalars"] = colors
        figure = nonkargs["figure"]
        plotter = nonkargs["plotter"]
        if isinstance(figure, int):
            figure = mlab.figure(figure)
        elif isinstance(figure, bool) and not figure:
            figure = mlab.figure(bgcolor=(1, 1, 1))
        elif isinstance(figure, core.scene.Scene):
            pass
        elif isinstance(self.__figure, core.scene.Scene):
            figure = self.__figure
        else:
            figure = mlab.figure(bgcolor=(1, 1, 1))
        self.__figure = figure
        kargs["figure"] = figure
        plotter(self.column(c.xcol), self.column(c.ycol), self.column(c.zcol), self.column(c.ucol), self.column(c.vcol),
                self.column(c.wcol), **kargs)
        if nonkargs["show_plot"]:
            mlab.show()
        return self.__figure

    def quiver_plot(self, xcol=None, ycol=None, ucol=None, vcol=None, **kargs):
        """Make a 2D Quiver plot from the data.

        Args:
            xcol (index): Xcolumn index or label
            ycol (index): Y column index or label
            zcol (index): Z column index or label
            ucol (index): U column index or label
            vcol (index): V column index or label
            wcol (index): W column index or label

        Keyword Arguments:
            xlabel (string) X axes label. Deafult is None - guess from xvals or metadata
            ylabel (string): Y axes label, Default is None - guess from metadata
            zlabel (string): Z axis label, Default is None - guess from metadata
            plotter (function): Function to use to plot data. Defaults to plt.contour
            headlength,headwidth,headaxislength (float): Controls the size of the quiver heads
            show_plot (bool): Turn on interfactive plotting and show plot when drawn
            save_filename (string or None): If set to a string, save the plot with this filename
            figure (integer or matplotlib.figure or boolean): Controls which figure is used for the plot, or if a new figure is opened.
            **kargs (dict): Other arguments are passed on to the plotter.


        Returns:
            A matplotlib figure instance.

        Keyword arguments are all passed through to :py:func:`matplotlib.plt.quiver`.

        """
        locals().update(self._fix_cols(xcol=xcol, ycol=ycol, ucol=ucol, vcol=vcol, **kargs))
        defaults = {
            "pivot": "center",
            "color": (0, 0, 0, 0.5),
            "headlength": 5,
            "headaxislength": 5,
            "headwidth": 4,
            "units": "xy",
            "plotter": plt.quiver,
            "show_plot": True,
            "figure": self.__figure,
            "title": self.filename,
            "xlabel": self._col_label(self.find_col(xcol)),
            "ylabel": self._col_label(self.find_col(ycol))
        }
        otherkargs = ["units", "angles", "scale", "scale_units", "width", "headwidth", "headlength", "headaxislength",
                      "minshaft", "minlength", "pivot", "color"]

        if "template" in kargs: #Catch template in kargs
            self.template=kargs.pop("template")

        kargs, nonkargs = self._fix_kargs(None, defaults, otherkargs=otherkargs, **kargs)
        plotter = nonkargs["plotter"]
        self.__figure, ax = self._fix_fig(nonkargs["figure"])
        fig = plotter(self.column(self.find_col(xcol)), self.column(self.find_col(ycol)),
                      self.column(self.find_col(ucol)), self.column(self.find_col(vcol)), **kargs)
        self._fix_titles(0, "non", **nonkargs)
        return fig

    def subplot(self, *args, **kargs):
        """Pass throuygh for plt.subplot().

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
        fig, ax = self.template.new_figure(self.__figure.number)
        sp = plt.subplot(*args, **kargs)
        if len(args) == 1:
            rows = args[0] // 100
            cols = (args[0] // 10) % 10
            index = args[0] % 10
        else:
            rows = args[0]
            cols = args[1]
            index = args[2]
        if len(self._subplots) < rows * cols:
            self._subplots.extend([None for i in range(rows * cols - len(self._subplots))])
        self._subplots[index - 1] = sp
        return sp

    def x2(self):
        """Generate a new set of axes with a second x-scale.

        Returns:
            The new matplotlib.axes instance.
        """
        ax = self.fig.gca()
        ax2 = ax.twiny()
        plt.sca(ax2)
        return ax2

    def y2(self):
        """Generate a new set of axes with a second y-scale.

        Returns:
            The new matplotlib.axes instance
        """
        ax = self.fig.gca()
        ax2 = ax.twinx()
        plt.subplots_adjust(right=self.__figure.subplotpars.right - 0.05)
        plt.sca(ax2)
        return ax2


def hsl2rgb(h, s, l):
    """Converts from hsl colourspace to rgb colour space with numpy arrays for speed.

    Args:
        h (array): Hue value
        s (array): Saturation value
        l (array): Luminence value

    Returns:
        2D array (Mx3) of unsigned 8bit integers
        """
    w = _np_.where
    if isinstance(h, float):
        h = _np_.array([h])
    if isinstance(s, float):
        s = _np_.array([s])
    if isinstance(l, float):
        l = _np_.array([l])

    if h.shape != l.shape or h.shape != s.shape:
        raise RuntimeError("Must have equal shaped arrays for h, s and l")

    rgb = _np_.zeros((len(h), 3))
    hls = _np_.column_stack([h, l, s])
    for i in range(len(h)):
        rgb[i,:] = _np_.array(hls_to_rgb(*hls[i]))
    return (255 * rgb).astype('u1')
