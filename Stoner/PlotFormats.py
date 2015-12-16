# -*- coding: utf-8 -*-
"""

Plot Templates module - contains classes that style plots for Stoner.Plot and pyplot
Created on Fri Feb 07 19:57:30 2014

@author: Gavin Burnell
"""

from Stoner.compat import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, Formatter
from matplotlib.ticker import AutoLocator
from os.path import join, dirname, realpath,exists
from sys import platform as _platform
from numpy.random import normal
from inspect import getfile

try:
    import seaborn as sns
    SEABORN=True
except ImportError:
    SEABORN=False

import numpy as _np_


class TexFormatter(Formatter):
    """An axis tick label formatter that emits Tex formula mode code
    so that large numbers are registered as \\times 10^{power}
    rather than ysing E notation."""

    def __call__(self, value, pos=None):
        """Return the value ina  suitable texable format"""
        if value is None or _np_.isnan(value):
            ret = ""
        elif value != 0.0:
            power = _np_.floor(_np_.log10(_np_.abs(value)))
            if _np_.abs(power) < 4:
                ret = "${}$".format(value)
            else:
                v = value / (10 ** power)
                ret = "${}\\times 10^{{{:.0f}}}$".format(v, power)
        else:
            ret = "$0.0$"
        return ret

    def format_data(self, value):
        return self.__call__(value)

    def format_data_short(self, value):
        return "{:g}".format(value)


class TexEngFormatter(EngFormatter):
    """An axis tick label formatter that emits Tex formula mode code
    so that large numbers are registered as \\times 10^{power}
    rather than ysing E notation."""

    prefix = {
        3: "k",
        6: "M",
        9: "G",
        12: "T",
        15: "P",
        18: "E",
        21: "Z",
        24: "Y", -3: "m", -6: "\\mu", -9: "n", -12: "p", -15: "f", -18: "a", -21: "z", -24: "y"
    }

    def __call__(self, value, pos=None):
        """Return the value ina  suitable texable format"""
        if value is None or _np_.isnan(value):
            ret = ""
        elif value != 0.0:
            power = _np_.floor(_np_.log10(_np_.abs(value)))
            power = (_np_.sign(power) * (_np_.floor(_np_.abs(power / 3.0))) - 1) * 3.0
            if _np_.abs(power) < 3:
                ret = "${}$".format(value)
            else:
                v = int(value / (10 ** power))
                ret = "${}\\mathrm{{{} {}}}$".format(v, self.prefix[int(power)], self.unit)
        else:
            ret = "$0.0$"
        return ret

    def format_data(self, value):
        return self.__call__(value)

    def format_data_short(self, value):
        return "{:g}".format(value)


class DefaultPlotStyle(object):
    """Produces a default plot style.

    To produce alternative plot styles, create subclasses of this plot. Either override or
    create additional attributes to define rc Parameters (see Matplotlib documentation for
    available rc parameters) and override the :py:meth:Stoner.PlotFormats.DefaultPlotStyle.customise`
    method to carry out additional plot formatting.

    Attributes:
        fig_width_pt (float): Preferred width of plot in points
        show_xlabel (bool): Show the title in the plot
        show_ylabel (bool): Show the x-axis Labels
        show_zlabel (bool): show the y-xaxis labels
        show_title (bool): show the title
        show_legend (bool): show the legend
        stylename (string): Name of the matplotlib style to use
        stylesheet (list): Calculated list of stylesheets found by traversing the class heirarchy

    Example
        .. plot:: samples/plotstyles/default.py
            :include-source:

    """
    """Internal class attributes."""
    _inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    _mm_per_inch = 25.4
    _golden_mean = (_np_.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    """Settings for this figure type. All instance attributes which start template_
    will be used. Once the leading template_ is stripped, all _ characters are replaced
    with . and then the attributes are mapped to a dictionary and used to update the rcParams
    dictionary

    """
    show_xlabel = True
    show_ylabel = True
    show_zlabel = True
    show_title = True
    show_legend = True
    xformatter = TexFormatter
    yformatter = TexFormatter
    zformatter = TexFormatter
    xlocater = AutoLocator
    ylocater = AutoLocator
    zlocater = AutoLocator
    stylename = "default"

    subplot_settings = {
        "panels": {
            "xlabel": (False, False, True),
            "ylabel": (True, True, True),
            "zlabel": (False, False, False),
            "title": (True, False, False)
        },
        "subplots": {
            "xlabel": (True, True, True),
            "ylabel": (True, True, True),
            "zlabel": (False, False, False),
            "title": (True, True, True)
        },
        "y2": {
            "xlabel": (True, False, False),
            "ylabel": (True, True, True),
            "zlabel": (False, False, False),
            "title": (True, False, False)
        }
    }

    def __init__(self, **kargs):
        """Create a template instance of this template.

        Keyword arguments may be supplied to set default parameters. Any Matplotlib rc parameter
        may be specified, with .'s replaced with _ and )_ replaced with __.
        """
        self._stylesheet=None
        self.update(**kargs)

    def __call__(self, **kargs):
        """Calling the template object can manipulate the rcParams that will be set."""
        for k in kargs:
            ok=k
            if k.startswith("template_"):
                k=k[9:]
            nk = k.replace("_", ".").replace("..", "_")
            if nk in plt.rcParams:
                super(DefaultPlotStyle, self).__setattr__("template_{}".format(k), kargs[ok])

    def __getattr__(self, name):
        """Provide magic to read certain attributes of the template."""
        if name.startswith("template_"):  #Magic conversion to rcParams
            attrname = name[9:].replace("_", ".").replace("..", "_")
            if attrname in plt.rcParams:
                return plt.rcParams[attrname]
            else:
                raise AttributeError("template attribute not in rcParams")
        elif name == "showlegend":
            return self.show_legend and len(plt.gca().get_legend_handles_labels()[1]) > 1
        else:
            return super(DefaultPlotStyle, self).__getattribute__(name)

    def __setattr__(self, name, value):
        """Ensure stylesheet can't be overwritten and provide magic for template attributes."""
        if name.startswith("template_"):
            attrname = name[9:].replace("_", ".").replace("..", "_")
            plt.rcParams[attrname] = value
            super(DefaultPlotStyle, self).__setattr__(name, value)
        else:
            super(DefaultPlotStyle, self).__setattr__(name, value)


    @property
    def stylesheet(self):
        """Horribly hacky method to traverse over the class heirarchy for style sheet names."""
        if self._stylesheet is not None and self._stylesheet[0]==self.stylename: # Have we cached a copy of our stylesheets ?
            return self._stylesheet[1]
        levels = type.mro(type(self))[:-1]
        sheets=[]
        for c in levels: # Iterate through all possible parent classes and build a list of stylesheets
            if c is self.__class__:
                continue
            for f in [join(dirname(getfile(c)), c.stylename + ".mplstyle"),
                      join(dirname(getfile(c)), "stylelib",c.stylename + ".mplstyle"),
                      join(dirname(realpath(__file__)), "stylelib",c.stylename + ".mplstyle"),
                ]: # Look in first of all the same directory as the class file and then in a stylib folder
                if exists(f):
                    sheets.append(f)
                    break
            else: # Fallback, does the parent class define a builtin stylesheet ?
                if c.stylename in plt.style.available:
                    sheets.append(c.stylename)
        #Now do the same for this class, but allow the stylename to be an instance variable as well
        for f in [join(dirname(getfile(self.__class__)), self.stylename + ".mplstyle"),
                  join(dirname(getfile(self.__class__)), "stylelib",self.stylename + ".mplstyle"),
            ]:
            if exists(f):
                sheets.append(f)
                break
        else:
            if self.stylename in plt.style.available:
                sheets.append(self.stylename)

        self._stylesheet=self.stylename,sheets
        return sheets

    @stylesheet.setter
    def stylesheet(self,value):
        raise AttributeError("Can't set the stylesheet value, this is dervied from the stylename aatribute.")


    def update(self, **kargs):
        """Update the template with new attributes from keyword arguments.
        Keyword arguments may be supplied to set default parameters. Any Matplotlib rc parameter
        may be specified, with .'s replaced with _ and )_ replaced with __.
        """
        for k in kargs:
            if k in dir(self) and not callable(self.__getattr__(k)):
                self.__setattr__(k,kargs[k])
            elif not k.startswith("_"):
                self.__setattr__("template_" + k, kargs[k])

    def new_figure(self, figure=False, **kargs):
        """This is called by PlotFile to setup a new figure before we do anything."""
        plt.rcdefaults()  # Start by resetting to our default settings
        params = dict()
        self.apply()
        if "fig_width_pt" in dir(self):
            self.fig_width = self.fig_width_pt * self._inches_per_pt
        if "fig_height_pt" in dir(self):
            self.fig_height = self.fig_width * self._golden_mean  # height in inches
        if "fig_ratio" in dir(self) and "fig_width" in dir(self):
            self.fig_height = self.fig_width / self.fig_ratio
        if "fig_width" and "fig_height" in self.__dict__:
            self.template_figure_figsize = (self.fig_width, self.fig_height)
        for attr in dir(self):
            if attr.startswith("template_"):
                attrname = attr[9:].replace("_", ".").replace("..", "_")
                value = self.__getattribute__(attr)
                if attrname in plt.rcParams.keys():
                    params[attrname] = value
        plt.rcParams.update(params)  # Apply these parameters
        if "projection" in kargs:
            projection = kargs["projection"]
            del kargs["projection"]
        else:
            projection = "rectilinear"
        if isinstance(figure, bool) and not figure:
            ret = None
        elif figure is not None:
            fig = plt.figure(figure, figsize=self.template_figure_figsize)
            if len(fig.axes) == 0:
                rect = [plt.rcParams["figure.subplot.{}".format(i)] for i in ["left", "bottom", "right", "top"]]
                rect[2] = rect[2] - rect[0]
                rect[3] = rect[3] - rect[1]
                if projection == "3d":
                    ax = fig.add_subplot(111, projection="3d")
                else:
                    ax = fig.add_axes(rect)
            else:
                ax = fig.gca(projection=projection)
            ret = fig
        else:
            if projection == "3d":
                ret = plt.figure(figsize=self.template_figure_figsize, **kargs)
                ax = ret.add_subplot(111, projection="3d")
            else:
                ret, ax = plt.subplots(figsize=self.template_figure_figsize, **kargs)
        return ret, ax

    def apply(self):
        """Scan for all attributes that start templtate_ and build them into a dictionary
        to update matplotlib settings with.
        """
        plt.style.use(self.stylesheet)

        self.customise()

    def customise(self):
        """This method is supplied for sub classes to override to provide additional
        plot customisation after the rc paramaters are updated from the class and
        instance attributes."""
        pass

    def customise_axes(self, ax, plot):
        """This method is run when we have an axis to manipulate.

        Args:
            ax (matplotlib axes): The axes to be modified by this function.

        Note:
            In the DefaultPlotStyle class this method is used to set SI units
            plotting mode for all axes.
        """
        ax.xaxis.set_major_locator(self.xlocater())
        ax.yaxis.set_major_locator(self.ylocater())
        ax.set_xticklabels(ax.get_xticks(), size=self.template_xtick_labelsize)
        ax.set_yticklabels(ax.get_yticks(), size=self.template_ytick_labelsize)
        if isinstance(self.xformatter, Formatter):
            xformatter = self.xformatter
        else:
            xformatter = self.xformatter()
        if isinstance(self.yformatter, Formatter):
            yformatter = self.yformatter
        else:
            yformatter = self.yformatter()

        ax.xaxis.set_major_formatter(xformatter)
        ax.yaxis.set_major_formatter(yformatter)
        if "zaxis" in dir(ax):
            ax.zaxis.set_major_locator(self.zlocater())
            ax.set_zticklabels(ax.get_zticks(), size=self.template_ztick_labelsize)
            ax.zaxis.set_major_formatter(self.zformatter())

    def annotate(self, ix, multiple, plot, **kargs):
        """Call all the routines necessary to annotate the axes etc.

        Args:
            ix(integer): Index of current subplot
            multiple (string): how to handle multiple subplots
            plot (Stoner.PlotFile): The PlotFile boject we're working with
        """

        if multiple in self.subplot_settings:
            if ix == 0:
                i = 0
            elif ix == len(plot.axes):
                i = 2
            else:
                i = 1
            settings = {k: self.subplot_settings[multiple][k][i] for k in self.subplot_settings[multiple]}
        else:
            settings = {"xlabel": True, "ylabel": True, "zlabel": True, "title": True}
        try:
            if "xlabel" in kargs and self.show_xlabel and settings["xlabel"]:
                plt.xlabel(str(kargs["xlabel"]), size=self.template_axes_labelsize)
            if "ylabel" in kargs and self.show_ylabel and settings["ylabel"]:
                plt.ylabel(str(kargs["ylabel"]), size=self.template_axes_labelsize)
            if "zlabel" in kargs and self.show_zlabel and settings["zlabel"]:
                plot.fig.axes[0].set_zlabel(kargs["zlabel"], size=self.template_axes_labelsize)
            if "title" in kargs and self.show_title and settings["title"]:
                plt.title(kargs["title"])
            if self.showlegend:
                plt.legend()
        except AttributeError:
            pass

class GBPlotStyle(DefaultPlotStyle):
    """Template developed for Gavin's plotting.

    This is largely an experimental class for trying things out rather than
    for serious plotting.

    Example
        .. plot:: samples/plotstyles/GBStyle.py
            :include-source:


    """
    xformatter = TexEngFormatter
    yformatter = TexEngFormatter
    stylename = "GBStyle"

    def customise_axes(self, ax, plot):
        """Override the default axis configuration"""
        super(GBPlotStyle, self).customise_axes(ax, plot)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.spines["left"].set_position('zero')
        ax.spines["bottom"].set_position('zero')
        plt.draw


class JTBPlotStyle(DefaultPlotStyle):
    """Template class for Joe's Plot settings.

    Example
        .. plot:: samples/plotstyles/JTBStyle.py
            :include-source:

"""

    show_title = False
    stylename = "JTB"

    def customise_axes(self, ax, plot):
        pass


class JTBinsetStyle(DefaultPlotStyle):
    """Template class for Joe's Plot settings."""

    show_title = False
    stylename = "JTBinset"

    def customise_axes(self, ax, plot):
        pass


class ThesisPlotStyle(DefaultPlotStyle):
    """Template class for Joe's Plot settings."""

    show_title = False
    stylename = "thesis"


class PRBPlotStyle(DefaultPlotStyle):
    """A figure Style for making figures for Phys Rev * Jounrals.

    Example
        .. plot:: samples/plotstyles/PRBStyle.py
            :include-source:
    """
    show_title = False
    stylename = "PRB"

    def customise_axes(self, ax, plot):
        ax.locator_params(tight=True, nbins=4)


class SketchPlot(DefaultPlotStyle):
    """Turn on xkcd plot style.

    Implemented as a bit of a joke, but perhaps someone will use this in a real
    presentation one day ?

    Example
        .. plot:: samples/plotstyles/SketchStyle.py
            :include-source:


    """
    stylename = "sketch"

    def customise(self):
        plt.xkcd()

    def customise_axes(self, ax, plot):
        """Override the default axis configuration"""
        super(SketchPlot, self).customise_axes(ax, plot)
        ax.spines["top"].set_visible(False)
        if len(plot.axes) > 1 and plot.multiple == "y2":
            pass
        else:
            ax.spines["right"].set_visible(False)
            ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.label.set_rotation(normal(scale=5))
        ax.xaxis.label.set_x(0.9)
        ax.yaxis.label.set_rotation(normal(90, scale=5))
        ax.yaxis.label.set_y(0.9)
        for l in ax.get_xticklabels():
            l.set_rotation(normal(scale=2))
        for l in ax.get_yticklabels():
            l.set_rotation(normal(scale=2))

        plt.draw

if SEABORN: # extra classes if we have seaborn available

    class SeabornPlotStyle(DefaultPlotStyle):
        """A plotdtyle that makes use of the seaborn plotting package to make visually attractive plots.

        Attributes:
            stylename (str): The seaborn plot stlye to use - darkgrid, whitegrid, dark, white, or ticks
            context (str): The seaborn plot context for scaling elements - paper,notebook,talk, or poster
            palette (str): A name of a predefined seaborn palette.

        Example
            .. plot:: samples/plotstyles/SeabornStyle.py
                :include-source:
        """


        _stylename=None
        _context=None
        _palette=None

        @property
        def context(self):
            return self._context

        @context.setter
        def context(self,name):
            if name in ["paper", "notebook", "talk", "poster"]:
                self._context=name
            else:
                raise AttributeError("style name should be one of  {paper,notebook,talk,poster}")

        @property
        def palette(self):
            return self._palette

        @context.setter
        def palette(self,name):
            try:
                with sns.color_palette(name):
                    pass
                self._palette=name
            except Exception as e:
                raise e

        @property
        def stylename(self):
            return self._stylename

        @stylename.setter
        def stylename(self,name):
            if name in ["darkgrid", "whitegrid", "dark", "white", "ticks"]:
                self._stylename=name
            else:
                raise AttributeError("style name should be one of  {darkgrid, whitegrid, dark, white, ticks}")


        def apply(self):
            """Override base method to apply seaborn style sheets."""
            sns.set_style(style=self.stylename)
            sns.set_context(context=self.context)
            sns.set_palette(sns.color_palette(self._palette))
            self.customise()
