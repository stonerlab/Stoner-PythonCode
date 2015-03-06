# -*- coding: utf-8 -*-
"""

Plot Templates module - contains classes that style plots for Stoner.Plot and pyplot
Created on Fri Feb 07 19:57:30 2014

@author: Gavin Burnell
"""

from Stoner.compat import *
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter,Formatter
from matplotlib.ticker import AutoLocator
from os.path import join,dirname,realpath


import numpy as _np_

class TexFormatter(Formatter):
    """An axis tick label formatter that emits Tex formula mode code
    so that large numbers are registered as \\times 10^{power}
    rather than ysing E notation."""

    def __call__(self, value, pos=None):
        """Return the value ina  suitable texable format"""
        if value is None or _np_.isnan(value):
            ret=""
        elif value!=0.0:
            power=_np_.floor(_np_.log10(_np_.abs(value)))
            if _np_.abs(power)<4:
                ret="${}$".format(value)
            else:
                v=value/(10**power)
                ret="${}\\times 10^{{{:.0f}}}$".format(v,power)
        else:
            ret="$0.0$"
        return ret

    def format_data(self,value):
        return self.__call__(value)

    def format_data_short(self,value):
        return "{:g}".format(value)

class TexEngFormatter(EngFormatter):
    """An axis tick label formatter that emits Tex formula mode code
    so that large numbers are registered as \\times 10^{power}
    rather than ysing E notation."""

    prefix={3:"k",6:"M",9:"G",12:"T",15:"P",18:"E",21:"Z",24:"Y",
            -3:"m",-6:"\\mu",-9:"n",-12:"p",-15:"f",-18:"a",-21:"z",-24:"y"}

    def __call__(self, value, pos=None):
        """Return the value ina  suitable texable format"""
        if value is None or _np_.isnan(value):
            ret=""
        elif value!=0.0:
            power=_np_.floor(_np_.log10(_np_.abs(value)))
            power=(_np_.sign(power)*(_np_.floor(_np_.abs(power/3.0)))-1)*3.0
            if _np_.abs(power)<3:
                ret="${}$".format(value)
            else:
                v=int(value/(10**power))
                ret="${}\\mathrm{{{} {}}}$".format(v,self.prefix[int(power)],self.unit)
        else:
            ret="$0.0$"
        return ret

    def format_data(self,value):
        return self.__call__(value)

    def format_data_short(self,value):
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

    """

    """Internal class attributes."""
    _inches_per_pt = 1.0/72.27               # Convert pt to inch
    _mm_per_inch = 25.4
    _golden_mean = (_np_.sqrt(5)-1.0)/2.0         # Aesthetic ratio


    """Settings for this figure type. All instance attributes which start template_
    will be used. Once the leading template_ is stripped, all _ characters are replaced
    with . and then the attributes are mapped to a dictionary and used to update the rcParams
    dictionary"""
    fig_width_pt = 433.62
    fig_width=None
    fig_height=None
    show_xlabel=True
    show_ylabel=True
    show_zlabel=True
    show_title=True
    show_legend=True
    xformatter=TexFormatter
    yformatter=TexFormatter
    zformatter=TexFormatter
    xlocater=AutoLocator
    ylocater=AutoLocator
    zlocater=AutoLocator
    stylename="default"
    def __init__(self,**kargs):
        """Create a template instance of this template.

        Keyword arguments may be supplied to set default parameters. Any Matplotlib rc parameter
        may be specified, with .'s replaced with _ and )_ replaced with __.
        """
        self.update(**kargs)
        self.apply()
        
    def __getattr__(self,name):
        if name=="stylesheet":
            return self._stylesheet()
        
    def _stylesheet(self):
        """Horribly hacky method to traverse over the class heirarchy for style sheet names."""
        levels=type.mro(type(self))[:-1]
        return [join(dirname(realpath(__file__)),"stylelib",c.stylename+".mplstyle") for c in levels[::-1]]
        
    def update(self,**kargs):
        """Update the template with new attributes from keyword arguments.
        Keyword arguments may be supplied to set default parameters. Any Matplotlib rc parameter
        may be specified, with .'s replaced with _ and )_ replaced with __.
        """
        for k in kargs:
            if not k.startswith("_"):
                self.__setattr__("template_"+k,kargs[k])
        if "fig_width" not in kargs and self.fig_width is None:
            self.fig_width=self.fig_width_pt*self._inches_per_pt
        if "fig_height" not in kargs and self.fig_height is None:
            self.fig_height=self.fig_width*self._golden_mean      # height in inches

    def new_figure(self,figure=False):
        """This is called by PlotFile to setup a new figure before we do anything."""
        params=dict()
        if self.fig_width is None:
            self.fig_width=self.fig_width_pt*self._inches_per_pt
        if self.fig_height is None:
            self.fig_height=self.fig_width*self._golden_mean      # height in inches
        self.template_figure_figsize =  (self.fig_width,self.fig_height)
        for attr in dir(self):
            if attr.startswith("template_"):
                attrname=attr[9:].replace("_",".").replace("..","_")
                value=self.__getattribute__(attr)
                if attrname in plt.rcParams.keys():
                    params[attrname]=value
        plt.rcdefaults() #Reset to defaults
        plt.rcParams.update(params) # Apply these parameters

        if isinstance(figure,bool) and not figure:
            ret=None
        elif figure is not None:
            ret=plt.figure(figure,figsize=self.template_figure_figsize)
        else:
            ret=plt.figure(figsize=self.template_figure_figsize)
        return ret

    def apply(self):
        """Scan for all attributes that start templtate_ and build them into a dictionary
        to update matplotlib settings with.
        """
        plt.style.use(self.stylesheet)
        self.new_figure(False)

        self.customise()

    def customise(self):
        """This method is supplied for sub classes to override to provide additional
        plot customisation after the rc paramaters are updated from the class and
        instance attributes."""

    def customise_axes(self,ax):
        """This method is run when we have an axis to manipulate.

        Args:
            ax (matplotlib axes): The axes to be modified by this function.

        Note:
            In the DefaultPlotStyle class this method is used to set SI units
            plotting mode for all axes.
        """
        ax.xaxis.set_major_locator(self.xlocater())
        ax.yaxis.set_major_locator(self.ylocater())
        ax.set_xticklabels(ax.get_xticks(),size=self.template_xtick_labelsize)
        ax.set_yticklabels(ax.get_yticks(),size=self.template_ytick_labelsize)
        if  isinstance(self.xformatter,Formatter):
            xformatter=self.xformatter
        else:
            xformatter=self.xformatter()
        if  isinstance(self.yformatter,Formatter):
            yformatter=self.yformatter
        else:
            yformatter=self.yformatter()

        ax.xaxis.set_major_formatter(xformatter)
        ax.yaxis.set_major_formatter(yformatter)
        if "zaxis" in dir(ax):
            ax.zaxis.set_major_locator(self.zlocater())
            ax.set_zticklabels(ax.get_zticks(),size=self.template_ztick_labelsize)
            ax.zaxis.set_major_formatter(self.zformatter())

    def annotate(self,plot,**kargs):
        """Call all the routines necessary to annotate the axes etc.

        Args:
            plot (Stoner.PlotFile): The PlotFile boject we're working with
        """
        if "xlabel" in kargs and self.show_xlabel:
            plt.xlabel(str(kargs["xlabel"]),size=self.template_axes_labelsize)
        if "ylabel" in kargs and self.show_ylabel:
            plt.ylabel(str(kargs["ylabel"]),size=self.template_axes_labelsize)
        if "zlabel" in kargs and self.show_zlabel:
            plot.fig.axes[0].set_zlabel(kargs["zlabel"],size=self.template_axes_labelsize)
        if "title" in kargs and self.show_title:
            plt.title(kargs["title"])
        if self.show_legend and len(plt.gca().get_legend_handles_labels()[1])>1:
            plt.legend()


class GBPlotStyle(DefaultPlotStyle):
    """Template developed for Gavin's plotting."""
    xformatter=TexEngFormatter
    yformatter=TexEngFormatter


class JTBPlotStyle(DefaultPlotStyle):
    """Template class for Joe's Plot settings."""

    fig_width_pt=244
    fig_height_pt=244
    show_title=False
    stylename="JTB"

    def customise_axes(self,ax):
        pass

class JTBinsetStyle(DefaultPlotStyle):
    """Template class for Joe's Plot settings."""

    fig_width_pt=244
    fig_height_pt=244
    show_title=False
    stylename="JTBinset"

    def customise_axes(self,ax):
        pass

class ThesisPlotStyle(DefaultPlotStyle):
    """Template class for Joe's Plot settings."""

    fig_width = 6.0
    fig_height= 4.0# 6"x4" plot
    show_title=False
    stylename="thesis"

class PRBPlotStyle(DefaultPlotStyle):
    """A figure Style for making figures for Phys Rev * Jounrals."""
    fig_width_pt=244
    show_title=False
    stylename="PRB"
    
    def customise_axes(self,ax):
        ax.locator_params(tight=True, nbins=4)
