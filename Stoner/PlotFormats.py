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
import numpy as _np_

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
        templat_axes_labelsize (int): Axes Label size
        template_text_fontsize (int): Text font size
        template_legend_fontsize (int): Legend font size
        template_xtick_labelsize (int): X-axis tick label sizes
        template_ytick_labelsize (int): Y-axis tick label sizes
        template_xtick_direction ("in", "out", "both"): x-axis tick directions
        template_ytick_direction ("in", "out", "both"): y-axis tick directions
        template_xtick_major_size (int): x-axis tick size
        template_ytick_major_size (int): y-axis tick size
        template_font_family (string): Font for text
        template_xtick_major.pad (int): Padding between ticks and labels in x-axis
        template_ytick_major.pad (int): Padding between ticks and labels in y-axis
        template_font_size (int): Default font size
        template_lines_linewidth (int): Line width size
        template_axes_formatter_limits (tuple): Use scientific notations outside of data=log10(value)
         template_axes_grid (bool): Show grids on plot
        template_axes_color__cycle (list): Set of colors to cycle through for plots
        template_figure_facecolor (color): Override the grey colour of the plot border
        template_figure_subplot_left (float): Set the left margin
        template_figure_subplot_right (float): Set the right margin
        template_figure_subplot_bottom (float): Set the bottom margin
        template_figure_subplot_top (float): Set the top margin


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
    template_axes_labelsize=12
    template_text_fontsize=12
    template_legend_fontsize=10
    template_legend_frameon=False
    template_xtick_labelsize=11
    template_ytick_labelsize=11
    template_ztick_labelsize=11
    template_xtick_direction='in'
    template_ytick_direction='in'
    template_ztick_direction='in'
    template_xtick_major_size=5
    template_ytick_major_size=5
    template_ztick_major_size=5
    template_xtick_major_pad=4
    template_ytick_major_pad=4
    template_ztick_major_pad=4
    template_font_size=14
    template_lines_linewidth=1
    template_axes_formatter_limits=(-1, 1)
    template_axes_grid=False
    template_axes_color__cycle=['k','r','g','b','c','m','y']
    template_figure_facecolor=(1,1,1)
    template_figure_subplot_left=0.15
    template_figure_subplot_right=0.9
    template_figure_subplot_bottom=0.15
    template_figure_subplot_top=0.9

    def __init__(self,**kargs):
        """Create a template instance of this template.

        Keyword arguments may be supplied to set default parameters. Any Matplotlib rc parameter
        may be specified, with .'s replaced with _ and )_ replaced with __.
        """
        self.update(**kargs)
        self.apply()

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


class JTBPlotStyle(DefaultPlotStyle):
    """Template class for Joe's Plot settings."""

    fig_width_pt=244
    fig_height_pt=244
    show_title=False

    template_text_fontsize=18
    template_legend_fontsize=18
    template_xtick_labelsize=18
    template_ytick_labelsize=18
    template_xtick_direction='in'
    template_ytick_direction='in'
    template_xtick_major_size=3
    template_ytick_major_size=3
    template_font_family="Times New Roman"
    template_xtick_major_pad=5
    template_ytick_major_pad=5
    template_font_size=18
    template_lines_linewidth=2
    template_lines_linestyle='-'
    template_lines_markeredgewidth=0
    #template_lines_marker=itertools.cycle(('o','s','^','v','x'))
    template_lines_marker=''
    template_lines_markersize=5


    template_axes_labelsize=18
    template_axes_fontsize=18
    template_axes_labelpad=1
    template_axes_formatter_limits=(-4, 4)
    template_axes_grid=False
    template_axes_color__cycle=['k','r','b','g','c','m','y']

    template_figure_facecolor=(1,1,1)
    template_figure_autolayout=True
    template_figure_subplot_left    = 0.175  # the left side of the subplots of the figure
    template_figure_subplot_right   = 0.9    # the right side of the subplots of the figure
    template_figure_subplot_bottom  = 0.2    # the bottom of the subplots of the figure
    template_figure_subplot_top     = 0.95    # the top of the subplots of the figure
    template_figure_subplot_wspace  = -20
    template_figure_subplot_hspace  = -10
    template_lines_markersize=5

    template_legend_loc          ='upper center'
    template_legend_isaxes       =False
    template_legend_numpoints    =1      # the number of points in the legend line
    template_legend_fontsize     =15
    template_legend_borderpad    =0    # border whitespace in fontsize units
    template_legend_markerscale  =1.0    # the relative size of legend markers vs. original
    template_legend_labelspacing =0.5    # the vertical space between the legend entries in fraction of fontsize
    template_legend_handlelength =2.     # the length of the legend lines in fraction of fontsize
    template_legend_handleheight =0.7     # the height of the legend handle in fraction of fontsize
    template_legend_handletextpad=0.8    # the space between the legend line and legend text in fraction of fontsize
    template_legend_borderaxespad=0.5   # the border between the axes and legend edge in fraction of fontsize
    template_legend_columnspacing=2.    # the border between the axes and legend edge in fraction of fontsize
    template_legend_shadow       =False
    template_legend_frameon      =False   # whether or not to draw a frame around legend
    template_legend_scatterpoints=3 # number of scatter points


    def customise_axes(self,ax):
        pass

class JTBinsetStyle(DefaultPlotStyle):
    """Template class for Joe's Plot settings."""

    fig_width_pt=244
    fig_height_pt=244
    show_title=False

    template_text_fontsize=10
    template_legend_fontsize=10
    template_xtick_labelsize=10
    template_ytick_labelsize=10
    template_xtick_direction='in'
    template_ytick_direction='in'
    template_xtick_major_size=3
    template_ytick_major_size=3
    template_font_family="Times New Roman"
    template_xtick_major_pad=1
    template_ytick_major_pad=1
    template_font_size=10
    template_lines_linewidth=2
    template_lines_linestyle='-'
    template_lines_markeredgewidth=0
    #template_lines_marker=itertools.cycle(('o','s','^','v','x'))
    template_lines_marker=''
    template_lines_markersize=5


    template_axes_labelsize=10
    template_axes_fontsize=10
    template_axes_labelpad=-10
    template_axes_formatter_limits=(-4, 4)
    template_axes_grid=False
    template_axes_color__cycle=['k','r','b','g','c','m','y']

    template_figure_facecolor=(1,1,1)
    template_figure_autolayout=True
    template_figure_subplot_left    = 0.175  # the left side of the subplots of the figure
    template_figure_subplot_right   = 0.9    # the right side of the subplots of the figure
    template_figure_subplot_bottom  = 0.2    # the bottom of the subplots of the figure
    template_figure_subplot_top     = 0.95    # the top of the subplots of the figure
    template_figure_subplot_wspace  = -20
    template_figure_subplot_hspace  = -10
    template_lines_markersize=5

    template_legend_loc          ='upper center'
    template_legend_isaxes       =False
    template_legend_numpoints    =1      # the number of points in the legend line
    template_legend_fontsize     =10
    template_legend_borderpad    =-1    # border whitespace in fontsize units
    template_legend_markerscale  =1.0    # the relative size of legend markers vs. original
    template_legend_labelspacing =0.5    # the vertical space between the legend entries in fraction of fontsize
    template_legend_handlelength =2.     # the length of the legend lines in fraction of fontsize
    template_legend_handleheight =0.7     # the height of the legend handle in fraction of fontsize
    template_legend_handletextpad=0.8    # the space between the legend line and legend text in fraction of fontsize
    template_legend_borderaxespad=0.5   # the border between the axes and legend edge in fraction of fontsize
    template_legend_columnspacing=2.    # the border between the axes and legend edge in fraction of fontsize
    template_legend_shadow       =False
    template_legend_frameon      =False   # whether or not to draw a frame around legend
    template_legend_scatterpoints=3 # number of scatter points


    def customise_axes(self,ax):
        pass

class ThesisPlotStyle(DefaultPlotStyle):
    """Template class for Joe's Plot settings."""

    fig_width = 6.0
    fig_height= 4.0# 6"x4" plot
    show_title=False
    templat_axes_labelsize=11
    template_text_fontsize=11
    template_legend_fontsize=11
    template_xtick_labelsize=11
    template_ytick_labelsize=11
    template_xtick_direction='in'
    template_ytick_direction='in'
    template_xtick_major_size=8
    template_ytick_major_size=8
    template_font_family="Times"
    template_xtick_major_pad=5
    template_ytick_major_pad=5
    template_font_size=9
    template_lines_linewidth=2
    template_axes_formatter_limits=(-5, 5)
    template_figure_subplot_left=0.15
    template_figure_subplot_right=0.95
    template_figure_subplot_bottom=0.2
    template_figure_subplot_top=0.875
    template_figure_autolayout=False


class PRBPlotStyle(DefaultPlotStyle):
    """A figure Style for making figures for Phys Rev * Jounrals."""
    fig_width_pt=244
    show_title=False
    templat_axes_labelsize=10
    template_text_fontsize=10
    template_legend_fontsize=10
    template_xtick_labelsize=10
    template_ytick_labelsize=10
    template_xtick_direction='in'
    template_ytick_direction='in'
    template_xtick_major_size=5
    template_ytick_major_size=5
    template_font_family="Times New Roman"
    template_xtick_major_pad=2
    template_ytick_major_pad=2
    template_font_size=10
    template_lines_linewidth=1
    template_axes_formatter_limits=(-4, 4)
    template_axes_grid=False
    template_axes_color__cycle=['k','r','g','b','c','m','y']
    template_figure_facecolor=(1,1,1)
    template_figure_autolayout=True
    template_lines_markersize=3
    template_legend_isaxes       =True
    template_legend_numpoints    =2      # the number of points in the legend line
    template_legend_fontsize     =9
    template_legend_borderpad    =0    # border whitespace in fontsize units
    template_legend_markerscale  =1.0    # the relative size of legend markers vs. original
    template_legend_labelspacing =0.5    # the vertical space between the legend entries in fraction of fontsize
    template_legend_handlelength =2.     # the length of the legend lines in fraction of fontsize
    template_legend_handleheight =0.7     # the height of the legend handle in fraction of fontsize
    template_legend_handletextpad=0.8    # the space between the legend line and legend text in fraction of fontsize
    template_legend_borderaxespad=0.5   # the border between the axes and legend edge in fraction of fontsize
    template_legend_columnspacing=2.    # the border between the axes and legend edge in fraction of fontsize
    template_legend_shadow       =False
    template_legend_frameon      =False   # whether or not to draw a frame around legend
    template_legend_scatterpoints=3 # number of scatter points

    def customise_axes(self,ax):
        ax.locator_params(tight=True, nbins=4)
