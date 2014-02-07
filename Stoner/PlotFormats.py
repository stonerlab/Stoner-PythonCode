# -*- coding: utf-8 -*-
"""

Plot Templates module - contains classes that style plots for Stoner.Plot and pyplot
Created on Fri Feb 07 19:57:30 2014

@author: Gavin Burnell
"""

from Stoner.compat import *
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from numpy import sqrt


class DefaultPlotStyle(object):
    """Produces a default plot style.
    
    To produce alternative plot styles, create subclasses of this plot. Either override or
    create additional attributes to define rc Parameters (see Matplotlib documentation for
    available rc parameters) and override the :py:meth:Stoner.PlotFormats.DefaultPlotStyle.customise`
    method to carry out additional plot formatting.
    
    Attributes:
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
    _fig_width_pt = 433.62 # Get this from LaTeX using \showthe\columnwidth
    _inches_per_pt = 1.0/72.27               # Convert pt to inch
    _golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio


    """Settings for this figure type. All instance attributes which start template_
    will be used. Once the leading template_ is stripped, all _ characters are replaced
    with . and then the attributes are mapped to a dictionary and used to update the rcParams
    dictionary"""
    templat_axes_labelsize=12
    template_text_fontsize=12
    template_legend_fontsize=10
    template_xtick_labelsize=11
    template_ytick_labelsize=11
    template_xtick_direction='in'
    template_ytick_direction='in'
    template_xtick_major_size=5
    template_ytick_major_size=5
    template_font_family="Times New Roman"
    template_xtick_major_pad=4
    template_ytick_major_pad=4
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
        if "fig_width" not in kargs:
            self.fig_width=self._fig_width_pt*self._inches_per_pt
        if "fig_height" not in kargs:
            self.fig_height=self.fig_width*self._golden_mean      # height in inches
        self.template_figure_figsize =  [self.fig_width,self.fig_height]

    def apply(self):
        """Scan for all attributes that start templtate_ and build them into a dictionary
        to update matplotlib settings with.
        """
        params=dict()
        
        for attr in dir(self):
            if attr.startswith("template_"):
                attrname=attr[9:].replace("_",".").replace("..","_")
                value=self.__getattribute__(attr)
                params[attrname]=value
        plt.rcdefaults() #Reset to defaults
        plt.rcParams.update(params) # Apply these parameters
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
        ax.xaxis.set_major_formatter(EngFormatter())
        ax.yaxis.set_major_formatter(EngFormatter())

    

class JTBPlotStyle(DefaultPlotStyle):
    """Template class for Joe's Plot settings."""

    _fig_width_pt = 700.0 # Get this from LaTeX using \showthe\columnwidth
    templat_axes_labelsize=36
    template_text_fontsize=36
    template_legend_fontsize=24
    template_xtick_labelsize=28
    template_ytick_labelsize=28
    template_xtick_direction='in'
    template_ytick_direction='in'
    template_xtick_major_size=10
    template_ytick_major_size=10
    template_font_family="Arial"
    template_xtick_major_pad=20
    template_ytick_major_pad=20
    template_font_size=32
    template_lines_linewidth=4
    template_axes_formatter_limits=(-3, 3)
    template_figure_subplot_left=0.15
    template_figure_subplot_right=0.95
    template_figure_subplot_bottom=0.2
    template_figure_subplot_top=0.875 
    template_figure_autolayout=False 

