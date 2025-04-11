# -*- coding: utf-8 -*-
"""Plot Templates module - contains classes that style plots produced by :class:`.Data`."""

__all__ = [
    "TexFormatter",
    "TexEngFormatter",
    "DefaultPlotStyle",
    "GBPlotStyle",
    "JTBPlotStyle",
    "JTBinsetStyle",
    "PRBPlotStyle",
    "SketchPlot",
    "SeabornPlotStyle",
]
from os.path import join, dirname, realpath, exists
from inspect import getfile
from collections.abc import MutableMapping, Mapping

import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, Formatter
from matplotlib.ticker import AutoLocator
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.random import normal

try:
    import seaborn as sns

    SEABORN = True
except ImportError:
    SEABORN = False


def _round(value, offset=2):
    """Round numbers for the TexFormatters to avoid crazy numbers of decimal places."""
    for i in range(5):
        vt = np.round(value, i)
        if np.abs(value - vt) < 10 ** (-i - offset):
            value = vt
            break
    return value


def _add_dots(key):
    """Replace __ with . in key."""
    return key.replace("__", ".").replace("..", "__")


def _remove_dots(key):
    return key.replace(".", "__")


class TexFormatter(Formatter):
    r"""An axis tick label formatter that emits Tex formula mode code.

    Formatting is set so that large numbers are registered as :math`\times 10^{power}`
    rather than using E notation."""

    def __call__(self, value, pos=None):
        """Return the value ina  suitable texable format."""
        if value is None or np.isnan(value):
            ret = ""
        elif value != 0.0:
            power = np.floor(np.log10(np.abs(value)))
            if np.abs(power) < 4:
                ret = f"${round(value)}$"
            else:
                v = _round(value / (10**power))
                ret = f"${v}\\times 10^{{{power:.0f}}}$"
        else:
            ret = "$0.0$"
        return ret

    def format_data(self, value):
        """Return the full string representation of the value with the position unspecified."""
        return self.__call__(value)

    def format_data_short(self, value):  # pylint: disable=r0201
        """Return a short string version of the tick value.

        Defaults to the position-independent long value."""
        return f"{value:g}"


class TexEngFormatter(EngFormatter):
    """An axis tick label formatter that emits Tex formula mode code.

    Formatting is set so that large numbers are registered as with SI prefixes
    rather than using E notation.
    """

    prefix = {
        0: "",
        3: "k",
        6: "M",
        9: "G",
        12: "T",
        15: "P",
        18: "E",
        21: "Z",
        24: "Y",
        -3: "m",
        -6: "\\mu",
        -9: "n",
        -12: "p",
        -15: "f",
        -18: "a",
        -21: "z",
        -24: "y",
    }

    def __call__(self, value, pos=None):
        """Return the value ina  suitable texable format."""
        if value is None or np.isnan(value):
            ret = ""
        elif value != 0.0:
            power = np.floor(np.log10(np.abs(value)))
            pre = np.ceil(power / 3.0) * 3
            if -1 <= power <= 3 or pre == 0:
                ret = f"${round(value, 4)}\\,\\mathrm{{{self.unit}}}$"
            else:
                power = power % 3
                v = _round(value / (10**pre), 4)
                if np.abs(v) < 0.1:
                    v *= 1000
                    pre -= 3
                elif np.abs(v) > 1000.0:
                    v /= 1000
                    pre += 3.0

                ret = f"${v}\\mathrm{{{self.prefix[int(pre)]} {self.unit}}}$"
        else:
            ret = "$0.0$"
        return ret

    def format_data(self, value):
        """Return the full string representation of the value with the position unspecified."""
        return self.__call__(value)

    def format_data_short(self, value):  # pylint: disable=r0201
        """Return a short string version of the tick value.

        Defaults to the position-independent long value."""
        return f"{value:g}"


class DefaultPlotStyle(MutableMapping):
    """Produces a default plot style.

    To produce alternative plot styles, create subclasses of this plot. Either override or
    create additional attributes to define rc Parameters (see Matplotlib documentation for
    available rc parameters) and override the :py:meth:Stoner.pot.formats.DefaultPlotStyle.customise`
    method to carry out additional plot formatting.

    Attributes:
        fig_width_pt (float): Preferred width of plot in points
        show_xlabel (bool): Show the title in the plot
        show_ylabel (bool): Show the x-axis Labels
        show_zlabel (bool): show the y-xaxis labels
        show_title (bool): show the title
        show_legend (bool): show the legend
        stylename (string): Name of the matplotlib style to use
        stylesheet (list): Calculated list of stylesheets found by traversing the class hierarchy

    Example
        .. plot:: samples/plotstyles/default.py
            :include-source:
            :outname: defaultstyle

    """

    # Internal class attributes.
    _inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    _mm_per_inch = 25.4
    _golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio

    # Settings for this figure type. All instance attributes which start template_
    # will be used. Once the leading template_ is stripped, all _ characters are replaced
    # with . and then the attributes are mapped to a dictionary and used to update the rcParams
    # dictionary

    # pylint: disable=attribute-defined-outside-init

    show_xlabel = True
    show_ylabel = True
    show_zlabel = True
    show_title = True
    show_legend = True
    xformatter = TexEngFormatter
    yformatter = TexEngFormatter
    zformatter = TexEngFormatter
    xlocater = AutoLocator
    ylocater = AutoLocator
    zlocater = AutoLocator
    stylename = "default"

    subplot_settings = {
        "panels": {
            "xlabel": (False, False, True),
            "ylabel": (True, True, True),
            "zlabel": (False, False, False),
            "title": (True, False, False),
        },
        "subplots": {
            "xlabel": (True, True, True),
            "ylabel": (True, True, True),
            "zlabel": (False, False, False),
            "title": (True, True, True),
        },
        "y2": {
            "xlabel": (True, False, False),
            "ylabel": (True, True, True),
            "zlabel": (False, False, False),
            "title": (True, False, False),
        },
    }

    def __init__(self, *args, **kargs):
        """Create a template instance of this template.

        Keyword arguments may be supplied to set default parameters. Any Matplotlib rc parameter
        may be specified, with .'s replaced with __. A Mapping type object may be supplied as the first argument
        which will be used to upodate the rcParams first.
        """
        self._stylesheet = None
        # self.fig_width = None
        # self.fig_height = None

        self.update(kargs)

    def __call__(self, **kargs):
        """Call the template object can manipulate the rcParams that will be set."""
        for k, v in kargs.items():
            if k.startswith("template_"):
                nk = _add_dots(k[:9])
                if nk in plt.rcParams:
                    super().__setattr__(nk, v)
                    self[nk] = v
            else:
                self.update({_add_dots(k): v})

    def __delitem__(self, name):
        """Clear any setting that overrides the default for *name*."""
        if hasattr(self, name):
            default = getattr(type(self)(), name)
            setattr(self, name, default)
        elif name in plt.rcParams:
            params = dict(plt.rcParams)
            del params[name]
            plt.rcdefaults()
            plt.rcParams.update(params)
            super().__delattr__(_remove_dots(f"template_{name}"))
        else:
            raise KeyError(f"{name} is not recognised as part of the template")

    def __getattr__(self, name):
        """Provide magic to read certain attributes of the template."""
        if name.startswith("template_"):  # Magic conversion to rcParams
            attrname = _add_dots(name[9:])
            if attrname in plt.rcParams:
                return plt.rcParams[attrname]
            raise AttributeError("template attribute not in rcParams")
        if name == "showlegend":
            return self.show_legend and len(plt.gca().get_legend_handles_labels()[1]) > 1
        return super().__getattribute__(name)

    def __getitem__(self, name):
        """Try to match *name* to a style setting."""
        try:
            return self.__getattr__(name)
        except AttributeError:
            pass
        if name in plt.rcParams:
            return plt.rcParams[name]
        raise KeyError(f"{name} is not recognised as part of the template")

    def __iter__(self):
        """Iterate over stylesjeet settings."""
        attrs = [x for x in dir(self) if self._allowed_attr(x)]
        attrs += list(plt.rcParams.keys())
        attrs.sort()
        for f in attrs:
            yield f

    def __len__(self):
        """Implement a length of stylesheet."""
        i = len([x for x in dir(self) if self._allowed_attr(x)])
        i += len(list(plt.rcParams.keys()))
        return i

    def __setattr__(self, name, value):
        """Ensure stylesheet can't be overwritten and provide magic for template attributes."""
        if name.startswith("template_"):
            attrname = _add_dots(name[9:])
            if attrname not in plt.rcParams.keys():
                raise AttributeError(f"{attrname} is not an attribute of matplotlib.rcParams!")
            plt.rcParams[attrname] = value
            super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def __setitem__(self, name, value):
        """Set a stylesheet setting by *name*."""
        if hasattr(self, name):
            setattr(self, name, value)
        else:
            if name in plt.rcParams:
                plt.rcParams[name] = value
                name = _remove_dots(f"template_{name}")
                super().__setattr__(name, value)
            else:
                raise KeyError(f"{name} is not recognised as part of the template")

    def _allowed_attr(self, x, template=False):
        """Private method to test if this is a template attribute we can set."""
        return (
            not x.startswith("_")
            and (not template) ^ x.startswith("template_")
            and not callable(x)
            and not isinstance(getattr(type(self), x, None), property)
        )

    @property
    def stylesheet(self):
        """Horribly hacky method to traverse over the class hierarchy for style sheet names."""
        if (
            self._stylesheet is not None and self._stylesheet[0] == self.stylename
        ):  # Have we cached a copy of our stylesheets ?
            return self._stylesheet[1]
        levels = type.mro(type(self))[:-1]
        sheets = []
        classes = []
        for c in levels:  # Iterate through all possible parent classes and build a list of stylesheets
            if c is type(self) or c in classes or not isinstance(c, DefaultPlotStyle):
                continue
            for f in [
                join(realpath(dirname(getfile(c))), c.stylename + ".mplstyle"),
                join(dirname(realpath(getfile(c))), "stylelib", c.stylename + ".mplstyle"),
                join(dirname(realpath(__file__)), "stylelib", c.stylename + ".mplstyle"),
            ]:  # Look in first of all the same directory as the class file and then in a stylib folder
                if exists(f):
                    sheets.append(f)
                    break
            else:  # Fallback, does the parent class define a builtin stylesheet ?
                if c.stylename in plt.style.available:
                    sheets.append(c.stylename)
            classes.append(c)  # Stop double visiting files
        # Now do the same for this class, but allow the stylename to be an instance variable as well
        for f in [
            join(dirname(realpath(getfile(type(self)))), self.stylename + ".mplstyle"),
            join(dirname(realpath(getfile(type(self)))), "stylelib", self.stylename + ".mplstyle"),
        ]:
            if exists(f):
                sheets.append(f)
                break
        else:
            if self.stylename in plt.style.available:
                sheets.append(self.stylename)

        self._stylesheet = self.stylename, sheets
        return sheets

    @stylesheet.setter
    def stylesheet(self, value):  # pylint: disable=r0201
        """Just stop the stylesheet from being set."""
        raise AttributeError("Can't set the stylesheet value, this is derived from the stylename aatribute.")

    def clear(self):
        """Reset everything back o defaults."""
        attrs = [x for x in dir(self) if self._allowed_attr(x)]
        defaults = type(self)()
        for attr in attrs:
            setattr(self, attr, getattr(defaults, attr))
        plt.rcdefaults()
        attrs = [x for x in dir(self) if self._allowed_attr(x, template=True)]
        for attr in attrs:
            delattr(self, attr)

    def update(self, other):
        """Update the template with new attributes from keyword arguments.

        Up to one positional argument may be supplied

        Keyword arguments may be supplied to set default parameters. Any Matplotlib rc parameter
        may be specified, with .'s replaced with _ and )_ replaced with __.
        """
        match other:
            case Mapping():
                for k, val in other.items():
                    if k in dir(self) and not callable(self.__getattr__(k)):
                        self.__setattr__(k, val)
                    elif not k.startswith("_"):
                        self.__setattr__("template_" + k, val)
            case _:
                raise SyntaxError(
                    "Only one posotional argument which should be a Mapping subclass can be supplied toi update."
                )

    def new_figure(self, figure=False, **kargs):
        """Create a new figure.

        This is called by PlotMixin to setup a new figure before we do anything."""
        plt.style.use("default")
        params = dict()
        self.apply()
        if "fig_width_pt" in dir(self):
            self.fig_width = self.fig_width_pt * self._inches_per_pt
        if "fig_height_pt" in dir(self):
            self.fig_height = self.fig_width * self._golden_mean  # height in inches
        if "fig_ratio" in dir(self) and "fig_width" in dir(self):
            self.fig_height = self.fig_width / self.fig_ratio
        if "fig_width" and "fig_height" in self.__dict__:
            self.template_figure__figsize = (self.fig_width, self.fig_height)
        for attr in dir(self):
            if attr.startswith("template_"):
                attrname = _add_dots(attr[9:])
                value = self.__getattribute__(attr)
                if attrname in plt.rcParams.keys():
                    params[attrname] = value
        projection = kargs.pop("projection", "rectilinear")
        self.template_figure__figsize = kargs.pop("figsize", self.template_figure__figsize)  # pylint: disable=W0201
        if "ax" in kargs and isinstance(kargs["ax"], (Axes3D, plt.Axes)):
            # Giving an axis instance in kargs means we can use that as our figure
            figure = kargs["ax"].figure.number
        if isinstance(figure, bool) and not figure:
            return None, None
        elif figure is not None:
            if figure in plt.get_fignums():
                fig = plt.figure(figure)
            else:
                fig = plt.figure(figure, figsize=self.template_figure__figsize, layout="constrained")
            if len(fig.axes) == 0:
                rect = [plt.rcParams[f"figure.subplot.{i}"] for i in ["left", "bottom", "right", "top"]]
                rect[2] = rect[2] - rect[0]
                rect[3] = rect[3] - rect[1]
                if projection == "3d":
                    if not kargs.get("no_axes", False):
                        ax = fig.add_subplot(111, projection="3d")
                    else:
                        ax = None
                else:
                    if not kargs.get("no_axes", False):
                        ax = fig.add_axes(rect)
                    else:
                        ax = None
            else:
                if projection == "3d":
                    if "ax" in kargs:
                        ax = kargs.pop("ax")
                    else:
                        for ax in plt.gcf().axes:
                            if isinstance(ax, Axes3D):
                                break
                        else:
                            ax = plt.axes(projection="3d")
                else:
                    ax = kargs.pop("ax", fig.gca())

            return fig, ax
        else:
            no_axes = kargs.pop("no_axes", False)
            if projection == "3d":
                kargs.setdefault("layout", "constrained")
                ret = plt.figure(figsize=self.template_figure__figsize, **kargs)
                if not no_axes:
                    ax = ret.add_subplot(111, projection="3d")
                    return ret, ax
                else:
                    for ax in ret.axes:
                        ax.remove()
                    return ret, None
            else:
                kargs.setdefault("layout", "constrained")
                if not no_axes:
                    return plt.subplots(figsize=self.template_figure__figsize, **kargs)
                else:
                    ret = plt.figure(figsize=self.template_figure__figsize, **kargs)
                    for ax in ret.axes:
                        ax.remove()
                    return ret, None

    def apply(self):
        """Update matplotlib rc parameters from any attributes starting template_."""
        plt.style.use(self.stylesheet)
        for attr in dir(self):
            v = getattr(self, attr)
            if not attr.startswith("template_"):
                continue
            attr = _add_dots(attr[9:])
            if attr in plt.rcParams:
                plt.rcParams[attr] = v

        self.customise()

    def customise(self):
        """Implement hook to customise plot.

        This method is supplied for sub classes to override to provide additional
        plot customisation after the rc parameters are updated from the class and
        instance attributes."""

    def customise_axes(self, ax, plot):
        """Implement hook for for when we have an axis to manipulate.

        Args:
            ax (matplotlib axes):
                The axes to be modified by this function.

        Note:
            In the DefaultPlotStyle class this method is used to set SI units
            plotting mode for all axes.
        """
        ax.xaxis.set_major_locator(self.xlocater())
        ax.yaxis.set_major_locator(self.ylocater())
        ax.set_xticks(ax.get_xticks())
        ax.set_yticks(ax.get_yticks())
        ax.set_xticklabels(ax.get_xticks(), size=self.template_xtick__labelsize)
        ax.set_yticklabels(ax.get_yticks(), size=self.template_ytick__labelsize)
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
            ax.set_zticklabels(ax.get_zticks(), size=self.template_ztick__labelsize)
            ax.zaxis.set_major_formatter(self.zformatter())

    def annotate(self, ix, multiple, plot, **kargs):
        """Call all the routines necessary to annotate the axes etc.

        Args:
            ix(integer):
                Index of current subplot
            multiple (string):
                how to handle multiple subplots
            plot (Stoner.plot.PlotMixin):
                The PlotMixin boject we're working with
        """
        if multiple in self.subplot_settings:
            if ix == 0:
                i = 0
            elif ix == len(plot.axes) - 1:
                i = 2
            else:
                i = 1
            settings = {k: self.subplot_settings[multiple][k][i] for k in self.subplot_settings[multiple]}
        else:
            settings = {"xlabel": True, "ylabel": True, "zlabel": True, "title": True}
        try:
            if "xlabel" in kargs and self.show_xlabel and settings["xlabel"]:
                plt.xlabel(str(kargs["xlabel"]), size=self.template_axes__labelsize)
            if "ylabel" in kargs and self.show_ylabel and settings["ylabel"]:
                plt.ylabel(str(kargs["ylabel"]), size=self.template_axes__labelsize)
            if "zlabel" in kargs and self.show_zlabel and settings["zlabel"]:
                plot.fig.axes[0].set_zlabel(kargs["zlabel"], size=self.template_axes__labelsize)
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

    Example:
        .. plot:: samples/plotstyles/GBStyle.py
            :include-source:
            :outname: gbstyle
    """

    xformatter = TexEngFormatter
    yformatter = TexEngFormatter
    stylename = "GBStyle"

    def customise_axes(self, ax, plot):
        """Override the default axis configuration."""
        super().customise_axes(ax, plot)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.spines["left"].set_position("zero")
        ax.spines["bottom"].set_position("zero")
        plt.draw()


class JTBPlotStyle(DefaultPlotStyle):
    """Template class for Joe's Plot settings.

    Example:
        .. plot:: samples/plotstyles/JTBStyle.py
            :include-source:
            :outname: jtbstyle
    """

    show_title = False
    stylename = "JTB"

    def customise_axes(self, ax, plot):
        """Override the default axis configuration (or not)."""


class JTBinsetStyle(DefaultPlotStyle):
    """Template class for Joe's Plot settings."""

    show_title = False
    stylename = "JTBinset"

    def customise_axes(self, ax, plot):
        """Override the default axis configuration (or not)."""


class ThesisPlotStyle(DefaultPlotStyle):
    """Template class for Joe's Plot settings."""

    show_title = False
    stylename = "thesis"


class PRBPlotStyle(DefaultPlotStyle):
    """A figure Style for making figures for Phys Rev * Jounrals.

    Example:
        .. plot:: samples/plotstyles/PRBStyle.py
            :include-source:
            :outname: prbstyle
    """

    show_title = False
    stylename = "PRB"

    def customise_axes(self, ax, plot):
        """Override the default axis configuration."""
        ax.locator_params(tight=True, nbins=4)


class SketchPlot(DefaultPlotStyle):
    """Turn on xkcd plot style.

    Implemented as a bit of a joke, but perhaps someone will use this in a real
    presentation one day ?

    Example:
        .. plot:: samples/plotstyles/SketchStyle.py
            :include-source:
            :outname: sketchstyle
    """

    stylename = "sketch"

    def customise(self):
        """Force on xkcd style."""
        plt.xkcd()

    def customise_axes(self, ax, plot):
        """Override the default axis configuration."""
        super().customise_axes(ax, plot)
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

        plt.draw()


if SEABORN:  # extra classes if we have seaborn available

    class SeabornPlotStyle(DefaultPlotStyle):
        """A plotdtyle that makes use of the seaborn plotting package to make visually attractive plots.

        Attributes:
            stylename (str):
                The seaborn plot style to use - darkgrid, whitegrid, dark, white, or ticks
            context (str):
                The seaborn plot context for scaling elements - paper,notebook,talk, or poster
            palette (str):
                A name of a predefined seaborn palette.

        Example:
            .. plot:: samples/plotstyles/SeabornStyle.py
                :include-source:
                :outname: seabornstyle
        """

        _stylename = None
        _context = None
        _palette = None

        @property
        def context(self):
            """Provide context getter."""
            return self._context

        @context.setter
        def context(self, name):
            """Limit context to allowed values."""
            if name in ["paper", "notebook", "talk", "poster"]:
                self._context = name
            else:
                raise AttributeError("style name should be one of  {paper,notebook,talk,poster}")

        @property
        def palette(self):
            """Provide palette getter."""
            return self._palette

        @palette.setter
        def palette(self, name):
            """Force palette to take allowed values."""
            with sns.color_palette(name):
                pass
            self._palette = name

        @property
        def stylename(self):
            """Provide getter for stylename."""
            return self._stylename

        @stylename.setter
        def stylename(self, name):
            """Force stylename to take allowed values only."""
            if name in ["darkgrid", "whitegrid", "dark", "white", "ticks"]:
                self._stylename = name
            else:
                raise AttributeError("style name should be one of  {darkgrid, whitegrid, dark, white, ticks}")

        def apply(self):
            """Override base method to apply seaborn style sheets."""
            sns.set_style(style=self.stylename)
            sns.set_context(context=self.context)
            sns.set_palette(sns.color_palette(self._palette))
            self.customise()

else:
    SeabornPlotStyle = DefaultPlotStyle
