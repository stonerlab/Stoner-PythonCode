"""Provides the a class to facilitte easier plotting of Stoner Data.

Classes:
    PlotMixin:
        A class that uses matplotlib to plot data
"""

# pylint: disable=C0413
from __future__ import division

__all__ = ["PlotMixin"]
import copy
from collections.abc import Mapping
from functools import wraps

import numpy as np
from matplotlib import figure as mplfig
from matplotlib import pyplot as plt

from ..compat import int_types
from ..tools import (
    TypedList,
    all_type,
    fix_signature,
    get_option,
    isiterable,
)
from ..tools.decorators import class_modifier
from . import functions
from .formats import DefaultPlotStyle
from .utils import errorfill

try:  # Check we've got 3D plotting
    import mpl_toolkits.axisartist as AA  # noqa: F401 pylint: disable=unused-import
    from mpl_toolkits.axes_grid1 import (  # noqa: F401 pylint: disable=unused-import
        host_subplot,
        inset_locator,
    )
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 pylint: disable=unused-import

    _3D = True
except ImportError:
    _3D = False


@class_modifier([functions], adaptor=None, no_long_names=True, overload=True)
class PlotMixin:
    r"""A mixin class that works with :py:class:`Stoner.Core.DataFile` to add additional plotting functionality.

    Args:
        args(tuple):
            Arguments to pass to :py:meth:`Stoner.Core.DataFile.__init__`
        kargs (dict):
            keyword arguments to pass to \b DataFile.__init__

    Attributes:
        ax (matplotlib.Axes):
            The current axes on the current figure.
        axes (list of matplotlib.Axes):
            A list of all the axes on the current figure
        fig (matplotlib.figure):
            The current figure object being worked with
        fignum (int):
            The current figure's number.
        labels (list of string):
            List of axis labels as aternates to the column_headers
        showfig (bool or None):
            Controls whether plot functions return a copy of the figure (True), the DataFile (False) or
            Nothing (None)
        subplots (list of matplotlib.Axes):
            essentially the same as :py:attr:`PlotMixin.axes` but ensures that the list of subplots is
            synchronised to the number fo Axes.
        template (:py:class:`Sonter.plot.formats.DefaultPlotStyle` or instance):
            A plot style template subclass or object that determines the format and appearance of plots.
    """

    positional_fmt = [plt.plot, plt.semilogx, plt.semilogy, plt.loglog]
    no_fmt = [errorfill]
    _template = DefaultPlotStyle()
    _legend = True
    multiple = "common"

    def __init__(self, *args, **kargs):  # Do the import of plt here to speed module load
        """Import plt and then calls the parent constructor."""
        self._figure = None
        self._showfig = kargs.pop("showfig", True)  # Retains previous behaviour
        self._subplots = []
        self._public_attrs = {
            "fig": (int, mplfig.Figure),
            "labels": list,
            "template": (DefaultPlotStyle, type(DefaultPlotStyle)),
            "xlim": tuple,
            "ylim": tuple,
            "title": str,
            "xlabel": str,
            "ylabel": str,
            "_showfig": bool,
        }
        super().__init__(*args, **kargs)
        self._labels = TypedList(str, [])
        if self.debug:
            print("Done PlotMixin init")

    # ===========================================================================================================
    # Properties of PlotMixin
    # ===========================================================================================================

    @property
    def ax(self):
        """Return the current axis number."""
        return self.axes.index(self.fig.gca())

    @ax.setter
    def ax(self, value):
        """Change the current axis number."""
        if isinstance(value, int) and 0 <= value < len(self.axes):
            self.fig.sca(self.axes[value])
        else:
            raise IndexError("Figure doesn't have enough axes")

    @property
    def axes(self):
        """Return  the current axes object."""
        if isinstance(self._figure, mplfig.Figure):
            ret = self._figure.axes
        else:
            ret = None
        return ret

    @property
    def cmap(self):  # pylint: disable=r0201
        """Get the current cmap."""
        return plt.get_cmap()

    @cmap.setter
    def cmap(self, cmap):  # pylint: disable=r0201
        """Set the plot cmap."""
        return plt.set_cmap(cmap)

    @property
    def fig(self):
        """Get the current figure."""
        return self._figure

    @fig.setter
    def fig(self, value):
        """Set the current figure."""
        if isinstance(value, plt.Figure):
            self._figure = self.template.new_figure(value.number)[0]
        elif isinstance(value, int):
            value = plt.figure(value)
            self.fig = value
        elif value is None:
            self._figure = None
        else:
            raise NotImplementedError("fig should be a number of matplotlib figure")

    @property
    def fignum(self):
        """Return the current figure number."""
        return self._figure.number

    @property
    def labels(self):
        """Return the labels for the plot columns."""
        if len(self._labels) == 0:
            return self.column_headers
        if len(self._labels) < len(self.column_headers):
            self._labels.extend(copy.deepcopy(self.column_headers[len(self._labels) :]))
        return self._labels

    @labels.setter
    def labels(self, value):
        """Set the labels for the plot columns."""
        if value is None:
            self._labels = TypedList(str, self.column_headers)
        elif isiterable(value) and all_type(value, str):
            self._labels = TypedList(str, value)
        else:
            raise TypeError(f"labels should be iterable and all strings, or None, not {type(value)}")

    @property
    def showfig(self):
        """Return either the current figure or self or None.

        The return value depends on whether the attribute is True or False or None.
        """
        if self._showfig is None or get_option("no_figs"):
            return None
        if self._showfig:
            return self._figure
        return self

    @showfig.setter
    def showfig(self, value):
        """Force a figure to be displayed."""
        if not (value is None or isinstance(value, bool)):
            raise AttributeError(f"showfig should be a boolean value not a {type(value)}")
        self._showfig = value

    @property
    def subplots(self):
        """Return the subplot instances."""
        if self._figure is not None and len(self._figure.axes) > len(self._subplots):
            self._subplots = self._figure.axes
        return self._subplots

    @property
    def template(self):
        """Return the current plot template."""
        if not hasattr(self, "_template"):
            self.template = DefaultPlotStyle()
        return self._template

    @template.setter
    def template(self, value):
        """Set the current template."""
        if isinstance(value, type) and issubclass(value, DefaultPlotStyle):
            value = value()
        if isinstance(value, DefaultPlotStyle):
            self._template = value
        else:
            raise ValueError(f"Template is not of the right class:{type(value)}")
        self._template.apply()

    def _span_slice(self, col, num):
        """Create a slice that covers the range of a given column."""
        v1, v2 = self.span(col)
        v = np.linspace(v1, v2, num)
        delta = v[1] - v[0]
        if isinstance(delta, int_types):
            v2 = v2 + delta
        else:
            v2 = v2 + delta / 2
        return slice(v1, v2, delta)

    def _col_label(self, index, join=False):
        """Look up a column and see if it exists in self._lables, otherwise get from self.column_headers.

        Args:
            index (column index type):
                Column to return label for

        Returns:
            String type representing the column label.
        """
        ix = self.find_col(index)
        if isinstance(ix, list):
            if join:
                return ",".join([self._col_label(i) for i in ix])
            return [self._col_label(i) for i in ix]
        return self.labels[ix]

    def __dir__(self):
        """Handle the local attributes as well as the inherited ones."""
        attr = dir(type(self))
        attr.extend(super().__dir__())
        attr.extend(list(self.__dict__.keys()))
        attr.extend(["fig", "axes", "labels", "subplots", "template"])
        attr.extend(("xlabel", "ylabel", "title", "xlim", "ylim"))
        attr = list(set(attr))
        return sorted(attr)

    def __getattr__(self, name):
        """Wrap attribute access with extra renaming logic.

        Args:
            name (string):
                Name of attribute the following attributes are supported:
                    -   fig - the current plt figure reference
                    -   axes - the plt axes object for the current plot
                    -   xlim - the X axis limits
                    -   ylim - the Y axis limits

                    All other attributes are passed over to the parent class
        """
        func = None
        o_name = name
        mapping = dict(
            [
                ("plt_", (plt, "pyplot")),  # Need to be explicit in 2.7!
                ("ax_", (plt.Axes, "axes")),
                ("fig_", (plt.Figure, "figure")),
            ]
        )

        try:
            return object.__getattribute__(self, o_name)
        except AttributeError:
            pass

        if plt.get_fignums():  # Store the current figure and axes
            tfig = plt.gcf()
            tax = tfig.gca()  # protect the current axes and figure
        else:
            tfig = None
            tax = None

        # First look for a function in the pyplot package
        for prefix, (obj, key) in mapping.items():
            name = o_name[len(prefix) :] if o_name.startswith(prefix) else o_name
            if name in dir(obj):
                try:
                    return self._pyplot_proxy(name, key)
                except AttributeError:
                    pass

        # Nowcheck for prefixed access on axes and figures with get_
        if name.startswith("ax_") and f"get_{name[3:]}" in dir(plt.Axes):
            name = name[3:]
        if f"get_{name}" in dir(plt.Axes) and self._figure:
            ax = self.fig.gca()
            func = ax.__getattribute__(f"get_{name}")
        if name.startswith("fig_") and f"get_{name[4:]}" in dir(plt.Figure):
            name = name[4:]
        if f"get_{name}" in dir(plt.Figure) and self._figure:
            fig = self.fig
            func = fig.__getattribute__(f"get_{name}")

        if func is None:  # Ok Fallback to lookinf again at parent class
            return super().__getattr__(o_name)

        # If we're still here then we're calling a proxy from that figure or axes
        ret = func()
        if tfig is not None:  # Restore the current figures
            plt.figure(tfig.number)
            plt.sca(tax)
        return ret

    def _pyplot_proxy(self, name, what):
        """Proxy for accessing :py:module:`matplotlib.pyplot` functions."""
        if what not in ["axes", "figure", "pyplot"]:
            raise SyntaxError("pyplot proxy can't figure out what to get proxy from.")
        if what == "pyplot":
            obj = plt
        elif what == "figure" and self._figure:
            obj = self._figure
        elif what == "axes" and self._figure:
            obj = self._figure.gca()
        else:
            raise AttributeError(
                "Attempting to manipulate the methods on a figure or axes before a"
                + " figure has been created for this Data."
            )
        func = getattr(obj, name)

        if not callable(func):  # Bug out if this isn't a callable proxy!
            return func

        @wraps(func)
        def _proxy(*args, **kargs):
            ret = func(*args, **kargs)
            return ret

        return fix_signature(_proxy, func)

    def __setattr__(self, name, value):
        """Set the specified attribute.

        Args:
            name (string):
                The name of the attribute to set. The cuirrent attributes are supported:
                    -   fig - set the plt figure instance to use
                    -   xlabel - set the X axis label text
                    -   ylabel - set the Y axis label text
                    -   title - set the plot title
                    -   subtitle - set the plot subtitle
                    -   xlim - set the x-axis limits
                    -   ylim - set the y-axis limits

                Only "fig" is supported in this class - everything else drops through to the parent class
                value (any): The value of the attribute to set.
        """
        if plt.get_fignums():
            tfig = plt.gcf()
            if len(tfig.axes):
                tax = tfig.gca()  # protect the current axes and figure
            else:
                tax = None
        else:
            tfig = None
            tax = None
        func = None
        o_name = name
        if name.startswith("ax_") and f"set_{name[3:]}" in dir(plt.Axes):
            name = name[3:]
        if f"set_{name}" in dir(plt.Axes) and self.fig:
            if self.fig is None:  # oops we need a figure first!
                self.figure()
            ax = self.fig.gca()
            func = ax.__getattribute__(f"set_{name}")
        elif name.startswith("fig_") and f"set_{name[4:]}" in dir(plt.Figure):
            name = f"set_{name[4:]}"
            func = getattr(self.fig, name)
        elif f"set_{name}" in dir(plt.Figure) and self.fig:
            if self.fig is None:  # oops we need a figure first!
                self.figure()
            fig = self.fig
            func = fig.__getattribute__(f"set_{name}")

        if o_name in dir(type(self)) or func is None:
            try:
                return super().__setattr__(o_name, value)
            except AttributeError:
                pass

        if func is None:
            raise AttributeError(f"Unable to set attribute {o_name}")

        if isinstance(value, str) and "$" not in value:
            value = value.format(**self)

        if not isiterable(value) or isinstance(value, str):
            value = (value,)
        if isinstance(value, Mapping):
            func(**value)
        else:
            func(*value)
        if tfig is not None:
            plt.figure(tfig.number)
            if tax is not None:
                plt.sca(tax)
