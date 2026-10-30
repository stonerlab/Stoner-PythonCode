"""Share live targets, property forwarding and layout contexts across plot accessors."""

from collections.abc import Mapping
from contextlib import contextmanager

import stonerplots
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from stonerplots import (
    CentredAxes,
    DoubleXAxis,
    DoubleXYAxis,
    DoubleYAxis,
    InsetPlot,
    MultiPanel,
    OffsetPlots,
    PlotLabeller,
    SavedFigure,
    StackVertical,
    TexEngFormatter,
    TexFormatter,
)

from .properties import PlotProperty
from .state import PlotState, live_figure


class PlotAccessor:
    """Share owner-neutral rendering state without reading scientific values.

    Args:
        data (object):
            Scientific owner whose private dictionary stores transient plot state.

    Notes:
        Subclasses supply numerical preparation and drawing methods. Contexts
        retain their upstream signatures. No table or image storage is owned here.
    """

    #: Save figures and temporarily apply styles using stonerplots.
    SavedFigure = SavedFigure
    #: Arrange a regular or irregular grid, yielding the upstream panel sequence.
    MultiPanel = MultiPanel
    #: Arrange vertically stacked panels with shared X coordinates.
    StackVertical = StackVertical
    #: Offset plotted datasets vertically without changing their source data.
    OffsetPlots = OffsetPlots
    #: Add an independent right-hand Y axis, yielding its Matplotlib Axes.
    DoubleYAxis = DoubleYAxis
    #: Add an independent top X axis sharing Y, yielding its Matplotlib Axes.
    DoubleXAxis = DoubleXAxis
    #: Add independent top X and right Y axes, yielding one Matplotlib Axes.
    DoubleXYAxis = DoubleXYAxis
    #: Add an inset, yielding its Matplotlib Axes.
    InsetPlot = InsetPlot
    #: Place axes through a chosen origin using stonerplots.
    CentredAxes = CentredAxes
    #: Apply tick formatters and locators to axes selected by stonerplots.
    PlotLabeller = PlotLabeller
    #: Format ticks in mathematical notation.
    TexFormatter = TexFormatter
    #: Format ticks using engineering prefixes and mathematical notation.
    TexEngFormatter = TexEngFormatter

    @staticmethod
    @contextmanager
    def style(style=None, after_reset=False, *, extra=None):
        """Temporarily apply styles and rcParams overrides without creating figures.

        Args:
            style (str, path-like, dict, iterable or None):
                Style name, comma-separated names, style file, rcParams mapping
                or ordered sequence of these. Whitespace and empty entries in a
                comma-separated string are ignored. None uses
                ``stonerplots.default.style``, as does SavedFigure; an empty list
                or string applies no stylesheet.
            after_reset (bool):
                Apply styles after resetting to Matplotlib defaults. Defaults to
                False, retaining the caller's settings for unspecified properties.

        Keyword Arguments:
            extra (Mapping or None):
                Additional rcParams applied after all styles, as in SavedFigure.
                Defaults to None. These overrides take precedence over styles.

        Yields:
            None:
                Drawing inside the block inherits the temporary configuration.

        Raises:
            TypeError:
                If extra is not a mapping or None.
            KeyError:
                If an override names an unknown rcParam.
            ValueError:
                If Matplotlib rejects a parameter value.
            OSError:
                If a stylesheet cannot be found or read.

        Notes:
            Restore rcParams on exit, including exceptions and invalid overrides.
            This context does not select, create, save or close figures. Existing
            artists retain their properties. Lists containing dictionaries and
            path-like stylesheet arguments retain Matplotlib's native semantics.

        Example:
            >>> with d.plot.style("stoner, aps", extra={"lines.linewidth": 2}):
            ...     ax = d.plot()
        """
        if style is None:
            style = stonerplots.default.style
        elif isinstance(style, str):
            style = [name.strip() for name in style.split(",") if name.strip()]
        if extra is not None and not isinstance(extra, Mapping):
            raise TypeError("extra must be a mapping of Matplotlib rcParams, or None.")
        with plt.style.context(style, after_reset=after_reset), plt.rc_context(rc=extra):
            yield

    def __init__(self, data):
        """Bind to the owner without reading scientific values or creating rendering state."""
        self._data = data

    def __getattr__(self, name):
        """Resolve a public Matplotlib property without creating rendering targets."""
        if name.startswith("_"):
            raise AttributeError(name)
        return PlotProperty(name).__get__(self, type(self))

    def __setattr__(self, name, value):
        """Preserve accessor attributes and forward other public property setters."""
        if name.startswith("_") or hasattr(type(self), name):
            object.__setattr__(self, name, value)
        else:
            PlotProperty(name).__set__(self, value)

    def __dir__(self):
        """Include available Matplotlib properties in interactive completion."""
        names = set(super().__dir__())
        for target in (self.ax, self.figure):
            names.update(
                name[4:] for name in dir(target)
                if name.startswith(("get_", "set_")) and callable(getattr(target, name))
            )
        return sorted(names)

    @property
    def figure(self):
        """Return the recent live figure without changing pyplot's current target."""
        state = self._data.__dict__.get("_plot_state")
        return state.figure if state is not None and live_figure(state.figure) else None

    @property
    def ax(self):
        """Return the recent axes if it still belongs to the live figure."""
        state = self._data.__dict__.get("_plot_state")
        figure = self.figure
        return state.ax if figure is not None and state.ax in figure.axes else None

    @property
    def axes(self):
        """Return a list of the recent figure's axes, or an empty list."""
        figure = self.figure
        return list(figure.axes) if figure is not None else []

    def _record(self, figure, ax=None):
        """Record recent targets without changing the measurement's public attributes."""
        self._data.__dict__["_plot_state"] = PlotState(figure, ax)

    def new_figure(self, **kwargs):
        """Create and select an independent figure without creating axes.

        Keyword Arguments:
            **kwargs (dict):
                Arguments forwarded to matplotlib.pyplot.figure, such as figsize.

        Returns:
            matplotlib.figure.Figure:
                The new current figure. No existing figure is restyled.
        """
        if "num" in kwargs:
            raise ValueError("new_figure creates an independent figure; omit num.")
        figure = plt.figure(**kwargs)
        self._record(figure)
        return figure

    def show(self, *, block=None):
        """Show open Matplotlib figures, optionally overriding interactive blocking."""
        return plt.show(block=block)

    @staticmethod
    def _target(ax=None, projection="rectilinear"):
        """Use an explicit/current compatible axes, creating a figure only as needed."""
        if ax is not None:
            if not isinstance(ax, Axes) or not live_figure(ax.figure) or ax not in ax.figure.axes:
                raise ValueError("ax must be a live Matplotlib Axes.")
            if ax.name != projection:
                raise ValueError(f"ax requires projection {projection!r}, got {ax.name!r}.")
            return ax
        if plt.get_fignums():
            figure = plt.gcf()
            if not figure.axes:
                return figure.add_subplot(projection=projection)
            current = figure.gca()
            if current.name == projection:
                return current
        return plt.figure().add_subplot(projection=projection)
