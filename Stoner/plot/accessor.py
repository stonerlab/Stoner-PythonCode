"""Provide explicit, callable scientific plotting for a Data instance."""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from stonerplots import (
    CentredAxes,
    DoubleYAxis,
    InsetPlot,
    MultiPanel,
    PlotLabeller,
    SavedFigure,
    StackVertical,
    TexEngFormatter,
    TexFormatter,
)

from .selection import prepare_traces, resolve_columns, uncertainty_specs
from .state import PlotState, live_figure
from .utils import errorfill as draw_errorfill

__all__ = ["DataPlotter"]

_XY_KINDS = ("line", "scatter", "errorbar", "errorfill")
_AUTO = object()
_LEGACY_OPTIONS = {
    "figure", "template", "plotter", "multiple", "show_plot", "save_filename",
    "xcol", "ycol", "zcol", "ucol", "vcol", "wcol", "showfig",
}


class DataPlotter:
    """Resolve and draw scientific plots without changing measurement state.

    Args:
        data (Data):
            Owner of the live columns, exclusions and metadata used on each call.

    Notes:
        The ax and figure properties inspect recent live rendering targets; axes
        lists all axes in that figure. None of these properties creates a figure.
        The labels property supplies editable column-label overrides, defaulting
        to the current column headers.

        Drawing returns an Axes, independently of legacy showfig/no_figs options.
        Calls inherit Matplotlib's current axes/style unless ax is supplied. They
        neither show nor save figures. The context classes and tick formatters
        are direct stonerplots exports with their upstream signatures and handles.
        Automatic field plots currently use the retained legacy scientific helpers.
    """

    # Folder bulk calls inspect the name of their callable, including accessors.
    __name__ = "plot"

    #: Save figures and temporarily apply styles using stonerplots.
    SavedFigure = SavedFigure
    #: Arrange a regular or irregular grid, yielding the upstream panel sequence.
    MultiPanel = MultiPanel
    #: Arrange vertically stacked panels with shared X coordinates.
    StackVertical = StackVertical
    #: Add an independent right-hand Y axis, yielding its Matplotlib Axes.
    DoubleYAxis = DoubleYAxis
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
    #: Temporarily apply Matplotlib style names, files or rcParams dictionaries.
    style = staticmethod(plt.style.context)

    def __init__(self, data):
        """Bind to the owner without reading columns or creating rendering state."""
        self._data = data

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

    @property
    def labels(self):
        """Return mutable label overrides, filling unspecified entries from headers."""
        labels = self._data.__dict__.setdefault("_labels", [])
        headers = list(self._data.column_headers)
        labels.extend(headers[len(labels):])
        return labels

    @labels.setter
    def labels(self, value):
        """Set column labels, or use current headers when value is None."""
        if value is None:
            value = []
        if isinstance(value, str):
            raise TypeError("labels must be a sequence of strings, or None.")
        labels = list(value)
        if not all(isinstance(label, str) for label in labels):
            raise TypeError("labels must contain only strings.")
        self._data.__dict__["_labels"] = labels

    def _record(self, figure, ax=None):
        """Record recent targets and keep the remaining field helpers in sync."""
        self._data.__dict__["_plot_state"] = PlotState(figure, ax)
        self._data.__dict__["_figure"] = figure

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

    def __call__(self, x=None, y=None, *, kind="auto", z=None, u=None, v=None, w=None,
                 xerr=None, yerr=None, startx=0, no_guess=False, **kwargs):
        """Choose and draw a plot using explicit columns, roles and sensible defaults.

        Args:
            x, y (column selector or None):
                Coordinate columns; None uses assigned roles or column-count defaults.
                Y can select several columns. No third positional argument is accepted.

        Keyword Arguments:
            kind (str):
                auto infers the scientific plot; line, scatter, errorbar and
                errorfill select the corresponding explicit ordinary method.
            z (column selector or None):
                Scalar-field coordinate; False disables an assigned Z role.
            u (column selector or None):
                First vector component or voxel intensity column.
            v (column selector or None):
                Second vector component; False disables the assigned role.
            w (column selector or None):
                Third vector component; False disables the assigned role.
            xerr (column selector, float, array or None):
                None infers errors for automatic/error plots; False disables them.
                Integers select columns, floats give constants, and arrays contain
                symmetric (N,) or asymmetric (2, N) errors.
            yerr (column selector, float, array, sequence or None):
                Error specifications as for xerr. A sequence matching the trace
                count supplies specifications separately for each Y trace.
            startx (int):
                Position at which to select an assigned X group.
            no_guess (bool):
                Disable unassigned-table guesses: 2/3/4 columns mean XY/XYE/XDYE.
            **kwargs (dict):
                Ordinary plots accept ax, fmt, label, xlabel, ylabel, title,
                xscale, yscale and legend, with remaining keywords passed directly
                to the selected Matplotlib primitive. See the named methods.

        Returns:
            matplotlib.axes.Axes:
                Primary axes containing the plotted data, without displaying it.

        Raises:
            ValueError:
                If the kind, selectors, uncertainty input or target is unsuitable.
            TypeError:
                If more than two positional arguments are supplied.

        Notes:
            Explicit coordinates override roles. Column-count guesses are used
            only for otherwise unassigned tables and can infer uncertainty columns
            even when X/Y are explicitly supplied; kind='line' requests a plain line.
            Invalid/excluded samples are removed independently for each trace.
        """
        if kind not in ("auto", *_XY_KINDS):
            raise ValueError(f"Unknown plot kind {kind!r}; use auto or one of {_XY_KINDS}.")
        columns = resolve_columns(self._data, x=x, y=y, z=z, u=u, v=v, w=w, startx=startx, no_guess=no_guess)
        if kind == "auto" and columns.kind != "xy":
            return self._field(columns, **kwargs)
        if kind == "auto":
            xspec, yspecs = uncertainty_specs(columns, xerr, yerr)
            kind = "errorbar" if any(spec is not None and spec is not False for spec in (xspec, *yspecs)) else "line"
        if kind in ("line", "scatter"):
            if xerr is not None and xerr is not False or yerr is not None and yerr is not False:
                raise ValueError(f"{kind} does not draw uncertainty; use errorbar or errorfill.")
            xerr = yerr = False
        traces = prepare_traces(self._data, columns, xerr=xerr, yerr=yerr)
        return self._draw(kind, traces, xcolumn=columns.x, **kwargs)

    def line(self, x=None, y=None, **kwargs):
        """Draw plain lines, using current roles for omitted X/Y selectors.

        Args:
            x, y (column selector or None):
                Coordinates, with multiple Y columns allowed.

        Keyword Arguments:
            **kwargs (dict):
                ax selects an Axes; fmt and label can be strings or per-trace lists.
                xlabel, ylabel, title, xscale, yscale and legend control annotation.
                Other keywords are passed to Axes.plot. Assigned errors are ignored.

        Returns:
            matplotlib.axes.Axes:
                The axes containing the lines. Data and metadata are unchanged.
        """
        return self(x, y, kind="line", **kwargs)

    def scatter(self, x=None, y=None, **kwargs):
        """Draw XY points with Axes.scatter, ignoring assigned uncertainty roles.

        Args:
            x, y (column selector or None):
                Coordinates, with multiple Y columns allowed.

        Keyword Arguments:
            **kwargs (dict):
                ax and annotation keywords work as for line. Per-row c/s arrays
                are filtered with coordinate exclusions. fmt is unsupported; use
                marker, color and other Axes.scatter arguments.

        Returns:
            matplotlib.axes.Axes:
                The axes containing the points. Measurement state is unchanged.
        """
        return self(x, y, kind="scatter", **kwargs)

    def errorbar(self, x=None, y=None, *, xerr=None, yerr=None, **kwargs):
        """Draw XY data and aligned errors with Axes.errorbar.

        Args:
            x, y (column selector or None):
                Coordinate columns; omitted values use roles/defaults.

        Keyword Arguments:
            xerr (column selector, float, array or None):
                Error specifications as documented by the automatic call.
                Excluded/non-finite errors exclude their matching samples.
            yerr (column selector, float, array, sequence or None):
                Y uncertainties, optionally specified separately per trace.
            **kwargs (dict):
                ax, fmt, annotation controls and Axes.errorbar arguments.

        Returns:
            matplotlib.axes.Axes:
                The axes containing data and error bars, without changing measurements.
        """
        return self(x, y, kind="errorbar", xerr=xerr, yerr=yerr, **kwargs)

    def errorfill(self, x=None, y=None, *, xerr=None, yerr=None, **kwargs):
        """Draw XY lines with shaded uncertainty using the Stoner errorfill helper.

        Args:
            x, y (column selector or None):
                Coordinate columns; omitted values use roles/defaults.

        Keyword Arguments:
            xerr (column selector, float, array or None):
                Horizontal errors as for errorbar; use False to disable them.
            yerr (column selector, float, array, sequence or None):
                Vertical errors, optionally separate per trace. Only one direction
                of uncertainty is supported per trace; use False to disable the other.
            **kwargs (dict):
                ax, fmt, annotation controls and errorfill styling arguments,
                including alpha_fill for the shaded region.

        Returns:
            matplotlib.axes.Axes:
                The axes containing lines and shading, without changing measurements.
        """
        return self(x, y, kind="errorfill", xerr=xerr, yerr=yerr, **kwargs)

    def _text(self, value):
        """Interpolate metadata while retaining existing mathematical label strings."""
        return value.format(**self._data) if isinstance(value, str) and "$" not in value else value

    def _label(self, column):
        """Read an override or the current header without caching implicit defaults."""
        labels = self._data.__dict__.get("_labels", [])
        return labels[column] if column < len(labels) else self._data.column_headers[column]

    @staticmethod
    def _per_trace(value, count, name):
        """Validate per-trace format/legend labels before creating axes."""
        if isinstance(value, (list, tuple)):
            if len(value) != count:
                raise ValueError(f"{name} must contain one entry per selected Y column ({count}).")
            return list(value)
        return [value] * count

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

    def _draw(self, kind, traces, *, xcolumn, ax=None, fmt=None, label=_AUTO, xlabel=_AUTO, ylabel=_AUTO,
              title=_AUTO, xscale=None, yscale=None, legend=True, **kwargs):
        """Draw prepared ordinary traces without templates or pyplot render calls."""
        removed = _LEGACY_OPTIONS.intersection(kwargs)
        if removed:
            raise ValueError(f"Legacy plotting options {sorted(removed)} are unsupported; use ax= and the named methods.")
        formats = self._per_trace(fmt, len(traces), "fmt")
        labels = self._per_trace(label, len(traces), "label")
        if kind == "scatter" and fmt is not None:
            raise ValueError("scatter does not accept fmt; use marker= and color=.")
        if kind == "errorfill" and any(t.xerr is not None and t.yerr is not None for t in traces):
            raise ValueError("errorfill supports one uncertainty direction; disable xerr or yerr with False.")
        ax = self._target(ax)
        for trace, trace_fmt, trace_label in zip(traces, formats, labels):
            if not trace.x.size:
                continue
            options = {key: self._text(value) for key, value in kwargs.items()}
            options["label"] = self._text(self._label(trace.column) if trace_label is _AUTO else trace_label)
            self._draw_trace(ax, kind, trace, trace_fmt, options)
        self._annotate(ax, traces, xcolumn, xlabel, ylabel, title)
        if xscale is not None:
            ax.set_xscale(xscale)
        if yscale is not None:
            ax.set_yscale(yscale)
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend()
        self._record(ax.figure, ax)
        return ax

    @staticmethod
    def _draw_trace(ax, kind, trace, fmt, options):
        """Render one prepared trace using the requested Matplotlib primitive."""
        if kind == "line":
            args = () if fmt is None else (fmt,)
            ax.plot(trace.x, trace.y, *args, **options)
        elif kind == "scatter":
            for key in ("c", "s"):
                value = options.get(key)
                if value is not None and not isinstance(value, str) and np.ndim(value):
                    array = np.asarray(value)
                    if len(array) == len(trace.rows):
                        options[key] = array[trace.rows]
            ax.scatter(trace.x, trace.y, **options)
        elif kind == "errorbar":
            ax.errorbar(trace.x, trace.y, xerr=trace.xerr, yerr=trace.yerr,
                        fmt="" if fmt is None else fmt, **options)
        else:
            draw_errorfill(trace.x, trace.y, xerr=trace.xerr, yerr=trace.yerr, ax=ax, fmt=fmt, **options)

    def _annotate(self, ax, traces, xcolumn, xlabel, ylabel, title):
        """Fill missing source labels while retaining explicit caller annotations."""
        if xlabel is not _AUTO or not ax.get_xlabel():
            ax.set_xlabel(self._text(self._label(xcolumn) if xlabel is _AUTO else xlabel or ""))
        if ylabel is not _AUTO or not ax.get_ylabel():
            ax.set_ylabel(self._text(", ".join(self._label(t.column) for t in traces) if ylabel is _AUTO else ylabel or ""))
        if title is not _AUTO or not ax.get_title():
            ax.set_title(self._text(self._data.basename if title is _AUTO else title or ""))

    def _field(self, columns, **kwargs):
        """Keep automatic scientific fields available until their stage 3 port."""
        from . import functions

        helpers = {"surface": functions.plot_xyz, "vector": functions.plot_xyuv,
                   "vector3d": functions.plot_xyzuvw, "voxels": functions.plot_voxels}
        selector_names = {"surface": "xyz", "vector": "xyuvw", "vector3d": "xyzuvw", "voxels": "xyzu"}
        kind = columns.kind
        target = kwargs.pop("ax", None)
        if target is not None:
            if not isinstance(target, Axes) or not live_figure(target.figure):
                raise ValueError("ax must be a live Matplotlib Axes.")
            plt.sca(target)
            self._data.__dict__["_figure"] = target.figure
        selectors = {"xcol": columns.x}
        for name in selector_names[kind][1:]:
            values = getattr(columns, name)
            selectors[f"{name}col"] = values[0] if values else None
        kwargs["show_plot"] = False
        helpers[kind](self._data, **selectors, **kwargs)
        figure = self._data.__dict__["_figure"]
        ax = figure.gca()
        self._record(figure, ax)
        return ax
