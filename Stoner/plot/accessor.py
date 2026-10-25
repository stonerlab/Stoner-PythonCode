"""Provide explicit, callable scientific plotting for a Data instance."""

from collections.abc import Mapping
from contextlib import contextmanager

import numpy as np
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

from .fields import draw_field, draw_matrix
from .selection import prepare_traces, resolve_columns, uncertainty_specs
from .state import PlotState, live_figure
from .statistics import draw_distribution, draw_hexbin
from .utils import errorfill as draw_errorfill

__all__ = ["DataPlotter"]

_XY_KINDS = ("line", "scatter", "errorbar", "errorfill")
_FIELD_KINDS = (
    "surface",
    "contour",
    "pcolormesh",
    "image",
    "matrix",
    "scatter3d",
    "vector",
    "vector3d",
    "quiver",
    "voxels",
    "waterfall",
)
_STAT_KINDS = ("hist", "box", "kde", "hexbin")
_AUTO = object()
_LEGACY_OPTIONS = {
    "figure",
    "template",
    "plotter",
    "multiple",
    "show_plot",
    "save_filename",
    "xcol",
    "ycol",
    "zcol",
    "ucol",
    "vcol",
    "wcol",
    "showfig",
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
        Field methods use Matplotlib and return their primary Axes as well.
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
        labels.extend(headers[len(labels) :])
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
        """Record recent targets without changing the measurement's public attributes."""
        self._data.__dict__["_plot_state"] = PlotState(figure, ax)

    def _projection_for(self, x=None, y=None, *, kind="auto", **kwargs):
        """Resolve a folder member's required projection without creating axes."""
        if kind not in ("auto", *_XY_KINDS, *_FIELD_KINDS, *_STAT_KINDS):
            raise ValueError(f"Unknown plot kind {kind!r}.")
        if kind == "auto":
            selections = {key: kwargs[key] for key in ("z", "u", "v", "w", "startx", "no_guess") if key in kwargs}
            kind = resolve_columns(self._data, x=x, y=y, **selections).kind
        return "3d" if kind in ("surface", "matrix", "scatter3d", "vector3d", "voxels") else "rectilinear"

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

    def __call__(
        self,
        x=None,
        y=None,
        *,
        kind="auto",
        z=None,
        u=None,
        v=None,
        w=None,
        xerr=None,
        yerr=None,
        startx=0,
        no_guess=False,
        **kwargs,
    ):
        """Choose and draw a plot using explicit columns, roles and sensible defaults.

        Args:
            x, y (column selector or None):
                Coordinate columns; None uses assigned roles or column-count defaults.
                Y can select several columns. No third positional argument is accepted.

        Keyword Arguments:
            kind (str):
                auto infers the scientific plot. An explicit kind selects its
                named method, including fields and hist/box/KDE/hexbin plots.
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
        if kind not in ("auto", *_XY_KINDS, *_FIELD_KINDS, *_STAT_KINDS):
            raise ValueError(f"Unknown plot kind {kind!r}; use auto or a named plotting method.")
        if kind in _STAT_KINDS:
            if any(value is not None and value is not False for value in (z, u, v, w, xerr, yerr)):
                raise ValueError("Statistical plots do not accept field or uncertainty selectors.")
            annotations = self._stat_options(kind, kwargs)
            if kind != "hexbin":
                if x is not None:
                    raise ValueError(f"{kind} selects sample columns with y=; X is not required.")
                return draw_distribution(
                    self, kind, y, startx=startx, no_guess=no_guess, annotations=annotations, **kwargs
                )
            columns = resolve_columns(
                self._data, x=x, y=y, z=False, u=False, v=False, w=False, startx=startx, no_guess=no_guess
            )
            return draw_hexbin(self, columns, annotations=annotations, **kwargs)
        if kind == "matrix":
            if any(value is not None for value in (z, u, v, w, xerr, yerr)):
                raise ValueError("matrix uses x/y coordinates and a table block, not field or uncertainty selectors.")
            return self._matrix(x, y, **kwargs)
        if kind in _FIELD_KINDS and not no_guess and not any(role != "." for role in self._data.setas):
            required = {"vector": "xyuv", "quiver": "xyuv", "vector3d": "xyzuvw", "voxels": "xyzu"}.get(kind, "xyz")
            selections = {"x": x, "y": y, "z": z, "u": u, "v": v, "w": w}
            for index, name in enumerate(required):
                if selections[name] is None and index < self._data.shape[1]:
                    selections[name] = index
            x, y, z, u, v, w = (selections[name] for name in "xyzuvw")
        columns = resolve_columns(self._data, x=x, y=y, z=z, u=u, v=v, w=w, startx=startx, no_guess=no_guess)
        if kind == "auto" and columns.kind != "xy":
            kind = columns.kind
        if kind in _FIELD_KINDS:
            if xerr is not None and xerr is not False or yerr is not None and yerr is not False:
                raise ValueError(
                    "Field plots do not draw uncertainties; omit xerr/yerr or use an ordinary error plot."
                )
            return self._field(kind, columns, **kwargs)
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

    def hist(self, y=None, **kwargs):
        """Draw histograms from independently valid samples using Axes.hist.

        Args:
            y (column selector or None):
                Sample columns, in Stoner selector order. Defaults to assigned Y
                columns in the selected X group, or all real numeric columns if
                none are assigned. No separate X coordinate is required.

        Keyword Arguments:
            **kwargs (dict):
                ax selects a 2D Axes. startx selects a role group; no_guess=True
                disables the numeric-column fallback. weights is one column
                selector or a source-row array, filtered separately per series.
                label supplies legend labels; None suppresses them. legend=True,
                xlabel, ylabel, title, xscale and yscale control presentation.
                Other keywords go to Axes.hist, including bins, range, density,
                cumulative, stacked, histtype and orientation.

        Returns:
            matplotlib.axes.Axes:
                Primary axes containing the distributions, without changing data.

        Notes:
            Each selected column independently excludes masked/non-finite values
            and matching invalid weights. Assigned errors and unrelated masks are
            ignored. Every selected series must retain samples. Bins are shared
            across series; stacked means adding bin values, not arranging panels.
            Native histogram weights may be negative. Drawing does not show/save.
        """
        return self(y=y, kind="hist", **kwargs)

    def box(self, y=None, **kwargs):
        """Draw one box per independently filtered sample column.

        Args:
            y (column selector or None):
                Sample columns with the same role/numeric defaults as hist.

        Keyword Arguments:
            **kwargs (dict):
                ax, startx, no_guess and annotation controls work as for hist.
                label supplies category labels, defaulting to column labels;
                None clears them. orientation is vertical or horizontal. Native
                Axes.boxplot options include whis, showfliers, notch, positions,
                widths, patch_artist and box/median/whisker properties. Weights,
                grouping and layout forwarding are unsupported.

        Returns:
            matplotlib.axes.Axes:
                Primary 2D Axes. Only selected samples determine box statistics;
                no values, exclusions, roles or metadata are changed.
        """
        return self(y=y, kind="box", **kwargs)

    def kde(self, y=None, **kwargs):
        """Estimate and draw a Gaussian probability density for each sample column.

        Args:
            y (column selector or None):
                Sample columns with the same role/numeric defaults as hist.

        Keyword Arguments:
            **kwargs (dict):
                ax, weights, label, legend and annotation controls work as for
                hist. bw_method is SciPy gaussian_kde's bandwidth factor, name
                or callable (None uses Scott's rule). ind is an integer point
                count of at least two, or a finite evaluation vector. By default
                each series uses 1000 points, extending half its observed range
                beyond each end. Other keywords go to Axes.plot.

        Returns:
            matplotlib.axes.Axes:
                Primary 2D Axes containing density curves, without changing data.

        Raises:
            ValueError:
                If a series has fewer than two distinct effective samples,
                weights are negative/all zero, or bandwidth/evaluation is invalid.

        Notes:
            Exclusions are independent per series. Zero-weight samples do not
            determine the estimator or default range. Gaussian kernels extend
            beyond the observed data; density is not clipped at a physical bound.
        """
        return self(y=y, kind="kde", **kwargs)

    def hexbin(self, x=None, y=None, *, C=None, **kwargs):
        """Bin jointly valid XY samples into hexagons with Axes.hexbin.

        Args:
            x, y (column selector or None):
                One column each, with ordinary scientific role/count defaults.

        Keyword Arguments:
            C (column selector, array or None):
                Optional reduced values, as one column or a source-row array.
                None counts points. Masks/non-finite values combine with X/Y.
            **kwargs (dict):
                ax selects a 2D Axes. colorbar=True adds a colour scale. xlabel,
                ylabel and title override source annotations. Other arguments go
                to Axes.hexbin, including gridsize, mincnt, reduce_C_function,
                extent, bins, cmap, norm and xscale/yscale for logarithmic bins.
                Assigned errors and unrelated column masks are ignored.

        Returns:
            matplotlib.axes.Axes:
                Primary axes, retained as the recent target after adding a colour
                bar. Counts/reductions use only finite, unexcluded joint samples.
        """
        return self(x, y, kind="hexbin", C=C, **kwargs)

    @staticmethod
    def _stat_options(kind, kwargs):
        """Reject backend/group/layout forwarding before selecting statistical axes."""
        removed = _LEGACY_OPTIONS | {
            "backend",
            "subplots",
            "layout",
            "secondary_y",
            "by",
            "data",
            "figsize",
            "return_type",
            "fmt",
            "zlabel",
        }
        if kind == "box":
            removed |= {"vert", "labels", "tick_labels"}
        unsupported = removed.intersection(kwargs)
        if unsupported:
            raise ValueError(
                f"Unsupported statistical options {sorted(unsupported)}; use named methods, ax= and contexts."
            )
        return {key: kwargs.pop(key) for key in ("xlabel", "ylabel", "title") if key in kwargs}

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

    def _draw(
        self,
        kind,
        traces,
        *,
        xcolumn,
        ax=None,
        fmt=None,
        label=_AUTO,
        xlabel=_AUTO,
        ylabel=_AUTO,
        title=_AUTO,
        xscale=None,
        yscale=None,
        legend=True,
        **kwargs,
    ):
        """Draw prepared ordinary traces without templates or pyplot render calls."""
        removed = _LEGACY_OPTIONS.intersection(kwargs)
        if removed:
            raise ValueError(
                f"Legacy plotting options {sorted(removed)} are unsupported; use ax= and the named methods."
            )
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
            ax.errorbar(trace.x, trace.y, xerr=trace.xerr, yerr=trace.yerr, fmt="" if fmt is None else fmt, **options)
        else:
            draw_errorfill(trace.x, trace.y, xerr=trace.xerr, yerr=trace.yerr, ax=ax, fmt=fmt, **options)

    def _annotate(self, ax, traces, xcolumn, xlabel, ylabel, title):
        """Fill missing source labels while retaining explicit caller annotations."""
        if xlabel is not _AUTO or not ax.get_xlabel():
            ax.set_xlabel(self._text(self._label(xcolumn) if xlabel is _AUTO else xlabel or ""))
        if ylabel is not _AUTO or not ax.get_ylabel():
            ax.set_ylabel(
                self._text(", ".join(self._label(t.column) for t in traces) if ylabel is _AUTO else ylabel or "")
            )
        if title is not _AUTO or not ax.get_title():
            ax.set_title(self._text(self._data.basename if title is _AUTO else title or ""))

    @staticmethod
    def _field_options(kwargs):
        """Separate annotations and reject legacy controls before creating axes."""
        removed = (
            _LEGACY_OPTIONS | {"griddata", "projection", "backend", "mode", "scale_factor", "colormap"}
        ).intersection(kwargs)
        if removed:
            raise ValueError(f"Unsupported field options {sorted(removed)}; select a named method and use ax=.")
        annotations = {key: kwargs.pop(key) for key in ("xlabel", "ylabel", "zlabel", "title") if key in kwargs}
        return annotations

    def _field(self, kind, columns, **kwargs):
        """Draw through the shared Matplotlib field preparation/rendering path."""
        annotations = self._field_options(kwargs)
        return draw_field(self, kind, columns, annotations=annotations, **kwargs)

    def _matrix(self, x, y, **kwargs):
        """Draw a matrix without passing its coordinate arrays to column resolution."""
        annotations = self._field_options(kwargs)
        return draw_matrix(self, x, y, annotations=annotations, **kwargs)

    def surface(self, x=None, y=None, z=None, **kwargs):
        """Draw a scalar Z(X,Y) field as a Matplotlib 3D surface.

        Args:
            x (column selector or None):
                X coordinates; omitted selectors use roles or explicit-method defaults.
            y (column selector or None):
                One Y coordinate column.
            z (column selector or None):
                Scalar response column. Unassigned tables use columns 0/1/2.

        Keyword Arguments:
            **kwargs (dict):
                ax selects a live 3D Axes. shape=(nx, ny) controls inclusive grid
                centres; defaults count distinct coordinates. xlim/ylim specify
                grid bounds. method is linear (default), nearest or cubic.
                colorbar=True adds a scalar colour scale. xlabel, ylabel, zlabel
                and title override annotations; None clears one. Other keywords
                go to Axes3D.plot_surface, including cmap, norm, vmin/vmax and alpha.

        Returns:
            matplotlib.axes.Axes:
                Primary 3D Axes. Measurements, roles, masks and metadata are unchanged.

        Notes:
            Required columns are masked jointly. Excluded measured grid sites and
            points outside the linear/cubic interpolation hull remain missing.
            Regular lattices retain their original values and holes; scattered
            samples use scipy.interpolate.griddata. Duplicate sites use the first
            valid sample. Drawing never chooses Mayavi, shows or saves a figure.
        """
        return self(x, y, z=z, kind="surface", **kwargs)

    def contour(self, x=None, y=None, z=None, **kwargs):
        """Draw scalar-field contour lines, or filled contours with filled=True.

        Args:
            x (column selector or None):
                X coordinate column, as for surface.
            y (column selector or None):
                Y coordinate column.
            z (column selector or None):
                Scalar response column.

        Keyword Arguments:
            **kwargs (dict):
                Grid, exclusions, annotations and colorbar controls work as for
                surface. ax must be 2D; filled=False selects Axes.contour and True
                selects Axes.contourf. Other keywords include levels and cmap.

        Returns:
            matplotlib.axes.Axes:
                Primary 2D Axes, without modifying the measurements.
        """
        return self(x, y, z=z, kind="contour", **kwargs)

    def pcolormesh(self, x=None, y=None, z=None, **kwargs):
        """Draw a gridded scalar field with Axes.pcolormesh.

        Args:
            x (column selector or None):
                X coordinate column.
            y (column selector or None):
                Y coordinate column.
            z (column selector or None):
                Scalar response column.

        Keyword Arguments:
            **kwargs (dict):
                Grid, mask, annotation and colorbar controls work as for surface.
                ax selects a 2D Axes; other keywords go to Axes.pcolormesh.

        Returns:
            matplotlib.axes.Axes:
                Primary 2D Axes, with missing grid cells masked.
        """
        return self(x, y, z=z, kind="pcolormesh", **kwargs)

    def image(self, x=None, y=None, z=None, **kwargs):
        """Draw a scalar field as an image with its physical coordinate extent.

        Args:
            x (column selector or None):
                Horizontal pixel-centre coordinates.
            y (column selector or None):
                Vertical pixel-centre coordinates.
            z (column selector or None):
                Scalar response column.

        Keyword Arguments:
            **kwargs (dict):
                Grid, mask, annotation and colorbar controls work as for surface.
                Other keywords go to Axes.imshow. Defaults use origin='lower' and
                aspect='auto'; bounds extend half a cell beyond the outer centres.

        Returns:
            matplotlib.axes.Axes:
                Primary 2D Axes. The image retains scalar values for normalisation
                and colour bars, rather than pre-converting them to RGB.
        """
        return self(x, y, z=z, kind="image", **kwargs)

    def scatter3d(self, x=None, y=None, z=None, **kwargs):
        """Draw ungridded XYZ samples with Axes3D.scatter.

        Args:
            x (column selector or None):
                X coordinate column.
            y (column selector or None):
                Y coordinate column.
            z (column selector or None):
                Z coordinate column.

        Keyword Arguments:
            **kwargs (dict):
                ax selects a 3D Axes. Annotation controls work as for surface;
                other keywords go to Axes3D.scatter. Per-row c/s/colour arrays are
                filtered with coordinate exclusions. No interpolation is applied.

        Returns:
            matplotlib.axes.Axes:
                Primary 3D Axes, without changing measurements.
        """
        return self(x, y, z=z, kind="scatter3d", **kwargs)

    def vector(self, x=None, y=None, u=None, v=None, w=None, **kwargs):
        """Draw a 2D vector-colour image with an optional arrow overlay.

        Args:
            x (column selector or None):
                X coordinate column.
            y (column selector or None):
                Y coordinate column.
            u (column selector or None):
                X vector component.
            v (column selector or None):
                Y vector component.
            w (column selector, False or None):
                Optional third component for colour mapping; False disables it.

        Keyword Arguments:
            **kwargs (dict):
                ax, shape, method and xlim/ylim work as for image. arrows=True
                overlays U/V arrows; quiver_kwargs supplies separate arrow styling
                as for quiver. Remaining keywords go to Axes.imshow (alpha=0.75
                by default), including interpolation="bilinear" or "bicubic"
                for display smoothing. Directional RGB colours have no scalar colour bar.

        Returns:
            matplotlib.axes.Axes:
                Primary 2D Axes, without changing measurements or column roles.

        Notes:
            Hue uses atan2(U,V); saturation is vector magnitude relative to the
            maximum, and W controls luminance across its range. Constant W uses
            neutral luminance; a zero field is finite neutral grey. Required
            coordinate/component exclusions apply to both image and arrows.
            shape and method interpolate RGB colours rather than vector components;
            arrows retain their original samples. Neither grid nor display
            interpolation applies a statistical smoothing filter to measurements.
        """
        return self(x, y, u=u, v=v, w=w, kind="vector", **kwargs)

    def quiver(self, x=None, y=None, u=None, v=None, **kwargs):
        """Draw ungridded 2D U/V arrows in data coordinates.

        Args:
            x (column selector or None):
                X coordinate column.
            y (column selector or None):
                Y coordinate column.
            u (column selector or None):
                X vector component.
            v (column selector or None):
                Y vector component.

        Keyword Arguments:
            **kwargs (dict):
                ax selects a 2D Axes. Other keywords go to Axes.quiver, with
                pivot='mid', angles='xy', scale_units='xy' and scale=1 defaults.
                Annotation controls work as for surface. Unused Z/W roles are ignored.

        Returns:
            matplotlib.axes.Axes:
                Primary 2D Axes, with joint X/Y/U/V exclusions applied.
        """
        return self(x, y, u=u, v=v, kind="quiver", **kwargs)

    def vector3d(self, x=None, y=None, z=None, u=None, v=None, w=None, **kwargs):
        """Draw XYZ positions and U/V/W vectors with Matplotlib 3D quiver.

        Args:
            x (column selector or None):
                X coordinate column.
            y (column selector or None):
                Y coordinate column.
            z (column selector or None):
                Z coordinate column.
            u (column selector or None):
                X vector component.
            v (column selector or None):
                Y vector component.
            w (column selector or None):
                Z vector component.

        Keyword Arguments:
            **kwargs (dict):
                ax selects a 3D Axes. Annotation controls work as for surface;
                remaining keywords go to Axes3D.quiver. Defaults colour arrows
                with the vector method's directional mapping; color/colors can
                override it. All six selected columns contribute to validity.

        Returns:
            matplotlib.axes.Axes:
                Primary 3D Axes. Installed optional packages do not change backend.
        """
        return self(x, y, z=z, u=u, v=v, w=w, kind="vector3d", **kwargs)

    def voxels(self, x=None, y=None, z=None, u=None, **kwargs):
        """Draw scalar U(X,Y,Z) values as physical voxels.

        Args:
            x (column selector or None):
                X cell-centre coordinate column.
            y (column selector or None):
                Y cell-centre coordinate column.
            z (column selector or None):
                Z cell-centre coordinate column.
            u (column selector or None):
                Scalar intensity column.

        Keyword Arguments:
            **kwargs (dict):
                shape=(nx, ny, nz), grid bounds xlim/ylim/zlim and method control
                gridding. visible is a callable of grid-centre X/Y/Z arrays;
                filled is an alternative Boolean grid. Both combine with validity.
                face_alpha=0.5 and edge_alpha=0.9 control mapped opacity.
                cmap, norm and vmin/vmax control the same scalar map as the colour
                bar (colorbar=True). Remaining keywords go to Axes3D.voxels.

        Returns:
            matplotlib.axes.Axes:
                Primary 3D Axes; absent/excluded samples are never filled as zeros.
        """
        return self(x, y, z=z, u=u, kind="voxels", **kwargs)

    def matrix(self, x=None, y=None, **kwargs):
        """Draw a surface from a table whose rows index X and columns index Y.

        Args:
            x (column selector, array or None):
                X coordinates. None uses the first column and excludes it from
                the matrix. A vector supplies explicit coordinates instead.
            y (row index, array or None):
                Y coordinates. None parses selected column headers as floats.
                A row index reads its values and starts the default block below it.

        Keyword Arguments:
            **kwargs (dict):
                rect=(row, column[, rows, columns]) selects a block. Explicit
                coordinate arrays can span the whole table or only that block.
                ax selects a 3D Axes. colorbar=True, annotation controls and other
                Axes3D.plot_surface keywords work as for surface. For an (nx, ny)
                matrix, rcount=ny and ccount=nx render all existing samples;
                linewidth=0 and antialiased=True improve polygon-edge rendering.

        Returns:
            matplotlib.axes.Axes:
                Primary 3D Axes. The matrix's masks and values are preserved;
                non-square tables are transposed only for Matplotlib's Y/X grid.

        Notes:
            This method does not interpolate new matrix values. Image interpolation
            and field-gridding shape/method options are unsupported; use surface
            on a row-wise XYZ table when numerical interpolation is required.
        """
        return self(x, y, kind="matrix", **kwargs)

    def waterfall(self, x=None, y=None, z=None, **kwargs):
        """Draw grouped Z(X,Y) traces with the existing Joy Division waterfall helper.

        Args:
            x (column selector or None):
                Horizontal coordinate column.
            y (column selector or None):
                Column grouping each trace.
            z (column selector or None):
                Vertical response, offset between successive Y groups.

        Keyword Arguments:
            **kwargs (dict):
                ax selects a 2D Axes. y_shift, bg_color, color, axes_color,
                linewidth and legend_fmt go to the Stoner joy_division utility.
                Annotation overrides work as for surface. No gridding is applied.

        Returns:
            matplotlib.axes.Axes:
                Primary 2D Axes containing only finite, unexcluded samples.
        """
        return self(x, y, z=z, kind="waterfall", **kwargs)
