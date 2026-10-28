"""Prepare independent statistical samples and draw native Matplotlib plots."""

from dataclasses import dataclass
from inspect import signature
from numbers import Integral
from re import Pattern

import numpy as np
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

from .selection import prepare_traces

__all__ = []

_BOX_PARAMETERS = signature(Axes.boxplot).parameters
_AUTO = object()


def _numeric_column(data, position):
    """Read one real numeric column without materialising unrelated measurements."""
    values = np.ma.asarray(data.column(position))
    if values.dtype.kind not in "biuf":
        raise TypeError("Statistical plotting requires real numerical columns.")
    return values


def _row_values(data, specification, name):
    """Resolve optional weights/reduced values as one column or a source-row array."""
    if specification is None:
        return None
    if isinstance(specification, bool):
        raise TypeError(f"{name} must be one column selector or an array with one value per source row.")
    if isinstance(specification, (Integral, str, Pattern)):
        positions = data.find_col(specification, force_list=True)
        if len(positions) != 1:
            raise ValueError(f"{name} must select exactly one column; {specification!r} matched {positions!r}.")
        values = _numeric_column(data, positions[0])
    else:
        values = np.ma.asarray(specification)
    if values.shape != (len(data),):
        raise ValueError(f"{name} must have shape ({len(data)},), matching source rows.")
    if values.dtype.kind not in "biuf":
        raise TypeError(f"{name} must contain real numerical values.")
    return values


@dataclass(frozen=True)
class PlotSeries:
    """Hold one independently filtered series and matching optional weights."""

    column: int
    values: np.ndarray
    weights: object


def prepare_series(data, y=None, *, startx=0, no_guess=False, weights=None):
    """Select Y roles or real numeric columns and filter each distribution separately."""
    if isinstance(startx, bool) or not isinstance(startx, Integral) or not 0 <= startx < max(data.shape[1], 1):
        raise ValueError("startx must be a column position within the table.")
    if y is None:
        try:
            positions = data.setas._get_cols(startx=int(startx), no_guess=True)["ycol"]
        except ValueError as error:
            raise ValueError("No assigned X group exists at or after startx.") from error
        if not positions and not no_guess:
            positions = [index for index, dtype in enumerate(data.data.dtypes) if dtype.kind in "biuf"]
    else:
        if isinstance(y, bool):
            raise TypeError("y must select statistical sample columns.")
        positions = data.find_col(y, force_list=True)
    if not positions:
        raise ValueError("Select samples with y=, assign Y roles, or allow numeric-column guessing.")
    row_weights = _row_values(data, weights, "weights")
    series = []
    for position in positions:
        values = _numeric_column(data, position)
        valid = ~np.ma.getmaskarray(values) & np.isfinite(values.data)
        if row_weights is not None:
            valid &= ~np.ma.getmaskarray(row_weights) & np.isfinite(row_weights.data)
        if not valid.any():
            raise ValueError(f"No finite, unexcluded samples remain in column {position}.")
        series.append(
            PlotSeries(
                position, values.data[valid].copy(), None if row_weights is None else row_weights.data[valid].copy()
            )
        )
    return tuple(series)


def _density_curves(series, bw_method, ind):
    """Validate all KDEs and evaluation coordinates before selecting a target."""
    if ind is not None and (isinstance(ind, bool) or isinstance(ind, Integral) and ind < 2):
        raise ValueError("ind must be an integer of at least two points or a finite evaluation vector.")
    if ind is not None and not isinstance(ind, Integral):
        ind = np.asarray(ind)
        if ind.ndim != 1 or ind.size < 2 or ind.dtype.kind not in "biuf" or not np.isfinite(ind).all():
            raise ValueError("ind must be an integer of at least two points or a finite evaluation vector.")
    curves = []
    for sample in series:
        values, weights = sample.values.astype(float), sample.weights
        if weights is not None:
            if (weights < 0).any() or not (weights > 0).any():
                raise ValueError("KDE weights must be non-negative with positive total weight.")
            values, weights = values[weights > 0], weights[weights > 0]
        if len(values) < 2 or np.ptp(values) == 0:
            raise ValueError(f"KDE requires at least two distinct samples in column {sample.column}.")
        try:
            estimator = gaussian_kde(values, bw_method=bw_method, weights=weights)
            if not np.isfinite(estimator.factor) or estimator.factor <= 0:
                raise ValueError("KDE bandwidth must be finite and positive.")
            if ind is None or isinstance(ind, Integral):
                span = np.ptp(values)
                points = np.linspace(
                    values.min() - span / 2, values.max() + span / 2, 1000 if ind is None else int(ind)
                )
            else:
                points = ind.copy()
            density = estimator(points)
        except (np.linalg.LinAlgError, ValueError) as error:
            raise ValueError(f"Cannot estimate KDE for column {sample.column}: {error}") from error
        if not np.isfinite(density).all():
            raise ValueError("KDE produced non-finite density values; check samples and bandwidth.")
        curves.append((points, density))
    return curves


def _finish(plot, ax, defaults, annotations, xscale=None, yscale=None):
    """Fill missing annotations while preserving caller text and tick formatters."""
    for name, default in zip(("xlabel", "ylabel", "title"), (*defaults, plot._data.basename)):
        if name in annotations or not getattr(ax, f"get_{name}")():
            getattr(ax, f"set_{name}")(plot._text(annotations.get(name, default) or ""))
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    plot._record(ax.figure, ax)
    return ax


def draw_distribution(
    plot,
    kind,
    y=None,
    *,
    startx=0,
    no_guess=False,
    ax=None,
    annotations=None,
    weights=None,
    label=_AUTO,
    legend=None,
    xscale=None,
    yscale=None,
    **kwargs,
):
    """Draw independently selected distributions without a pandas plotting intermediate."""
    series = prepare_series(plot._data, y, startx=startx, no_guess=no_guess, weights=weights)
    count = len(series)
    labels = (
        [plot._label(sample.column) for sample in series] if label is _AUTO else plot._per_trace(label, count, "label")
    )
    if any(value is not None and not isinstance(value, str) for value in labels):
        raise TypeError("label must contain strings or None.")
    labels = [plot._text(value) for value in labels]
    options = {key: plot._text(value) for key, value in kwargs.items()}
    orientation = options.pop("orientation", "vertical")
    if orientation not in ("vertical", "horizontal") or kind == "kde" and orientation != "vertical":
        raise ValueError("Use vertical/horizontal orientation; KDE uses a horizontal value axis.")
    if kind == "box" and weights is not None:
        raise ValueError("box does not support weights.")
    if kind == "box":
        options["orientation" if "orientation" in _BOX_PARAMETERS else "vert"] = (
            orientation if "orientation" in _BOX_PARAMETERS else orientation == "vertical"
        )
        options["tick_labels" if "tick_labels" in _BOX_PARAMETERS else "labels"] = [value or "" for value in labels]
    if kind == "kde":
        curves = _density_curves(series, options.pop("bw_method", None), options.pop("ind", None))
    ax = plot._target(ax)
    names = ", ".join(plot._label(sample.column) for sample in series)
    if kind == "hist":
        samples = [sample.values for sample in series]
        row_weights = None if weights is None else [sample.weights for sample in series]
        ax.hist(
            samples[0] if count == 1 else samples,
            weights=None if row_weights is None else row_weights[0] if count == 1 else row_weights,
            label=labels[0] if count == 1 else labels,
            orientation=orientation,
            **options,
        )
        frequency = "Density" if options.get("density", False) else "Count" if weights is None else "Weighted count"
        if options.get("cumulative", False):
            frequency = (
                "Cumulative probability" if options.get("density", False) else "Cumulative " + frequency.lower()
            )
        defaults = (names, frequency)
    elif kind == "box":
        ax.boxplot([sample.values for sample in series], **options)
        defaults = ("", names)
    else:
        for (points, density), name in zip(curves, labels):
            ax.plot(points, density, label=name, **options)
        defaults = (names, "Density")
    if orientation == "horizontal":
        defaults = defaults[::-1]
    if (kind != "box" if legend is None else legend) and ax.get_legend_handles_labels()[0]:
        ax.legend()
    return _finish(plot, ax, defaults, annotations or {}, xscale, yscale)


def draw_hexbin(plot, columns, *, C=None, ax=None, annotations=None, colorbar=True, **kwargs):
    """Draw counts or reduced C values with joint X/Y/C exclusions."""
    if len(columns.y) != 1:
        raise ValueError("hexbin requires exactly one X and one Y column.")
    trace = prepare_traces(plot._data, columns)[0]
    values = _row_values(plot._data, C, "C")
    x, y = trace.x, trace.y
    if values is not None:
        values = values[trace.rows]
        valid = ~np.ma.getmaskarray(values) & np.isfinite(values.data)
        x, y, values = x[valid], y[valid], values.data[valid].copy()
        if not x.size:
            raise ValueError("No finite, unexcluded joint X/Y/C samples remain.")
    options = {key: plot._text(value) for key, value in kwargs.items()}
    ax = plot._target(ax)
    artist = ax.hexbin(x, y, C=values, **options)
    if colorbar:
        ax.figure.colorbar(artist, ax=ax)
    return _finish(plot, ax, (plot._label(columns.x), plot._label(columns.y[0])), annotations or {})
