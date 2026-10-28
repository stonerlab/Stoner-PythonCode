"""Resolve scientific plotting columns and prepare independent masked traces."""

from dataclasses import dataclass
from numbers import Integral, Real
from re import Pattern

import numpy as np


@dataclass(frozen=True)
class PlotColumns:
    """Describe resolved column positions without retaining measurement arrays."""

    x: int
    y: tuple
    z: tuple
    u: tuple
    v: tuple
    w: tuple
    xerr: object
    yerr: tuple

    @property
    def kind(self):
        """Choose a scientific plot from the resolved coordinates and components."""
        if self.u and self.v:
            return "vector3d" if self.z and self.w else "vector"
        if self.z:
            return "voxels" if self.u else "surface"
        if self.u or self.v or self.w:
            raise ValueError("Incomplete vector columns: specify u and v, or select kind='line'.")
        return "xy"


def _positions(data, explicit, default, name):
    """Resolve one coordinate selection using the shared Stoner column resolver."""
    selection = default if explicit is None else explicit
    if selection is False or selection is None:
        return ()
    positions = tuple(data.find_col(selection, force_list=True))
    if name != "y" and len(positions) > 1:
        raise ValueError(f"{name} must select exactly one column; {selection!r} matched {positions!r}.")
    return positions


def resolve_columns(data, *, x=None, y=None, z=None, u=None, v=None, w=None, startx=0, no_guess=False):
    """Resolve explicit selectors over role assignments and the existing count defaults.

    Args:
        data (Data):
            Measurement table whose current schema is resolved.

    Keyword Arguments:
        x, y, z, u, v, w (column selector or None):
            Explicit selectors; None uses the selected role group. False disables
            a coordinate. Only y can select several columns.
        startx (int):
            Position at which to look for the next assigned X group.
        no_guess (bool):
            Disable column-count guesses when no roles are assigned.

    Returns:
        PlotColumns:
            Positions and matching assigned uncertainty columns, without reading values.

    Raises:
        ValueError:
            If X or Y is unavailable, or a scalar coordinate selects several columns.
    """
    if isinstance(startx, bool) or not isinstance(startx, Integral) or not 0 <= startx < max(data.shape[1], 1):
        raise ValueError("startx must be a column position within the table.")
    try:
        defaults = data.setas._get_cols(startx=int(startx), no_guess=no_guess)
    except ValueError as err:
        raise ValueError("No assigned X group exists at or after startx.") from err
    positions = {
        name: _positions(data, value, defaults.get(f"{name}col"), name)
        for name, value in {"x": x, "y": y, "z": z, "u": u, "v": v, "w": w}.items()
    }
    if not positions["x"] or not positions["y"]:
        raise ValueError("Cannot infer a plot: assign X/Y roles or supply x= and y= column selectors.")
    default_y = defaults.get("ycol") or []
    default_errors = defaults.get("yerr") or []
    errors = dict(zip(default_y, default_errors))
    xerr = defaults.get("xerr")
    return PlotColumns(
        x=positions["x"][0],
        y=positions["y"],
        z=positions["z"],
        u=positions["u"],
        v=positions["v"],
        w=positions["w"],
        xerr=xerr,
        yerr=tuple(errors.get(col) for col in positions["y"]),
    )


def _uncertainty(data, specification):
    """Read a column, scalar or symmetric/asymmetric array of uncertainties."""
    if specification is None or specification is False:
        return None
    if isinstance(specification, bool):
        raise TypeError("An uncertainty must be a column, scalar, array or False.")
    if isinstance(specification, (Integral, str, Pattern)):
        position = data.find_col(specification, force_list=True)
        if len(position) != 1:
            raise ValueError(f"Each uncertainty must select exactly one column; {specification!r} matched {position!r}.")
        values = data.column(position[0])
    elif isinstance(specification, Real):
        values = np.full(len(data), specification)
    else:
        values = specification
    result = np.ma.asarray(values, dtype=float)
    if result.shape not in ((len(data),), (2, len(data))):
        raise ValueError(f"Uncertainty shape must be ({len(data)},) or (2, {len(data)}), got {result.shape}.")
    return result


def uncertainty_specs(columns, xerr, yerr):
    """Match uncertainty arguments to selected Y columns, preserving role pairing."""
    xspec = columns.xerr if xerr is None else xerr
    if yerr is None:
        yspecs = columns.yerr
    elif isinstance(yerr, (list, tuple)) and len(yerr) == len(columns.y):
        yspecs = tuple(yerr)
    else:
        yspecs = (yerr,) * len(columns.y)
    return xspec, yspecs


@dataclass(frozen=True)
class PlotTrace:
    """Hold one filtered XY trace and its aligned uncertainties and source rows."""

    column: int
    x: np.ndarray
    y: np.ndarray
    xerr: object
    yerr: object
    rows: np.ndarray


def prepare_traces(data, columns, *, xerr=False, yerr=False):
    """Validate all selected traces before drawing, applying exclusions per series.

    Args:
        data (Data):
            Table to read without modifying its numerical or metadata state.
        columns (PlotColumns):
            Resolved coordinate and uncertainty roles.

    Keyword Arguments:
        xerr, yerr (column selector, float, array, sequence or None):
            None infers assigned errors; False disables them. Integer/string/pattern
            values select columns, floating scalars are constant errors, and arrays
            contain values. A Y sequence matching the trace count supplies errors
            per trace. Arrays can have shape (N,) or (2, N).

    Returns:
        tuple of PlotTrace:
            Independent, finite, unexcluded traces. Empty individual traces remain
            present to preserve per-trace style indexing.

    Raises:
        ValueError:
            If errors have invalid shapes/negative visible values or no trace has samples.
        TypeError:
            If selected coordinate columns are not real numerical values.
    """
    xvalues = np.ma.asarray(data.column(columns.x))
    xspec, yspecs = uncertainty_specs(columns, xerr, yerr)
    xerror = _uncertainty(data, xspec)
    traces = []
    for column, yspec in zip(columns.y, yspecs):
        yvalues = np.ma.asarray(data.column(column))
        if xvalues.dtype.kind not in "biuf" or yvalues.dtype.kind not in "biuf":
            raise TypeError("XY plotting requires real numerical coordinate columns.")
        valid = ~(np.ma.getmaskarray(xvalues) | np.ma.getmaskarray(yvalues))
        valid &= np.isfinite(xvalues.data) & np.isfinite(yvalues.data)
        yerror = _uncertainty(data, yspec)
        for error in (xerror, yerror):
            if error is not None:
                invalid = np.ma.getmaskarray(error) | ~np.isfinite(error.data)
                valid &= ~invalid if invalid.ndim == 1 else ~invalid.any(axis=0)
        for error in (xerror, yerror):
            if error is not None and np.any(error.data[..., valid] < 0):
                raise ValueError("Uncertainties must be non-negative at plotted samples.")
        traces.append(
            PlotTrace(
                column,
                xvalues.data[valid].copy(),
                yvalues.data[valid].copy(),
                None if xerror is None else xerror.data[..., valid].copy(),
                None if yerror is None else yerror.data[..., valid].copy(),
                valid,
            )
        )
    if not any(trace.x.size for trace in traces):
        raise ValueError("No finite, unexcluded samples remain in the selected traces.")
    return tuple(traces)
