"""Prepare independent curve-fit inputs from explicit operation selections."""

from copy import deepcopy

import numpy as np

from ...core.selection import resolve_selection
from ...core.column_selection import _single_column


def _observations(datafile, specification, raw, *, uncertainty=False):
    """Resolve a column or complete external observation vector with its exclusions."""
    if isinstance(specification, np.ndarray):
        values = np.ma.asarray(specification)
        if values.shape != (len(datafile),):
            raise ValueError("External fitting arrays must match the complete row count")
        return values
    if uncertainty and isinstance(specification, (float, np.floating)):
        return np.ma.array(np.full(len(datafile), specification))
    return raw[:, _single_column(datafile, specification, "observation")]


def curve_fit_inputs(datafile, xcol, ycol, sigma, where, bounds):
    """Resolve per-channel inputs and full-axis model/residual exclusions."""
    raw = datafile.to_numpy()
    columns = datafile._col_args(scalar=False, xcol=xcol, ycol=None if isinstance(ycol, np.ndarray) else ycol)
    xcols = np.atleast_1d(columns.xcol).tolist()
    x = raw[:, xcols]
    x_values = x.data[:, 0] if len(xcols) == 1 else x.data
    model_mask = np.any(np.ma.getmaskarray(x) | ~np.isfinite(x.data), axis=1)
    selected = resolve_selection(where, datafile._storage_owner._state.values)
    if bounds is not None:
        selected &= np.asarray(datafile.search_index(xcols[0], bounds), dtype=bool)
    ycols = [ycol] if isinstance(ycol, np.ndarray) else columns.ycol
    if not ycols:
        raise ValueError("No y channels were specified for fitting")
    errors = sigma if sigma is not None else columns.yerr
    if errors is not None and not isinstance(errors, (list, np.ndarray, float, np.floating)):
        errors = datafile.find_col(errors, force_list=True)
    if isinstance(errors, list):
        if len(errors) == 0:
            errors = [None] * len(ycols)
        elif len(errors) == 1:
            errors = errors * len(ycols)
        elif len(errors) != len(ycols):
            raise ValueError("Uncertainty columns must match the fitted y channels")
    else:
        errors = [errors] * len(ycols)
    inputs = []
    for yc, error in zip(ycols, errors):
        y = _observations(datafile, yc, raw)
        residual_mask = model_mask | np.ma.getmaskarray(y) | ~np.isfinite(y.data)
        included = selected & ~residual_mask
        uncertainty = None
        if error is not None:
            uncertainty = _observations(datafile, error, raw, uncertainty=True)
            included &= ~np.ma.getmaskarray(uncertainty) & np.isfinite(uncertainty.data)
            if np.any(uncertainty.data[included] <= 0):
                raise ValueError("Eligible fitting uncertainties must be strictly positive")
            uncertainty = uncertainty.data[included]
        channel_columns = deepcopy(columns)
        channel_columns.xcol = xcols[0] if len(xcols) == 1 else xcols
        channel_columns.ycol = yc if isinstance(yc, np.ndarray) else [yc]
        inputs.append(
            (
                x_values[included],
                y.data[included],
                uncertainty,
                channel_columns,
                x_values,
                y.data,
                model_mask,
                residual_mask,
            )
        )
    return inputs
