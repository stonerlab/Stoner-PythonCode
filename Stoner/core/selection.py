"""Resolve operation selections without changing stored values or exclusions."""

import numpy as np
import pandas as pd
import xarray as xr


def _reject_where(options, operation):
    """Reject unsupported operation selections before numerical work or mutation."""
    if "where" in options:
        raise TypeError(f"{operation} does not support where; use explicit bounds or a separate selected object")


def resolve_selection(where, target):
    """Return a Boolean selection matching a table's rows or an intensity array.

    Args:
        where (array-like, pandas.Series, xarray.DataArray or None):
            True includes an observation. None includes every observation.
        target (pandas.DataFrame or xarray.DataArray):
            Authoritative axes against which to check shape and labels.

    Returns:
        numpy.ndarray:
            Independent Boolean selection with no implicit alignment or broadcasting.

    Raises:
        ValueError:
            The selector has missing values, non-Boolean dtype or incompatible axes.
    """
    table = isinstance(target, pd.DataFrame)
    shape = (len(target),) if table else target.shape
    if where is None:
        return np.ones(shape, dtype=bool)
    if isinstance(where, pd.Series):
        if not table or not where.index.is_unique or not where.index.equals(target.index):
            raise ValueError("where must have exactly the table's unique row index in order")
        if not pd.api.types.is_bool_dtype(where.dtype) or where.isna().any():
            raise ValueError("where must contain Boolean values without missing entries")
        values = where.to_numpy(dtype=bool)
    elif isinstance(where, xr.DataArray):
        if table or where.dims != target.dims:
            raise ValueError("where must have exactly the intensity dimensions in order")
        for name in set(target.dims) | set(where.coords):
            if name not in where.coords or name not in target.coords:
                raise ValueError("where must match intensity coordinates")
            if not where.coords[name].equals(target.coords[name]):
                raise ValueError("where must match intensity coordinates")
        values = where.values
    else:
        if np.ma.isMaskedArray(where) and np.any(np.ma.getmaskarray(where)):
            raise ValueError("where cannot contain missing entries")
        values = np.asarray(where)
    if values.dtype.kind != "b" or values.shape != shape:
        raise ValueError(f"where must be Boolean with shape {shape}; broadcasting is not supported")
    return values.copy()


def selected_image_values(dataset, where=None):
    """Return temporary masked pixels, combining selection, exclusions and geometry."""
    intensity = dataset["intensity"]
    included = resolve_selection(where, intensity)
    included &= ~dataset["excluded"].values & np.isfinite(intensity.values)
    if "frame" in intensity.dims:
        height = dataset["valid_height"].values[:, None, None]
        width = dataset["valid_width"].values[:, None, None]
        included &= np.arange(intensity.sizes["y"])[None, :, None] < height
        included &= np.arange(intensity.sizes["x"])[None, None, :] < width
    return np.ma.array(intensity.values, mask=~included, copy=True)
