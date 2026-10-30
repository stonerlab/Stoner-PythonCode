"""Resolve image coordinates without changing scientific values or calibration."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..enums import CoordinateMode
from .storage_bridge import owner_of


@dataclass(frozen=True)
class ImageGrid:
    """Describe detached separable pixel centres, edges and axis labels."""

    x: np.ndarray
    y: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    xlabel: str
    ylabel: str
    uniform: bool


def _axis(dataset, name, dimension):
    """Validate a coordinate vector and calculate pixel edges in source order."""
    if not isinstance(name, str) or name not in dataset.coords:
        raise ValueError(f"{dimension} must name an existing spatial coordinate.")
    coord = dataset.coords[name]
    if coord.dims != (dimension,):
        raise ValueError(f"Coordinate {name!r} must be a one-dimensional {dimension} vector; maps are not supported.")
    raw = coord.values
    if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError(f"Coordinate {name!r} must contain finite real values.")
    values = raw.astype(float, copy=True)
    if raw.dtype.kind in "iu" and not np.array_equal(raw.astype(object), values.astype(object)):
        raise ValueError(f"Coordinate {name!r} loses precision when converted for rendering.")
    edges, uniform = coordinate_edges(values)
    label = str(coord.attrs.get("long_name", name))
    units = coord.attrs.get("units")
    if units:
        label += f" / {units}"
    return values, edges, label, uniform


def coordinate_edges(values):
    """Return midpoint edges for strictly monotonic centres, or unit width for one pixel."""
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5]), True
    steps = np.diff(values)
    if not (np.all(steps > 0) or np.all(steps < 0)):
        raise ValueError("Spatial coordinates must be strictly monotonic; select pixel mode for unordered axes.")
    edges = np.r_[values[0] - steps[0]/2, values[:-1] + steps/2, values[-1] + steps[-1]/2]
    if not np.isfinite(edges).all():
        raise ValueError("Spatial coordinate edges must remain finite.")
    return edges, bool(np.allclose(steps, steps[0], rtol=1e-7, atol=abs(steps[0])*1e-10))


def resolve_grid(image, x=None, y=None, coordinates=CoordinateMode.NATIVE):
    """Resolve separable coordinates for an image handle, retaining crop offsets.

    Args:
        image (ImageFile):
            Image whose live owner supplies the spatial grid.
        x, y (str or None):
            Existing coordinate names; None selects the corresponding dimension.
        coordinates (CoordinateMode or str):
            Native vectors by default, or local pixel indices. Pixel mode rejects
            explicit coordinate names. Wrong-domain enums are rejected.

    Returns:
        ImageGrid:
            Independent centre/edge vectors and labels. A singleton has unit width.

    Raises:
        TypeError:
            If coordinates is neither a supported string nor CoordinateMode.
        ValueError:
            If coordinates are incompatible, non-finite, non-monotonic or a map.
    """
    if isinstance(coordinates, Enum) and not isinstance(coordinates, CoordinateMode):
        raise TypeError("coordinates must be CoordinateMode or 'native'/'pixel'.")
    if not isinstance(coordinates, (str, CoordinateMode)):
        raise TypeError("coordinates must be CoordinateMode or 'native'/'pixel'.")
    coordinates = CoordinateMode(coordinates)
    owner = owner_of(image)
    if owner is None:
        raise ValueError("Image plotting requires a populated ImageFile owner.")
    dataset = owner._state.dataset
    if coordinates is CoordinateMode.PIXEL:
        if x is not None or y is not None:
            raise ValueError("Pixel mode uses local indices; omit x and y coordinate names.")
        xv, yv = (np.arange(dataset.sizes[axis], dtype=float) for axis in ("x", "y"))
        xe, _ = coordinate_edges(xv)
        ye, _ = coordinate_edges(yv)
        return ImageGrid(xv, yv, xe, ye, "x / pixel", "y / pixel", True)
    xv, xe, xl, xu = _axis(dataset, "x" if x is None else x, "x")
    yv, ye, yl, yu = _axis(dataset, "y" if y is None else y, "y")
    return ImageGrid(xv, yv, xe, ye, xl, yl, xu and yu)
