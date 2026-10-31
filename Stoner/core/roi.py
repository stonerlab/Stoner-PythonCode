"""Describe reusable spatial selections independently of image ownership and plotting."""

import json
from dataclasses import dataclass
from enum import Enum

import numpy as np
import xarray as xr
from skimage.measure import points_in_poly

from ..enums import CoordinateMode
from .selection import resolve_selection

__all__ = ["ROI"]


def geometry_signature(target):
    """Capture labelled geometry without retaining intensities or a mutable view."""
    target = _target(target)
    return json.dumps({"shape": target.shape, "coords": target.coords.to_dataset().to_dict()},
                      sort_keys=True, allow_nan=False)


def _target(reference):
    """Require an explicit two-dimensional labelled spatial reference."""
    if isinstance(reference, xr.Dataset):
        reference = reference["intensity"]
    if not isinstance(reference, xr.DataArray) or reference.dims != ("y", "x"):
        raise TypeError("reference must be an intensity DataArray with dimensions ('y', 'x').")
    return reference


def _mode(coordinates):
    """Validate the coordinate enum without accepting other string enums."""
    if isinstance(coordinates, Enum) and not isinstance(coordinates, CoordinateMode):
        raise TypeError("coordinates must be CoordinateMode or 'pixel'/'native'.")
    if not isinstance(coordinates, str):
        raise TypeError("coordinates must be CoordinateMode or 'pixel'/'native'.")
    return CoordinateMode(coordinates)


def _numbers(values, length):
    """Validate finite real geometric parameters, excluding Boolean values."""
    if any(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError("Boolean values are not geometric parameters.")
    raw = np.asarray(values)
    if raw.shape != (length,) or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError(f"Expected {length} finite real geometric parameters.")
    return tuple(float(value) for value in raw)


def _units(reference):
    """Capture explicit unit labels; no implicit physical-unit conversion occurs."""
    reference = _target(reference)
    result = []
    for axis in ("x", "y"):
        coord = reference.coords.get(axis)
        if coord is None or coord.dims != (axis,):
            raise ValueError("Native ROI geometry requires separable x and y coordinates.")
        units = coord.attrs.get("units", "")
        if not isinstance(units, str):
            raise TypeError("Coordinate units must be strings.")
        result.append(units)
    return tuple(result)


@dataclass(frozen=True, slots=True, init=False)
class ROI:
    """An immutable spatial inclusion description, independent of scientific exclusions.

    Use the named constructors rather than instantiating this class directly.
    Geometry defaults to local pixel coordinates: x is column and y is row.
    Native geometry requires a reference intensity DataArray to capture its units.
    Boxes are half-open; circle, ellipse and polygon boundaries are included.
    Union, intersection, difference and complement use ``|``, ``&``, ``-`` and ``~``.
    Descriptions contain no image owner, renderer or mutable numerical buffer.
    """

    _kind: str
    _values: tuple
    _coordinates: CoordinateMode | None
    _units: tuple | None

    def __init__(self):
        raise TypeError("Use ROI.box, ROI.from_mask, ROI.all or another named constructor.")

    @classmethod
    def _make(cls, kind, values=(), coordinates=None, units=None):
        result = object.__new__(cls)
        for name, value in (("_kind", kind), ("_values", tuple(values)),
                            ("_coordinates", coordinates), ("_units", units)):
            object.__setattr__(result, name, value)
        return result

    @classmethod
    def _geometry(cls, kind, values, coordinates, reference):
        mode = _mode(coordinates)
        if mode is CoordinateMode.NATIVE:
            if reference is None:
                raise ValueError("Native ROI construction requires reference=image.image.intensity.")
            units = _units(reference)
        else:
            units = None
        return cls._make(kind, values, mode, units)

    @classmethod
    def box(cls, xmin, xmax, ymin, ymax, *, coordinates=CoordinateMode.PIXEL, reference=None):
        """Build a lower-inclusive, upper-exclusive box without negative-index wrapping.

        Args:
            xmin, xmax, ymin, ymax (float):
                Finite bounds in x/y order. Equal bounds describe an empty region.

        Keyword Arguments:
            coordinates (CoordinateMode or str):
                Pixel coordinates by default; native selects physical axis values.
            reference (xarray.DataArray or None):
                Required for native geometry; captures x/y unit labels only.

        Returns:
            ROI:
                Reusable immutable region; construction changes no image.
        """
        values = _numbers((xmin, xmax, ymin, ymax), 4)
        if values[0] > values[1] or values[2] > values[3]:
            raise ValueError("Box lower bounds must not exceed upper bounds.")
        return cls._geometry("box", values, coordinates, reference)

    @classmethod
    def circle(cls, centre, radius, *, coordinates=CoordinateMode.PIXEL, reference=None):
        """Build a closed circle with finite x/y centre and strictly positive radius.

        Args:
            centre (tuple):
                Centre in (x, y) order.
            radius (float):
                Positive radius in the selected coordinate units.

        Keyword Arguments:
            coordinates (CoordinateMode or str):
                Pixel by default; native requires matching x/y unit labels.
            reference (xarray.DataArray or None):
                Native coordinate reference, as for :meth:`box`.

        Returns:
            ROI:
                Immutable circular region including its boundary.
        """
        return cls.ellipse(centre, (radius, radius), coordinates=coordinates, reference=reference)

    @classmethod
    def ellipse(cls, centre, radii, *, coordinates=CoordinateMode.PIXEL, reference=None):
        """Build a closed axis-aligned ellipse with positive x/y radii.

        Args:
            centre (tuple):
                Centre in (x, y) order.
            radii (tuple):
                Positive radii in (x, y) order.

        Keyword Arguments:
            coordinates (CoordinateMode or str):
                Pixel by default; native axes must have equal unit labels.
            reference (xarray.DataArray or None):
                Native coordinate reference, as for :meth:`box`.

        Returns:
            ROI:
                Immutable ellipse including its boundary.
        """
        values = _numbers(centre, 2) + _numbers(radii, 2)
        if min(values[2:]) <= 0:
            raise ValueError("Ellipse radii must be strictly positive.")
        result = cls._geometry("ellipse", values, coordinates, reference)
        if result._units is not None and result._units[0] != result._units[1]:
            raise ValueError("Native circle/ellipse axes must use the same units.")
        return result

    @classmethod
    def polygon(cls, vertices, *, coordinates=CoordinateMode.PIXEL, reference=None):
        """Build a closed polygon from at least three finite (x, y) vertices.

        Args:
            vertices (array-like):
                Ordered vertices with shape (n, 2). Edge pixels are included.

        Keyword Arguments:
            coordinates (CoordinateMode or str):
                Pixel by default, or native physical coordinates.
            reference (xarray.DataArray or None):
                Native coordinate reference, as for :meth:`box`.

        Returns:
            ROI:
                Immutable polygon evaluated at pixel centres without rounding.
        """
        vertices = np.asarray(vertices)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError("A polygon requires at least three (x, y) vertices.")
        values = tuple(_numbers(vertex, 2) for vertex in vertices)
        if np.linalg.matrix_rank(np.asarray(values) - values[0]) < 2:
            raise ValueError("Polygon vertices must not be collinear.")
        return cls._geometry("polygon", values, coordinates, reference)

    @classmethod
    def from_mask(cls, included, *, reference):
        """Capture an exact Boolean inclusion array bound to labelled geometry.

        Args:
            included (Boolean array or xarray.DataArray):
                True includes pixels; shape and labels must match the reference.

        Keyword Arguments:
            reference (xarray.DataArray):
                Intensity axes to capture. Later grid/label mismatches are errors.

        Returns:
            ROI:
                Independent immutable mask description; caller edits cannot alter it.
        """
        target = _target(reference)
        values = resolve_selection(included, target)
        return cls._make("mask", (geometry_signature(target), values.shape,
                                  np.packbits(values.ravel()).tobytes().hex()))

    @classmethod
    def all(cls):
        """Return an explicit whole-domain selection, bypassing an active ROI."""
        return cls._make("all")

    def _combine(self, other, kind):
        if not isinstance(other, ROI):
            return NotImplemented
        if self._coordinates and other._coordinates and self._coordinates != other._coordinates:
            raise ValueError("Combining pixel and native geometry requires explicit conversion to masks.")
        if self._units is not None and other._units is not None and self._units != other._units:
            raise ValueError("ROI coordinate units must match.")
        return self._make(kind, (self, other), self._coordinates or other._coordinates,
                          self._units if self._units is not None else other._units)

    def __or__(self, other):
        return self._combine(other, "union")

    def __and__(self, other):
        return self._combine(other, "intersection")

    def __sub__(self, other):
        return self._combine(other, "difference")

    def __invert__(self):
        return self._make("complement", (self,), self._coordinates, self._units)

    def resolve(self, target, *, grid=None):
        """Return detached labelled Boolean membership, without checking intensities.

        Args:
            target (xarray.DataArray):
                Two-dimensional intensity axes defining the current domain.

        Keyword Arguments:
            grid (ImageGrid or None):
                Optional validated native grid from the image coordinate service.
                Standalone descriptions otherwise validate separable x/y centres.

        Returns:
            xarray.DataArray:
                Independent inclusion array with the target's coordinates.

        Raises:
            ValueError:
                Geometry, mask labels or physical unit labels are incompatible.
        """
        target = _target(target)
        kind, values = self._kind, self._values
        if kind == "all":
            result = np.ones(target.shape, dtype=bool)
        elif kind == "mask":
            if geometry_signature(target) != values[0]:
                raise ValueError("ROI mask geometry/labels do not match the target.")
            result = np.unpackbits(np.frombuffer(bytes.fromhex(values[2]), dtype=np.uint8),
                                   count=int(np.prod(values[1]))).reshape(values[1]).astype(bool)
        elif kind in {"union", "intersection", "difference", "complement"}:
            left = values[0].resolve(target, grid=grid).values
            if kind == "complement":
                result = ~left
            else:
                right = values[1].resolve(target, grid=grid).values
                result = left | right if kind == "union" else left & (right if kind == "intersection" else ~right)
        else:
            if self._coordinates is CoordinateMode.NATIVE:
                if _units(target) != self._units:
                    raise ValueError("Native ROI units do not match the target; no implicit unit conversion occurs.")
                if grid is None:
                    vectors = [np.asarray(target[axis].values) for axis in ("x", "y")]
                    for vector in vectors:
                        if vector.dtype.kind not in "iuf" or not np.isfinite(vector).all():
                            raise ValueError("Native ROI coordinates must be finite real vectors.")
                        steps = np.diff(vector)
                        if len(steps) and not (np.all(steps > 0) or np.all(steps < 0)):
                            raise ValueError("Native ROI coordinates must be strictly monotonic.")
                else:
                    vectors = [grid.x, grid.y]
            else:
                vectors = [np.arange(target.sizes[axis]) for axis in ("x", "y")]
            x, y = np.meshgrid(*vectors)
            if kind == "box":
                xmin, xmax, ymin, ymax = values
                result = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)
            elif kind == "ellipse":
                cx, cy, rx, ry = values
                result = ((x-cx)/rx)**2 + ((y-cy)/ry)**2 <= 1
            else:
                result = points_in_poly(np.column_stack((x.ravel(), y.ravel())), np.asarray(values)).reshape(target.shape)
        return xr.DataArray(result.copy(), dims=target.dims, coords=target.coords, name="included").copy(deep=True)

    def to_dict(self):
        """Export a detached version-1 JSON-compatible description for explicit replay."""
        values = [value.to_dict() if isinstance(value, ROI) else value for value in self._values]
        return json.loads(json.dumps({"version": 1, "kind": self._kind, "values": values,
                                      "coordinates": self._coordinates, "units": self._units}))

    @classmethod
    def from_dict(cls, record):
        """Validate and import a version-1 record without evaluating arbitrary code.

        Args:
            record (dict):
                Record exported by :meth:`to_dict`; unknown fields/versions fail.

        Returns:
            ROI:
                Independent validated immutable description.
        """
        if not isinstance(record, dict) or set(record) != {"version", "kind", "values", "coordinates", "units"}:
            raise ValueError("Invalid ROI record fields.")
        if type(record["version"]) is not int or record["version"] != 1:
            raise ValueError("Unsupported ROI record version.")
        kind, values = record["kind"], record["values"]
        if not isinstance(values, list):
            raise TypeError("ROI record values must be a list.")
        mode, units = record["coordinates"], record["units"]
        reference = None
        if mode is not None:
            mode = _mode(mode)
            if mode is CoordinateMode.NATIVE:
                if not isinstance(units, list) or len(units) != 2 or any(not isinstance(v, str) for v in units):
                    raise ValueError("Native records require two unit labels.")
                reference = xr.DataArray([[0]], dims=("y", "x"), coords={"x": [0], "y": [0]})
                for axis, unit in zip(("x", "y"), units):
                    reference[axis].attrs["units"] = unit
            elif units is not None:
                raise ValueError("Pixel records must not specify native units.")
        elif units is not None:
            raise ValueError("Unit labels require native coordinates.")
        options = {"coordinates": mode, "reference": reference}
        if kind == "box" and len(values) == 4 and mode is not None:
            result = cls.box(*values, **options)
        elif kind == "ellipse" and len(values) == 4 and mode is not None:
            result = cls.ellipse(values[:2], values[2:], **options)
        elif kind == "polygon" and mode is not None:
            result = cls.polygon(values, **options)
        elif kind == "all" and not values and mode is None:
            result = cls.all()
        elif kind in {"union", "intersection", "difference", "complement"} and len(values) == (
                1 if kind == "complement" else 2):
            children = [cls.from_dict(value) for value in values]
            result = ~children[0] if kind == "complement" else children[0]._combine(children[1], kind)
        elif kind == "mask" and mode is None and len(values) == 3:
            signature, shape, packed = values
            try:
                geometry = json.loads(signature)
                coordinates = xr.Dataset.from_dict(geometry["coords"])
                target = xr.DataArray(np.empty(shape), dims=("y", "x"), coords=coordinates.coords)
                raw = np.frombuffer(bytes.fromhex(packed), dtype=np.uint8)
                if len(raw) != (target.size + 7)//8 or list(shape) != geometry["shape"]:
                    raise ValueError("Invalid packed mask geometry.")
                included = np.unpackbits(raw, count=target.size).reshape(shape).astype(bool)
                result = cls.from_mask(included, reference=target)
            except (TypeError, KeyError, ValueError) as error:
                raise ValueError("Invalid ROI mask record.") from error
        else:
            raise ValueError("Invalid ROI record kind or parameters.")
        if result._coordinates != mode or result._units != (tuple(units) if units is not None else None):
            raise ValueError("ROI record coordinate fields are inconsistent.")
        return result
