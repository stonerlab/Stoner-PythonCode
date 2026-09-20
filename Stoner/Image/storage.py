"""Validate and exchange eager xarray image values, exclusions and calibration."""

from copy import deepcopy
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from warnings import warn

import numpy as np
import xarray as xr

from ..core.base import TypeHintedDict
from ..core.storage import _copy_metadata
from .storage_coordinates import pack_coordinates

__all__ = ["Frame", "ImageStorage"]


def _fill(value, dtype):
    """Check that a scalar fill value is representable in the image dtype."""
    if np.ndim(value) != 0:
        raise ValueError("Fill value must be scalar")
    converted = np.ma.array(np.empty(0, dtype=dtype), fill_value=value).fill_value
    if dtype.kind in "biu" and converted != value:
        raise ValueError("Fill value is outside the intensity dtype")
    if dtype.kind == "f" and np.isfinite(value) and not np.isfinite(converted):
        raise ValueError("Fill value is outside the intensity dtype")
    return converted


@dataclass
class Frame:
    """Carry stable frame identity, display name, typed metadata and fill value."""

    id: str
    name: str
    metadata: TypeHintedDict
    fill_value: object

    def __deepcopy__(self, memo):
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        result.id, result.name = self.id, self.name
        result.metadata = _copy_metadata(self.metadata, memo)
        result.fill_value = deepcopy(self.fill_value, memo)
        return result


@dataclass
class ImageStorage:
    """Describe a detached, versioned image or ragged-stack package.

    Attributes:
        kind (str):
            ``image`` for y/x values or ``stack`` for frame/y/x values.
        dataset (xarray.Dataset):
            Eager intensity and Boolean exclusions, dimension coordinates, and
            per-frame valid heights/widths for stacks. Padding is excluded zero.
        metadata (TypeHintedDict):
            Typed owner metadata, independent of Dataset attrs.
        fill_value (scalar):
            Fill value compatible with the intensity dtype.
        frames (list of Frame):
            Ordered frame records; empty for an image.
        version (int):
            Package version, currently one.
    """

    kind: str
    dataset: xr.Dataset
    metadata: TypeHintedDict
    fill_value: object
    frames: list = field(default_factory=list)
    version: int = 1

    def validate(self):
        """Reject unsupported ranks, dtypes, identities, calibration and padding."""
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported image storage version")
        if self.kind not in ("image", "stack"):
            raise ValueError("Unknown image storage kind")
        if not isinstance(self.dataset, xr.Dataset):
            raise TypeError("Image storage requires an xarray Dataset")
        ds = self.dataset
        dims = ("y", "x") if self.kind == "image" else ("frame", "y", "x")
        variables = {"intensity", "excluded"}
        if self.kind == "stack":
            variables |= {"valid_height", "valid_width"}
        if set(ds.data_vars) != variables or set(ds.sizes) != set(dims):
            raise ValueError("Unexpected image variables or dimensions")
        for name in variables:
            if not isinstance(ds[name].data, np.ndarray):
                raise TypeError("Only eager NumPy-backed variables are supported")
        if ds.intensity.dims != dims or ds.excluded.dims != dims:
            raise ValueError("Intensity and exclusions must have the prescribed dimension order")
        dtype = ds.intensity.dtype
        if dtype.kind not in "biuf" or dtype.fields is not None:
            raise TypeError("Images require real numeric or Boolean intensities")
        if ds.excluded.dtype != np.dtype(bool):
            raise TypeError("Exclusions must be Boolean")
        if not isinstance(self.metadata, TypeHintedDict):
            raise TypeError("Image metadata must be a TypeHintedDict")
        _fill(self.fill_value, dtype)
        for axis in dims:
            if axis not in ds.coords or ds.coords[axis].dims != (axis,):
                raise ValueError("Explicit one-dimensional dimension coordinates are required")
        for name, coord in ds.coords.items():
            if not isinstance(coord.data, np.ndarray):
                raise TypeError("Only eager coordinates are supported")
            if name != "frame" and coord.dtype.kind not in "biufMm":
                raise TypeError("Coordinates must be numeric or temporal")
        for axis in ("y", "x"):
            if ds[axis].dtype.kind not in "iuf" or not np.isfinite(ds[axis].values).all():
                raise ValueError("Spatial coordinates must be finite real numbers")
        if self.kind == "image":
            if self.frames or min(ds.sizes.values()) == 0:
                raise ValueError("Single images require nonempty axes and no frame records")
            if any(coord.dims not in ((), ("y",), ("x",), ("y", "x")) for coord in ds.coords.values()):
                raise ValueError("Image calibration must use scalar, axis or y/x coordinate maps")
        else:
            self._validate_frames()

    def _validate_frames(self):
        """Validate ordered frame records, valid rectangles and physical tails."""
        ds = self.dataset
        ids = []
        for frame in self.frames:
            if not isinstance(frame, Frame) or not isinstance(frame.metadata, TypeHintedDict):
                raise TypeError("Frames require typed frame records")
            if not isinstance(frame.id, str) or str(UUID(frame.id)) != frame.id:
                raise ValueError("Frame IDs must be canonical UUID strings")
            if not isinstance(frame.name, str):
                raise TypeError("Frame names must be strings")
            _fill(frame.fill_value, ds.intensity.dtype)
            ids.append(frame.id)
        if len(set(ids)) != len(ids) or list(ds.frame.values) != ids:
            raise ValueError("Frame identities must be unique and match record order")
        for axis in ("y", "x"):
            if not np.array_equal(ds[axis].values, np.arange(ds.sizes[axis])):
                raise ValueError("Stack y/x coordinates must be positional pixels")
        for name in ("valid_height", "valid_width"):
            if ds[name].dims != ("frame",) or ds[name].dtype.kind not in "iu":
                raise ValueError("Valid extents must be integer frame variables")
        for i, frame in enumerate(self.frames):
            height, width = int(ds.valid_height.values[i]), int(ds.valid_width.values[i])
            if not 0 < height <= ds.sizes["y"] or not 0 < width <= ds.sizes["x"]:
                raise ValueError("Invalid frame extent")
            padding = np.ones((ds.sizes["y"], ds.sizes["x"]), bool)
            padding[:height, :width] = False
            if not ds.excluded.values[i][padding].all() or np.any(ds.intensity.values[i][padding] != 0):
                raise ValueError("Padding must remain excluded zero")
        for name, coord in ds.coords.items():
            if name in ("frame", "y", "x"):
                continue
            if coord.dims == ("frame", "y", "x"):
                if coord.dtype.kind not in "iuf":
                    raise ValueError("Coordinate maps must be real numeric values")
                for i, (height, width) in enumerate(zip(ds.valid_height.values, ds.valid_width.values)):
                    padding = np.ones(coord.shape[1:], bool)
                    padding[:height, :width] = False
                    if not np.isnan(coord.values[i][padding]).all():
                        raise ValueError("Coordinate maps require NaN padding")
            elif name in ("physical_y", "physical_x", "index_y", "index_x"):
                axis = name[-1]
                if coord.dims != ("frame", axis) or coord.dtype.kind not in "iuf":
                    raise ValueError("Physical coordinates require frame and spatial dimensions")
                extents = ds.valid_height.values if axis == "y" else ds.valid_width.values
                for values, extent in zip(coord.values, extents):
                    if not np.isfinite(values[:extent]).all() or not np.isnan(values[extent:]).all():
                        raise ValueError("Physical coordinates require finite valid values and NaN padding")
            elif coord.dims not in ((), ("frame",)):
                raise ValueError("Unsupported stack calibration coordinate")

    @classmethod
    def from_numpy(cls, values, *, metadata=None):
        """Import a two-dimensional image, preserving raw masked values and dtype."""
        array = np.ma.asarray(values)
        if array.ndim != 2:
            raise ValueError("An image must have two dimensions")
        ds = xr.Dataset({"intensity": (("y", "x"), array.data.copy()),
                         "excluded": (("y", "x"), np.ma.getmaskarray(array).copy())},
                        coords={"y": np.arange(array.shape[0]), "x": np.arange(array.shape[1])})
        fill = np.ma.array(np.empty(0, dtype=array.dtype), fill_value=array.fill_value).fill_value
        result = cls("image", ds, _copy_metadata(metadata) if metadata is not None else TypeHintedDict(), fill)
        result.validate()
        return result

    @classmethod
    def from_images(cls, images, *, names=None, metadata=None):
        """Pack image packages or arrays into detached, excluded-zero padded storage.

        Notes:
            Physical axes become padded per-frame coordinates. Scalar coordinates
            become frame coordinates. Native attributes must agree across inputs;
            incompatible units and unsupported auxiliary spatial mappings raise.
        """
        images = [image.copy() if isinstance(image, cls) else cls.from_numpy(image) for image in images]
        names = list(names) if names is not None else [str(i) for i in range(len(images))]
        if len(names) != len(images):
            raise ValueError("One name is required per image")
        for image in images:
            if image.kind != "image":
                raise ValueError("Only image packages can be packed")
        dtype = np.result_type(*[image.dataset.intensity.dtype for image in images]) if images else np.dtype(float)
        shape = (len(images), max((im.dataset.sizes["y"] for im in images), default=0),
                 max((im.dataset.sizes["x"] for im in images), default=0))
        values, excluded = np.zeros(shape, dtype=dtype), np.ones(shape, dtype=bool)
        frames, heights, widths = [], [], []
        for i, (image, name) in enumerate(zip(images, names)):
            raw = image.dataset.intensity.values
            if raw.dtype != dtype and raw.dtype.kind in "iu" and not np.array_equal(
                raw.astype(object), raw.astype(dtype).astype(object)
            ):
                raise ValueError("Stack dtype promotion would lose integer precision")
            height, width = raw.shape
            heights.append(height)
            widths.append(width)
            values[i, :height, :width] = raw
            excluded[i, :height, :width] = image.dataset.excluded.values
            frames.append(Frame(str(uuid4()), name, image.metadata, _fill(image.fill_value, dtype)))
        ds = xr.Dataset({"intensity": (("frame", "y", "x"), values),
                         "excluded": (("frame", "y", "x"), excluded),
                         "valid_height": ("frame", np.array(heights, dtype=int)),
                         "valid_width": ("frame", np.array(widths, dtype=int))},
                        coords={"frame": [f.id for f in frames], "y": np.arange(shape[1]), "x": np.arange(shape[2])})
        pack_coordinates(ds, images)
        result = cls("stack", ds, _copy_metadata(metadata) if metadata is not None else TypeHintedDict(),
                     np.ma.array(np.empty(0, dtype=dtype)).fill_value, frames)
        result.fill_value = np.ma.array(np.empty(0, dtype=dtype), fill_value=result.fill_value).fill_value
        result.validate()
        return result

    @classmethod
    def from_xarray(cls, dataset, *, metadata=None, frame_metadata=None):
        """Import a native Dataset without interpreting attrs as typed metadata."""
        if not isinstance(dataset, xr.Dataset) or "intensity" not in dataset:
            raise TypeError("Expected a Dataset containing intensity")
        kind = "stack" if dataset.intensity.ndim == 3 else "image"
        fill = np.ma.array(np.empty(0, dtype=dataset.intensity.dtype)).fill_value
        fill = np.ma.array(np.empty(0, dtype=dataset.intensity.dtype), fill_value=fill).fill_value
        frames = []
        if kind == "stack":
            if "frame" not in dataset.coords:
                raise ValueError("Stack frame IDs are required")
            supplied = frame_metadata if frame_metadata is not None else {}
            if set(supplied) - set(dataset.frame.values.tolist()):
                raise ValueError("Metadata refers to unknown frame IDs")
            frames = [Frame(identity, identity, _copy_metadata(supplied.get(identity, TypeHintedDict())), fill)
                      for identity in dataset.frame.values.tolist()]
        elif frame_metadata:
            raise ValueError("Single images cannot have frame metadata")
        result = cls(kind, dataset.copy(deep=True),
                     _copy_metadata(metadata) if metadata is not None else TypeHintedDict(), fill, frames)
        result.validate()
        return result

    def copy(self):
        """Validate and detach all values, coordinates, frame records and metadata."""
        self.validate()
        return deepcopy(self)

    def __deepcopy__(self, memo):
        self.validate()
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        result.kind, result.version = self.kind, self.version
        result.dataset = deepcopy(self.dataset, memo)
        result.metadata = _copy_metadata(self.metadata, memo)
        result.frames = deepcopy(self.frames, memo)
        result.fill_value = deepcopy(self.fill_value, memo)
        return result

    def to_numpy(self, *, masked=True, dtype=None):
        """Export detached raw values, including exclusions unless disabled."""
        self.validate()
        values = np.array(self.dataset.intensity.data, dtype=dtype, copy=True)
        if not masked:
            return values
        return np.ma.array(values, mask=self.dataset.excluded.values.copy(), fill_value=self.fill_value, copy=False)

    def to_xarray(self):
        """Export detached numerical and coordinate state, omitting typed metadata."""
        self.validate()
        if self.metadata or any(frame.metadata for frame in self.frames):
            warn("Xarray export omits typed metadata; retain ImageStorage for lossless use", UserWarning, stacklevel=2)
        return deepcopy(self.dataset)
