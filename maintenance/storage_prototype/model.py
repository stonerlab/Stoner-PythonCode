"""Exercise the batch 2 ownership model without replacing Stoner classes.

Only the primitives needed by the acceptance tests and benchmarks are provided.
These classes are experimental and are not substitutes for the public API.
"""

from contextlib import contextmanager
from collections.abc import MutableMapping
from copy import deepcopy
from dataclasses import dataclass
import re
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import xarray as xr

from Stoner.core.base import TypeHintedDict


def numeric(values):
    """Reject non-NumPy numeric dtypes rather than coercing them silently."""
    result = np.asarray(values)
    if result.dtype.kind not in "biufc":
        raise TypeError("Homogeneous NumPy numeric values required")
    return result


def metadata_copy(metadata, memo=None):
    """Copy typed metadata, reporting an uncopyable key explicitly."""
    if metadata is None:
        return TypeHintedDict()
    if not isinstance(metadata, TypeHintedDict):
        raise TypeError("Supply TypeHintedDict to preserve type hints")
    result = metadata.copy()
    memo = {} if memo is None else memo
    memo[id(metadata)] = result
    for key, value in metadata.items():
        try:
            super(TypeHintedDict, result).__setitem__(key, deepcopy(value, memo))
        except Exception as error:
            raise TypeError(f"Cannot copy metadata key {key!r}") from error
    return result


def validate_ids(ids):
    """Require unique UUID strings within one owner."""
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate identities")
    for identity in ids:
        if not isinstance(identity, str):
            raise ValueError("Identity must be a UUID string")
        UUID(identity)


def validate_mask(mask, shape):
    """Require explicit Boolean exclusions with exactly the value shape."""
    if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != shape:
        raise ValueError("Boolean mask must match values")


def masked_copy(values, excluded, fill_value):
    """Construct an independent numerical boundary with recoverable exclusions."""
    return np.ma.array(values, mask=excluded, fill_value=fill_value, copy=True)


def resolve(schema, selector):
    """Resolve positions, exact names and patterns without collapsing duplicates."""
    count = len(schema)
    if isinstance(selector, (int, np.integer)):
        if not -count <= selector < count:
            raise IndexError(selector)
        return int(selector) % count
    if isinstance(selector, slice):
        return list(range(count))[selector]
    if isinstance(selector, str):
        exact = [i for i, col in enumerate(schema) if col.header == selector]
        if exact:
            return exact[0]
        matches = resolve(schema, re.compile(selector))
        return matches[0]
    if isinstance(selector, re.Pattern):
        matches = [i for i, col in enumerate(schema) if selector.search(col.header)]
        if not matches:
            raise KeyError(selector.pattern)
        return matches
    if isinstance(selector, (list, tuple)):
        result = []
        for item in selector:
            found = resolve(schema, item)
            result.extend(found if isinstance(found, list) else [found])
        return result
    raise TypeError(selector)


def role_groups(schema):
    """Resolve explicitly assigned x groups with absolute error positions."""
    starts = [i for i, col in enumerate(schema) if col.role == "x"]
    groups = []
    for start, stop in zip(starts, starts[1:] + [len(schema)]):
        positions = {role: [i for i in range(start, stop) if schema[i].role == role] for role in "dyezfuvw"}
        groups.append(dict(xcol=start, xerr=next(iter(positions["d"]), None), ycol=positions["y"],
                           yerr=positions["e"], zcol=positions["z"], zerr=positions["f"],
                           ucol=positions["u"], vcol=positions["v"], wcol=positions["w"]))
    return groups


@dataclass(frozen=True)
class Column:
    """Stable identity, displayed header and role for one column."""

    id: str
    header: str
    role: str


@dataclass
class DataStorage:
    """Detached versioned values, exclusions and descriptive state."""

    values: pd.DataFrame
    excluded: np.ndarray
    schema: list
    metadata: TypeHintedDict
    fill_value: object
    version: int = 1
    dtype: object = None

    def __deepcopy__(self, memo):
        """Copy typed metadata without invoking its value-inference setter."""
        return DataStorage(self.values.copy(deep=True), self.excluded.copy(), deepcopy(self.schema, memo),
                           metadata_copy(self.metadata, memo), deepcopy(self.fill_value, memo), self.version, self.dtype)

    def validate(self):
        """Reject inconsistent packages before they acquire an owner."""
        if self.version != 1:
            raise ValueError("Unsupported storage version")
        ids = [col.id for col in self.schema]
        validate_ids(ids)
        if list(self.values.columns) != ids or not self.values.index.equals(pd.RangeIndex(len(self.values))):
            raise ValueError("Schema/axes mismatch")
        if any(not isinstance(col.header, str) or col.role not in tuple("xyzdefuvw.") for col in self.schema):
            raise ValueError("Invalid schema")
        dtypes = list(self.values.dtypes)
        if not isinstance(self.dtype, np.dtype) or self.dtype.kind not in "biufc":
            raise TypeError("Explicit NumPy dtype required, including empty tables")
        if any(not isinstance(dtype, np.dtype) or dtype.kind not in "biufc" for dtype in dtypes):
            raise TypeError("Unsupported dtype")
        if dtypes and any(dtype != dtypes[0] for dtype in dtypes):
            raise TypeError("Mixed dtypes require explicit conversion")
        if dtypes and dtypes[0] != self.dtype:
            raise ValueError("Stored dtype differs from frame")
        validate_mask(self.excluded, self.values.shape)
        metadata_copy(self.metadata)
        np.ma.array(self.values.to_numpy(dtype=self.dtype), fill_value=self.fill_value)


class MetadataView(MutableMapping):
    """Resolve durable metadata on each access and guard saved mapping references."""

    def __init__(self, owner, getter):
        self.owner, self.getter = owner, getter

    def __getitem__(self, key):
        return self.getter()[key]

    def __setitem__(self, key, value):
        self.owner._check_edit()
        self.getter()[key] = value

    def __delitem__(self, key):
        self.owner._check_edit()
        del self.getter()[key]

    def __iter__(self):
        return iter(self.getter())

    def __len__(self):
        return len(self.getter())

    def type(self, key):
        """Return the stored hint without inferring it from the value."""
        return self.getter().type(key)


class Mask:
    """Route Boolean cell edits to an owner without exposing its stored mask."""

    def __init__(self, owner):
        self.owner = owner

    def __getitem__(self, key):
        return self.owner.to_numpy().mask[key].copy()

    def __setitem__(self, key, value):
        self.owner._check_edit()
        with self.owner.edit_numpy() as draft:
            draft.mask[key] = value

    def __array__(self, dtype=None, copy=None):
        return np.array(self.owner.to_numpy().mask, dtype=dtype, copy=True)


class Editing:
    """Implement detached numerical drafts with atomic commit and rollback."""

    _editing = False

    def _check_edit(self):
        if self._editing:
            raise RuntimeError("Owner already being edited")

    @property
    def mask(self):
        """Return the owner-aware exclusion interface."""
        return Mask(self)

    @contextmanager
    def edit_numpy(self):
        """Yield a fixed-shape/dtype draft and commit only on normal exit."""
        self._check_edit()
        draft = self.to_numpy()
        shape, dtype = draft.shape, draft.dtype
        self._editing = True
        try:
            yield draft
            if draft.shape != shape or draft.dtype != dtype:
                raise ValueError("Editing cannot change shape or dtype")
            self._commit_array(draft)
        finally:
            self._editing = False


class Table(Editing):
    """Own a numeric DataFrame and explicit schema/mask for the experiment."""

    def __init__(self, values, headers=None, roles=None, mask=None, metadata=None):
        values = numeric(values)
        if values.ndim != 2:
            raise ValueError("Table must be two-dimensional")
        headers = list(headers) if headers is not None else [f"Column {i}" for i in range(values.shape[1])]
        roles = list(roles) if roles is not None else ["."] * values.shape[1]
        if len(headers) != values.shape[1] or len(roles) != len(headers):
            raise ValueError("Schema width mismatch")
        schema = [Column(str(uuid4()), header, role) for header, role in zip(headers, roles)]
        excluded = np.zeros(values.shape, bool) if mask is None else np.array(mask, copy=True)
        package = DataStorage(pd.DataFrame(values, columns=[col.id for col in schema], copy=True), excluded,
                              schema, metadata_copy(metadata), np.ma.array(values).fill_value, dtype=values.dtype)
        package.validate()
        self._state = package

    @classmethod
    def from_storage(cls, package):
        """Validate and detach every package field before importing."""
        package.validate()
        result = cls.__new__(cls)
        result._state = deepcopy(package)
        return result

    @classmethod
    def from_pandas(cls, frame, *, mask=None, setas=None, metadata=None, dtype=None, index="require_range"):
        """Import numeric columns positionally, requiring explicit index discard."""
        if index not in ("require_range", "discard"):
            raise ValueError("Invalid index policy")
        if index == "require_range" and not frame.index.equals(pd.RangeIndex(len(frame))):
            raise ValueError("Explicit index discard required")
        if any(not isinstance(item, np.dtype) or item.kind not in "biufc" for item in frame.dtypes):
            raise TypeError("Convert unsupported dtypes before import")
        if dtype is None and len(set(frame.dtypes)) > 1:
            raise TypeError("Mixed columns require dtype")
        if isinstance(frame.columns, pd.MultiIndex):
            if frame.columns.nlevels != 2:
                raise ValueError("Expected legacy two-level columns")
            headers = list(frame.columns.get_level_values(0))
            roles = list(frame.columns.get_level_values(1))
        else:
            headers, roles = list(frame.columns), None
        return cls(frame.to_numpy(dtype=dtype, copy=True), headers, setas if setas is not None else roles,
                   mask, metadata)

    def export_storage(self):
        """Return a lossless independent package."""
        return deepcopy(self._state)

    def to_numpy(self, *, masked=True, dtype=None):
        """Return detached raw values or a masked numerical boundary."""
        values = self._state.values.to_numpy(dtype=self._state.dtype if dtype is None else dtype, copy=True)
        if not masked:
            return values
        return np.ma.array(values, mask=self._state.excluded.copy(), fill_value=self._state.fill_value, copy=False)

    @property
    def column_ids(self):
        """Return stable identities in display order."""
        return tuple(col.id for col in self._state.schema)

    @property
    def metadata(self):
        """Expose typed metadata outside numerical editing transactions."""
        return MetadataView(self, lambda: self._state.metadata)

    def column(self, selector):
        """Return an independent selected column with its exclusion mask."""
        positions = resolve(self._state.schema, selector)
        values = self._state.values.iloc[:, positions].to_numpy(copy=True)
        return np.ma.array(values, mask=self._state.excluded[:, positions].copy(),
                           fill_value=self._state.fill_value, copy=False)

    def __getitem__(self, key):
        return self.to_numpy()[key]

    def __setitem__(self, key, value):
        self._check_edit()
        if value is np.ma.masked:
            self.mask[key] = True
            return
        incoming = numeric(np.ma.getdata(value))
        dtype = np.result_type(self._state.dtype, incoming.dtype)
        draft = self.to_numpy(dtype=dtype)
        draft[key] = value
        self._commit_array(draft)

    def _commit_array(self, draft):
        state = self._state
        candidate = DataStorage(pd.DataFrame(np.array(draft.data), columns=self.column_ids),
                                np.ma.getmaskarray(draft).copy(), state.schema,
                                state.metadata, draft.fill_value, dtype=draft.dtype)
        candidate.validate()
        self._state = candidate

    @contextmanager
    def edit_pandas(self):
        """Edit raw values without changing axes, dtype or exclusions."""
        self._check_edit()
        original = self._state.values
        draft = original.copy(deep=True)
        self._editing = True
        try:
            yield draft
            if not draft.index.equals(original.index) or not draft.columns.equals(original.columns):
                raise ValueError("Editing cannot change axes")
            if not draft.dtypes.equals(original.dtypes):
                raise ValueError("Editing cannot change dtype")
            self._state.values = draft.copy(deep=True)
        finally:
            self._editing = False

    def select(self, selector):
        """Select/reorder schema and exclusions with values, duplicating IDs safely."""
        positions = resolve(self._state.schema, selector)
        if isinstance(positions, int):
            positions = [positions]
        package = self.export_storage()
        seen, schema = set(), []
        for position in positions:
            col = package.schema[position]
            identity = str(uuid4()) if col.id in seen else col.id
            seen.add(identity)
            schema.append(Column(identity, col.header, col.role))
        package.values = package.values.iloc[:, positions].copy()
        package.values.columns = [col.id for col in schema]
        package.excluded = package.excluded[:, positions].copy()
        package.schema = schema
        return self.from_storage(package)

    def append(self, rows):
        """Append homogeneous numeric rows positionally, retaining existing masks."""
        self._check_edit()
        rows = np.ma.atleast_2d(rows)
        numeric(rows.data)
        if rows.shape[1] != self._state.values.shape[1]:
            raise ValueError("Row width mismatch")
        state = self._state
        dtype = np.result_type(state.dtype, rows.dtype)
        added = pd.DataFrame(np.array(rows.data, dtype=dtype, copy=True), columns=self.column_ids)
        values = pd.concat([state.values.astype(dtype), added], ignore_index=True)
        candidate = DataStorage(values, np.concatenate([state.excluded, np.ma.getmaskarray(rows)], axis=0),
                                state.schema, state.metadata, state.fill_value, dtype=dtype)
        candidate.validate()
        self._state = candidate
        return self


@dataclass
class Frame:
    """Frame identity and descriptive state independent of padded pixels."""

    id: str
    name: str
    metadata: TypeHintedDict
    fill_value: object

    def __deepcopy__(self, memo):
        """Copy frame metadata without reparsing explicit string values."""
        return Frame(self.id, self.name, metadata_copy(self.metadata, memo), deepcopy(self.fill_value, memo))


@dataclass
class ImageStorage:
    """Detached xarray intensity, exclusion, extent and frame state."""

    dataset: xr.Dataset
    frames: list
    metadata: TypeHintedDict
    fill_value: object
    version: int = 1

    def __deepcopy__(self, memo):
        """Copy backend arrays and typed metadata independently."""
        return ImageStorage(self.dataset.copy(deep=True), deepcopy(self.frames, memo),
                            metadata_copy(self.metadata, memo), deepcopy(self.fill_value, memo), self.version)

    def validate(self):
        """Check eager stack state and reject unmasked or nonzero padding."""
        if self.version != 1:
            raise ValueError("Unsupported storage version")
        ds = self.dataset
        if set(ds.data_vars) != {"intensity", "excluded", "valid_height", "valid_width"}:
            raise ValueError("Unexpected variables")
        for variable in ("intensity", "excluded"):
            if ds[variable].dims != ("frame", "y", "x") or not isinstance(ds[variable].data, np.ndarray):
                raise ValueError("Expected eager frame/y/x storage")
        values = numeric(ds.intensity.values)
        if values.dtype.kind == "c":
            raise TypeError("Complex images unsupported")
        validate_mask(ds.excluded.values, values.shape)
        ids = [frame.id for frame in self.frames]
        validate_ids(ids)
        if list(ds.frame.values) != ids:
            raise ValueError("Frame records/coordinates differ")
        for extent in ("valid_height", "valid_width"):
            if ds[extent].dims != ("frame",) or ds[extent].dtype.kind not in "iu":
                raise ValueError("Extents must be integers per frame")
        for i, frame in enumerate(self.frames):
            height, width = int(ds.valid_height.values[i]), int(ds.valid_width.values[i])
            if not 0 < height <= values.shape[1] or not 0 < width <= values.shape[2]:
                raise ValueError("Invalid extent")
            padding = np.ones(values.shape[1:], bool)
            padding[:height, :width] = False
            if not ds.excluded.values[i][padding].all() or np.any(values[i][padding] != 0):
                raise ValueError("Padding must be excluded zero")
            metadata_copy(frame.metadata)
            np.ma.array(values[i], fill_value=frame.fill_value)
        metadata_copy(self.metadata)
        np.ma.array(values, fill_value=self.fill_value)


class Stack(Editing):
    """Own an eager padded xarray Dataset with durable frame handles."""

    def __init__(self, images, names=None, metadata=None):
        images = [np.ma.array(image, copy=True) for image in images]
        if any(image.ndim != 2 or min(image.shape) == 0 for image in images):
            raise ValueError("Nonempty two-dimensional frames required")
        for image in images:
            numeric(image.data)
        dtype = np.result_type(*[image.dtype for image in images]) if images else np.dtype(float)
        shape = (len(images), max((a.shape[0] for a in images), default=0),
                 max((a.shape[1] for a in images), default=0))
        values, excluded = np.zeros(shape, dtype), np.ones(shape, bool)
        names = list(names) if names is not None else [str(i) for i in range(len(images))]
        if len(names) != len(images):
            raise ValueError("Name count differs")
        frames = []
        for i, image in enumerate(images):
            height, width = image.shape
            values[i, :height, :width] = image.data
            excluded[i, :height, :width] = np.ma.getmaskarray(image)
            frames.append(Frame(str(uuid4()), names[i], TypeHintedDict(), image.fill_value))
        ds = xr.Dataset({"intensity": (("frame", "y", "x"), values),
                         "excluded": (("frame", "y", "x"), excluded),
                         "valid_height": ("frame", np.array([a.shape[0] for a in images], dtype=int)),
                         "valid_width": ("frame", np.array([a.shape[1] for a in images], dtype=int))},
                        coords={"frame": [f.id for f in frames], "y": np.arange(shape[1]), "x": np.arange(shape[2])})
        self._state = ImageStorage(ds, frames, metadata_copy(metadata), np.ma.array(values).fill_value)
        self._state.validate()

    @classmethod
    def from_storage(cls, package):
        """Import independently after checking coordinates, extents and padding."""
        package.validate()
        result = cls.__new__(cls)
        result._state = deepcopy(package)
        return result

    def export_storage(self):
        """Return a lossless independent stack package."""
        return deepcopy(self._state)

    def to_numpy(self, *, masked=True, dtype=None):
        """Return padded values with explicit padding exclusions by default."""
        ds = self._state.dataset
        values = np.array(ds.intensity.values, dtype=dtype, copy=True)
        return np.ma.array(values, mask=ds.excluded.values.copy(), fill_value=self._state.fill_value,
                           copy=False) if masked else values

    def _commit_array(self, draft):
        candidate = self.export_storage()
        candidate.dataset["intensity"].data = draft.data.copy()
        candidate.dataset["excluded"].data = np.ma.getmaskarray(draft).copy()
        candidate.validate()
        self._state = candidate

    @contextmanager
    def edit_xarray(self):
        """Edit pixels/exclusions while preserving structural coordinate state."""
        self._check_edit()
        original = self._state.dataset
        draft = original.copy(deep=True)
        self._editing = True
        try:
            yield draft
            if not draft.coords.to_dataset().identical(original.coords.to_dataset()):
                raise ValueError("Coordinate edits forbidden")
            for name in original.data_vars:
                if name not in draft or draft[name].dtype != original[name].dtype:
                    raise ValueError("Dtype/variable edits forbidden")
            for name in ("valid_height", "valid_width"):
                if not draft[name].identical(original[name]):
                    raise ValueError("Extent edits forbidden")
            candidate = self.export_storage()
            candidate.dataset = draft.copy(deep=True)
            candidate.validate()
            self._state = candidate
        finally:
            self._editing = False

    def __getitem__(self, item):
        if isinstance(item, str):
            item = next(i for i, frame in enumerate(self._state.frames) if frame.name == item)
        return FrameHandle(self, self._state.frames[item].id)

    def reorder(self, positions):
        """Move frames and coordinates while preserving saved handle identity."""
        self._check_edit()
        candidate = self.export_storage()
        candidate.dataset = candidate.dataset.isel(frame=positions).copy(deep=True)
        candidate.frames = [candidate.frames[i] for i in positions]
        candidate.validate()
        self._state = candidate

    def insert(self, position, image, name="inserted"):
        """Rebuild padded storage for one inserted frame, retaining existing IDs."""
        self._check_edit()
        if set(self._state.dataset.coords) != {"frame", "y", "x"}:
            raise NotImplementedError("Calibrated insertion needs a coordinate remapping adapter")
        images = [self[i].to_numpy() for i in range(len(self._state.frames))]
        images.insert(position, image)
        names = [frame.name for frame in self._state.frames]
        names.insert(position, name)
        candidate = Stack(images, names, self._state.metadata)
        frames = deepcopy(self._state.frames)
        frames.insert(position, candidate._state.frames[position])
        candidate._state.frames = frames
        candidate._state.dataset = candidate._state.dataset.assign_coords(frame=[frame.id for frame in frames])
        candidate._state.validate()
        self._state = candidate._state


class FrameHandle(Editing):
    """Route pixel and mask writes by frame identity, even after reordering."""

    def __init__(self, owner, identity):
        self.owner, self.identity = owner, identity

    def _position(self):
        for i, frame in enumerate(self.owner._state.frames):
            if frame.id == self.identity:
                return i
        raise ReferenceError("Frame was deleted")

    def _check_edit(self):
        super()._check_edit()
        self.owner._check_edit()

    @property
    def metadata(self):
        """Return this frame's durable typed metadata."""
        return MetadataView(self, lambda: self.owner._state.frames[self._position()].metadata)

    def to_numpy(self):
        """Return a detached frame cropped to its valid extent."""
        i = self._position()
        ds = self.owner._state.dataset
        height, width = int(ds.valid_height.values[i]), int(ds.valid_width.values[i])
        return masked_copy(ds.intensity.values[i, :height, :width], ds.excluded.values[i, :height, :width],
                           self.owner._state.frames[i].fill_value)

    def __getitem__(self, key):
        return self.to_numpy()[key]

    def __setitem__(self, key, value):
        with self.edit_numpy() as draft:
            draft[key] = value

    @contextmanager
    def edit_numpy(self):
        """Lock the parent as well as the handle until commit or rollback."""
        self._check_edit()
        draft = self.to_numpy()
        shape, dtype = draft.shape, draft.dtype
        self.owner._editing = True
        self._editing = True
        try:
            yield draft
            if draft.shape != shape or draft.dtype != dtype:
                raise ValueError("Editing cannot change shape or dtype")
            self._commit_array(draft)
        finally:
            self._editing = False
            self.owner._editing = False

    def _commit_array(self, draft):
        i = self._position()
        ds = self.owner._state.dataset
        height, width = int(ds.valid_height.values[i]), int(ds.valid_width.values[i])
        if draft.shape != (height, width) or draft.dtype != ds.intensity.dtype:
            raise ValueError("Frame edits must retain shape and dtype")
        # Stage both typed buffers before either assignment; the validated interior
        # rectangle cannot change extents, calibration, metadata or padding.
        values = np.array(draft.data, copy=True)
        excluded = np.ma.getmaskarray(draft).copy()
        ds.intensity.values[i, :height, :width] = values
        ds.excluded.values[i, :height, :width] = excluded


@pd.api.extensions.register_dataframe_accessor("stoner_prototype")
class TableAccessor:
    """Compare explicit schema resolution without caching metadata in an accessor."""

    def __init__(self, frame):
        self.frame = frame

    def column(self, selector, *, schema):
        """Resolve raw columns using an explicitly supplied, validated schema."""
        if list(self.frame.columns) != [col.id for col in schema]:
            raise ValueError("Supply schema matching the current frame")
        return self.frame.iloc[:, resolve(schema, selector)].copy(deep=True)


@xr.register_dataset_accessor("stoner_prototype")
class ImageAccessor:
    """Compare a selected mask-aware operation with native xarray reduction."""

    def __init__(self, dataset):
        self.dataset = dataset

    def mean(self):
        """Reduce with explicit exclusions, including padding, without attrs state."""
        values = self.dataset.intensity.values
        mask = self.dataset.excluded.values
        validate_mask(mask, values.shape)
        return np.ma.array(values, mask=mask, copy=False).mean()
