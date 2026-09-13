"""Own validated image packages and commit detached numerical edit drafts."""

from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import xarray as xr

from ..core.storage_owner import _Metadata
from .storage import ImageStorage


class _ImageMask:
    """Route mask writes to the current owner, never to a saved array view."""

    def __init__(self, owner):
        self._owner = owner

    def __array__(self, dtype=None, copy=None):
        return np.array(self._owner._state.dataset.excluded.values, dtype=dtype, copy=True)

    def __getitem__(self, key):
        return np.asarray(self)[key]

    def __setitem__(self, key, value):
        with self._owner.edit_numpy() as draft:
            draft.mask[key] = value


class ImageOwner:
    """Maintain one authoritative xarray package with guarded edit transactions.

    Notes:
        This is the internal ownership boundary, not a replacement ImageFile API.
        NumPy drafts include masks; xarray drafts preserve all coordinate and extent
        state. Neither may change dtype or shape. Nested mutable metadata values
        remain outside numerical transactions and are never overwritten by drafts.
    """

    def __init__(self, package):
        if not isinstance(package, ImageStorage):
            raise TypeError("ImageOwner requires an ImageStorage package")
        self._state = package.copy()
        self._editing = False
        self._generation = 0

    @property
    def metadata(self):
        """Return a durable, transaction-guarded typed metadata interface."""
        return _Metadata(self)

    @property
    def mask(self):
        """Return the owner-aware exclusion interface."""
        return _ImageMask(self)

    @property
    def shape(self):
        """Return positional shape without exporting any values."""
        return self._state.dataset.intensity.shape

    @property
    def dtype(self):
        """Return the homogeneous intensity dtype."""
        return self._state.dataset.intensity.dtype

    def _check_writable(self):
        """Reject public writes while another draft is active."""
        if self._editing:
            raise RuntimeError("An image edit is already active")

    def _replace_metadata(self, metadata):
        """Publish a prepared typed dictionary at the owning metadata boundary."""
        self._check_writable()
        self._state.metadata = metadata

    def export_storage(self):
        """Return an independently editable lossless package."""
        return self._state.copy()

    def to_numpy(self, *, masked=True, dtype=None):
        """Return detached values with optional exclusions or dtype conversion."""
        return self._state.to_numpy(masked=masked, dtype=dtype)

    def to_xarray(self):
        """Return detached native state with the documented metadata-loss warning."""
        return self._state.to_xarray()

    def __getitem__(self, key):
        result = self.to_numpy()[key]
        if isinstance(result, np.ndarray):
            result.setflags(write=False)
            result._mask.setflags(write=False)
        return result

    def __setitem__(self, key, value):
        with self.edit_numpy() as draft:
            draft[key] = value

    def replace(self, package):
        """Validate a complete structural replacement before publishing it."""
        self._check_writable()
        if not isinstance(package, ImageStorage):
            raise TypeError("Replacement requires an ImageStorage package")
        if package.kind != self._state.kind:
            raise ValueError("Replacement cannot change image/stack kind")
        self._state = package.copy()
        self._generation += 1

    def region(self, rows, columns):
        """Return a live rectangular handle using two non-empty unit-step slices.

        Args:
            rows (slice):
                Positional row bounds, following NumPy clipping rules.
            columns (slice):
                Positional column bounds, following NumPy clipping rules.

        Returns:
            ImageRegion:
                A shared pixel, exclusion and metadata handle. Numerical exports
                detach; replacing the parent package invalidates the handle.
        """
        return ImageRegion(self, rows, columns)

    def clone(self):
        """Return an independent owner, including typed metadata and calibration."""
        return ImageOwner(self.export_storage())

    def _commit_dataset(self, dataset):
        """Commit numerical state while retaining current owner metadata objects."""
        state = self._state
        candidate = ImageStorage(state.kind, deepcopy(dataset), state.metadata, state.fill_value,
                                 state.frames, state.version)
        candidate.validate()
        self._state = candidate

    @contextmanager
    def edit_numpy(self):
        """Edit a fixed-shape masked draft and atomically publish successful changes."""
        self._check_writable()
        draft = self.to_numpy()
        self._editing = True
        try:
            yield draft
            if draft.shape != self.shape or draft.dtype != self.dtype:
                raise ValueError("NumPy image edits cannot change shape or dtype")
            ds = self._state.dataset.copy(deep=True)
            ds["intensity"].data = np.ma.getdata(draft).copy()
            ds["excluded"].data = np.ma.getmaskarray(draft).copy()
            self._commit_dataset(ds)
        finally:
            self._editing = False

    @contextmanager
    def edit_xarray(self):
        """Edit intensity/exclusions while rejecting structural or coordinate edits."""
        self._check_writable()
        original = self._state.dataset
        draft = deepcopy(original)
        self._editing = True
        try:
            yield draft
            if not isinstance(draft, xr.Dataset) or set(draft.data_vars) != set(original.data_vars):
                raise ValueError("Xarray edits cannot change the variable set")
            if not draft.coords.to_dataset().identical(original.coords.to_dataset()):
                raise ValueError("Xarray edits cannot change coordinates or calibration")
            for name in original.data_vars:
                if draft[name].dims != original[name].dims or draft[name].dtype != original[name].dtype:
                    raise ValueError("Xarray edits cannot change dimensions or dtypes")
                if name not in ("intensity", "excluded") and not draft[name].identical(original[name]):
                    raise ValueError("Xarray edits cannot change valid extents")
            self._commit_dataset(draft)
        finally:
            self._editing = False


class _RegionMetadata(_Metadata):
    """Route whole-dictionary imports to the root rather than a region package."""

    def __getattr__(self, name):
        if name in {"import_key", "import_all"}:
            def import_metadata(*args, **kwargs):
                self._owner._check_writable()
                return getattr(self._owner._root.metadata, name)(*args, **kwargs)
            return import_metadata
        return super().__getattr__(name)


class ImageRegion(ImageOwner):
    """Resolve a bounded image rectangle against its current parent state.

    Notes:
        Regions share the parent's transaction lock and typed metadata. Nested
        regions resolve directly against the root owner. No writable ndarray is
        retained. Structural replacement requires a detached clone or the parent;
        replacing the parent invalidates existing regions even at the same shape.
    """

    def __init__(self, parent, rows, columns):
        if not isinstance(parent, ImageOwner) or parent._state.kind != "image":
            raise TypeError("Regions require a two-dimensional image owner")
        bounds = []
        for selection, size in zip((rows, columns), parent.shape):
            if not isinstance(selection, slice):
                raise TypeError("Region bounds must be slices")
            start, stop, step = selection.indices(size)
            if step != 1 or stop <= start:
                raise ValueError("Regions require non-empty unit-step slices")
            bounds.append(slice(start, stop))
        if isinstance(parent, ImageRegion):
            bounds = [slice(outer.start + inner.start, outer.start + inner.stop)
                      for outer, inner in zip(parent._bounds, bounds)]
            parent = parent._root
        self._root = parent
        self._bounds = tuple(bounds)
        self._generation = parent._generation

    def _check_live(self):
        if self._generation != self._root._generation:
            raise ReferenceError("The region's parent image has been replaced")

    @property
    def _state(self):
        self._check_live()
        state = self._root._state
        dataset = state.dataset.isel(y=self._bounds[0], x=self._bounds[1])
        return ImageStorage("image", dataset, state.metadata, state.fill_value, version=state.version)

    @property
    def _editing(self):
        return self._root._editing

    @_editing.setter
    def _editing(self, value):
        self._root._editing = value

    @property
    def metadata(self):
        """Return shared, guarded parent metadata after checking handle validity."""
        self._check_live()
        return _RegionMetadata(self)

    def _check_writable(self):
        self._check_live()
        self._root._check_writable()

    def _replace_metadata(self, metadata):
        self._check_writable()
        self._root._replace_metadata(metadata)

    def replace(self, package):
        """Reject structural replacement through a shared rectangle."""
        self._check_writable()
        raise ValueError("Replace the parent image or clone the region before structural changes")

    def _commit_dataset(self, dataset):
        self._check_live()
        candidate = self._root._state.dataset.copy(deep=True)
        for name in ("intensity", "excluded"):
            candidate[name].values[self._bounds] = dataset[name].values
        self._root._commit_dataset(candidate)
