"""Expose shared scan headers through stable, writable frame metadata handles."""

from warnings import warn

import numpy as np
import xarray as xr

from ..core.base import TypeHintedDict
from ..core.storage import _copy_metadata
from .stack_owner import FrameOwner, _FrameMetadata
from .storage_bridge import bind


def write_exclusions(group, image):
    """Save raw-value exclusion and fill state alongside an existing format dataset."""
    package = image.export_storage()
    mask = package.dataset.excluded.values
    group.require_dataset("excluded", shape=mask.shape, dtype=bool)[...] = mask
    group.attrs["image_fill_value"] = package.fill_value


def read_exclusions(group, image):
    """Restore optional exclusion state while accepting older unmasked files."""
    package = image.export_storage()
    if "excluded" in group:
        mask = group["excluded"][...]
        if mask.dtype != np.dtype(bool) or mask.shape != image.shape:
            from ..core.exceptions import StonerLoadError
            raise StonerLoadError("Invalid stored image exclusion mask")
        package.dataset.excluded.data = mask
    package.fill_value = group.attrs.get("image_fill_value", package.fill_value)
    try:
        image.__dict__["_image_owner"].replace(package)
    except (ValueError, TypeError, OverflowError) as error:
        from ..core.exceptions import StonerLoadError
        raise StonerLoadError("Invalid stored image fill or exclusion state") from error
    return image


class _ScanMetadata(_FrameMetadata):
    """Read common headers as defaults and write only the frame's own record."""

    def __getitem__(self, key):
        local = self._owner._record.metadata
        return local[key] if key in local else self._owner._root._state.metadata[key]

    def __iter__(self):
        local = self._owner._record.metadata
        return iter(dict.fromkeys((*self._owner._root._state.metadata, *local)))

    def __len__(self):
        return sum(1 for _ in self)

    def type(self, key):
        """Return the effective hint without rebuilding the frame image package."""
        local = self._owner._record.metadata
        return local.type(key) if key in local else self._owner._root._state.metadata.type(key)

    def __setitem__(self, key, value):
        self._owner._check_writable()
        prepared = TypeHintedDict()
        prepared[key] = value
        target = self._owner._record.metadata
        for name, item in prepared.items():
            super(TypeHintedDict, target).__setitem__(name, item)
            target.types[name] = prepared.type(name)

    def __delitem__(self, key):
        self._owner._check_writable()
        del self._owner._record.metadata[key]


class ScanFrameOwner(FrameOwner):
    """Resolve scan-wide metadata beneath local frame overrides without copying storage."""

    @property
    def _state(self):
        state = super()._state
        metadata = _copy_metadata(self._root._state.metadata)
        for key, value in state.metadata.items():
            super(TypeHintedDict, metadata).__setitem__(key, value)
            metadata.types[key] = state.metadata.type(key)
        state.metadata = metadata
        return state

    @property
    def metadata(self):
        """Return the live common-header and frame-override interface."""
        self._root._index(self._identity)
        return _ScanMetadata(self)


class ScanMetadataMixin:
    """Store scan headers in the authoritative stack package metadata."""

    @property
    def _common_metadata(self):
        return self._stack_owner.metadata

    @_common_metadata.setter
    def _common_metadata(self, value):
        self._stack_owner._replace_metadata(value.copy())

    def _instantiate(self, index):
        image = super()._instantiate(index)
        bind(image, ScanFrameOwner(self._stack_owner, self.frame_ids[index]), image.image)
        return image

    def to_xarray(self, *, format="stack"):
        """Export canonical storage or named channels on one common image grid.

        Keyword Arguments:
            format (str):
                ``stack`` retains the canonical frame/y/x interchange. ``channels``
                exposes each channel as a Dataset variable with a separate Boolean
                exclusion variable. Both exports detach from the owner.

        Notes:
            The channel view is for native xarray analysis; use export_storage for
            lossless typed metadata. Irregular measured positions may be two-dimensional
            coordinates, without pretending that they are separable spatial axes.
        """
        if format == "stack":
            return super().to_xarray()
        if format != "channels":
            raise ValueError("format must be 'stack' or 'channels'")
        warn("The native channel Dataset omits typed metadata; use export_storage for lossless interchange",
             UserWarning, stacklevel=2)
        result = xr.Dataset()
        coordinates = None
        for index, name in enumerate(self.channels):
            image = self[index]
            package = image.export_storage()
            current = package.dataset.coords.to_dataset()
            if coordinates is not None and not coordinates.identical(current):
                raise ValueError("Channel export requires a common grid; regrid explicitly first")
            coordinates = current
            mask_name = f"{name}__excluded"
            if name in result or mask_name in result or name in current:
                raise ValueError("Channel names collide with coordinates or exclusion variables")
            result[name] = package.dataset.intensity
            result[mask_name] = package.dataset.excluded
            result[name].attrs["excluded"] = mask_name
            if "z-unit" in image.metadata:
                result[name].attrs["units"] = image.metadata["z-unit"]
        for axis, label in (("x", "PosX"), ("y", "PosY")):
            names = [name for name in self.channels if label in name]
            if len(names) == 1:
                result = result.assign_coords({f"physical_{axis}": result[names[0]]})
        return result.copy(deep=True)
