"""Connect the public ImageStack folder interface to authoritative stack storage."""

from copy import deepcopy
from operator import index as integer_index
import re

import numpy as np

from ..core.base import RegexpDict
from .core import ImageFile
from .numerical import numerical_image
from .storage import ImageStorage, _fill
from .stack_owner import StackOwner


class OwnedStackMixin:
    """Provide stack construction, interchange and folder edits through stable IDs."""

    def __init__(self, *args, **kwargs):
        from ..folders.core import BaseFolder

        self._stack_owner = StackOwner(ImageStorage.from_images([]))
        self._public_attrs_store = {}
        source = args[0] if args else None
        if isinstance(source, (BaseFolder, list, np.ndarray)):
            super().__init__(*args[1:], **kwargs)
            if isinstance(source, OwnedStackMixin):
                self._stack_owner = source._stack_owner.clone()
                self._public_attrs_store = deepcopy(source._public_attrs_store)
            elif isinstance(source, np.ndarray) and source.ndim == 3:
                self.imarray = source
            else:
                for item in source:
                    self.append(item)
        else:
            super().__init__(*args, **kwargs)

    @classmethod
    def from_storage(cls, package):
        """Construct an independent stack from a validated lossless package."""
        result = cls()
        result._stack_owner = StackOwner(package)
        return result

    @classmethod
    def from_xarray(cls, dataset, **kwargs):
        """Import a native stack Dataset with optional typed metadata records."""
        return cls.from_storage(ImageStorage.from_xarray(dataset, **kwargs))

    def export_storage(self):
        """Return a detached lossless stack package."""
        return self._stack_owner.export_storage()

    def to_xarray(self):
        """Return detached native state with the metadata-loss warning."""
        return self._stack_owner.to_xarray()

    def to_numpy(self, **kwargs):
        """Return independently writable values, including exclusions by default."""
        return self._stack_owner.to_numpy(**kwargs)

    def edit_numpy(self):
        """Edit a fixed-shape masked draft under the shared stack lock."""
        return self._stack_owner.edit_numpy()

    def edit_xarray(self):
        """Edit intensity and exclusions without altering coordinates or extents."""
        return self._stack_owner.edit_xarray()

    @property
    def frame_ids(self):
        """Return stable identities in current positional order."""
        return tuple(frame.id for frame in self._stack_owner._state.frames)

    @property
    def imarray(self):
        """Return a detached read-only masked array in frame/y/x order."""
        result = self.to_numpy().view(np.ma.MaskedArray)
        result.flags.writeable = False
        result._mask.flags.writeable = False
        return result

    @imarray.setter
    def imarray(self, value):
        owner = self._stack_owner
        owner._check_writable()
        value = np.ma.asarray(value)
        if value.ndim != 3:
            raise ValueError("Stack assignment requires frame/y/x values")
        if value.shape == owner.shape:
            package = owner.export_storage()
            package.dataset.intensity.data = np.ma.getdata(value).copy()
            package.dataset.excluded.data = np.ma.getmaskarray(value).copy()
            for frame in package.frames:
                frame.fill_value = _fill(frame.fill_value, value.dtype)
            package.fill_value = _fill(package.fill_value, value.dtype)
            owner.replace(package)
        else:
            if (owner._state.dataset.attrs or set(owner._state.dataset.coords) != {"frame", "y", "x"}
                    or any(variable.attrs for variable in owner._state.dataset.variables.values())):
                raise ValueError("Use a calibrated storage package for structural stack replacement")
            names = [f"Untitled-{i}" for i in range(len(value))]
            owner.replace(ImageStorage.from_images(value, names=names, metadata=owner.metadata.copy()))

    @property
    def shape(self):
        """Return frame/y/x dimensions without exporting data."""
        return self._stack_owner.shape

    @property
    def max_size(self):
        """Return the maximum valid height and width."""
        return self.shape[1:]

    def __names__(self):
        return [frame.name for frame in self._stack_owner._state.frames]

    def __lookup__(self, name):
        if isinstance(name, (int, np.integer)) and not isinstance(name, (bool, np.bool_)):
            try:
                return self._stack_owner._index(name)
            except IndexError as error:
                raise KeyError(name) from error
        if name in self.frame_ids:
            return self.frame_ids.index(name)
        names = self.__names__()
        if name in names:
            return names.index(name)
        lookup = RegexpDict((name, i) for i, name in enumerate(names))
        return lookup[name]

    def _instantiate(self, index):
        image = self._stack_owner.image(index, image_type=self.type)
        for name, value in self._public_attrs_store.get(self.frame_ids[index], {}).items():
            setattr(image, name, value)
        return image

    def __getter__(self, name, instantiate=True):
        index = self.__lookup__(name)
        return self._instantiate(index)

    def fetch(self, *args, **kwargs):
        """Return this eager stack; every frame is already held by its owner."""
        return self

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.__getter__(self._stack_owner._index(key))
        if isinstance(key, tuple):
            return self.imarray[key]
        if isinstance(key, slice):
            result = self.clone
            keep = set(self.frame_ids[key])
            for identity in tuple(result.frame_ids):
                if identity not in keep:
                    result._stack_owner.delete(identity)
            result.reorder(self.frame_ids[key])
            return result
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            with self.edit_numpy() as draft:
                draft[key] = value
        elif isinstance(key, (int, np.integer)):
            self.__setter__(self._stack_owner._index(key), value)
        else:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        if isinstance(key, (int, np.integer)):
            self.__deleter__(self._stack_owner._index(key))
        else:
            super().__delitem__(key)

    def __floordiv__(self, other):
        """Return the framewise XMCD ratio with matching geometry and dtypes."""
        if not isinstance(other, OwnedStackMixin):
            return NotImplemented
        if self.shape != other.shape or self._stack_owner.dtype != other._stack_owner.dtype:
            raise ValueError("XMCD stacks require matching shapes and dtypes")
        result = self.clone
        for index, (left, right) in enumerate(zip(self, other)):
            result[index] = left // right
        return result

    def _prepare(self, value):
        image = value if isinstance(value, ImageFile) else ImageFile(value)
        package = image.export_storage()
        attrs = {name: deepcopy(getattr(image, name, None)) for name in image._public_attrs
                 if name not in {"data", "image", "metadata", "_fromstack", "filename"}}
        return package, attrs

    def __setter__(self, name, value, force_insert=False):
        if isinstance(value, str):
            from pathlib import Path
            filename = Path(value)
            if not filename.is_file():
                filename = Path(self.directory) / filename
            value = self.type(filename, **self.extra_args)
        try:
            if force_insert or name is None:
                raise KeyError(name)
            index = self.__lookup__(name)
        except KeyError:
            return self.__inserter__(len(self), name or self.make_name(value), value)
        package, attrs = self._prepare(value)
        owner = self._stack_owner
        owner._check_writable()
        old = owner._state
        images = [owner.frame(i).export_storage() for i in range(len(self))]
        images[index] = package
        candidate = ImageStorage.from_images(images, names=self.__names__(), metadata=owner.metadata.copy())
        for frame, previous in zip(candidate.frames, old.frames):
            frame.id = previous.id
        candidate.dataset = candidate.dataset.assign_coords(frame=list(self.frame_ids))
        for variable in ("frame", "y", "x", "valid_height", "valid_width"):
            candidate.dataset[variable].attrs = deepcopy(old.dataset[variable].attrs)
        candidate.fill_value = _fill(old.fill_value, candidate.dataset.intensity.dtype)
        structural = (old.dataset.intensity.dtype != candidate.dataset.intensity.dtype
                      or not old.dataset.drop_vars(["intensity", "excluded"]).identical(
                          candidate.dataset.drop_vars(["intensity", "excluded"])))
        if structural:
            owner.replace(candidate)
        else:
            candidate.validate()
            owner._commit_dataset(candidate.dataset)
            owner._state.frames = candidate.frames
        self._public_attrs_store[self.frame_ids[index]] = attrs

    def __inserter__(self, index, name, value):
        package, attrs = self._prepare(value)
        frame = self._stack_owner.insert(index, package, name=str(name))
        self._public_attrs_store[frame.id] = attrs

    def insert(self, index, value):
        """Insert at a clipped positional index, retaining existing frame identities."""
        index = integer_index(index)
        index = max(0, len(self) + index) if index < 0 else min(index, len(self))
        name = self.make_name(value)
        original, suffix = name, 1
        while name in self.__names__():
            name = f"{original}({suffix})"
            suffix += 1
        self.__inserter__(index, name, value)

    def __deleter__(self, key):
        index = self.__lookup__(key)
        identity = self.frame_ids[index]
        self._stack_owner.delete(identity)
        self._public_attrs_store.pop(identity, None)

    def __clear__(self):
        self._stack_owner.replace(ImageStorage.from_images([]))
        self._public_attrs_store.clear()

    def reorder(self, identities):
        """Reorder every frame by stable ID, retaining saved item handles."""
        self._stack_owner.reorder(identities)
        return self

    def sort(self, key=None, reverse=False, recurse=True):
        """Sort names or metadata through an identity-preserving permutation."""
        if recurse:
            for group in self.groups.values():
                group.sort(key=key, reverse=reverse, recurse=True)
        if key is None:
            get_key = lambda i: self.__names__()[i]
        elif isinstance(key, str):
            get_key = lambda i: self[i].metadata.get(key)
        elif isinstance(key, re.Pattern):
            get_key = lambda i: key.match(self.__names__()[i]).groups()
        else:
            get_key = lambda i: key(self[i])
        order = sorted(range(len(self)), key=get_key, reverse=reverse)
        return self.reorder([self.frame_ids[i] for i in order])

    def convert(self, dtype, force_copy=False, uniform=False, normalise=True):
        """Convert all valid intensities, retaining calibration, masks and identities."""
        from .imagefuncs import convert
        package = self.export_storage()
        values = convert(package.to_numpy(), dtype, force_copy=force_copy, uniform=uniform, normalise=normalise)
        package.dataset.intensity.data = np.ma.getdata(values).copy()
        for i, frame in enumerate(package.frames):
            h, w = int(package.dataset.valid_height[i]), int(package.dataset.valid_width[i])
            package.dataset.intensity.values[i, h:, :] = 0
            package.dataset.intensity.values[i, :, w:] = 0
            try:
                frame.fill_value = _fill(frame.fill_value, values.dtype)
            except (ValueError, TypeError, OverflowError):
                default = np.ma.array(np.empty(0, dtype=values.dtype)).fill_value
                frame.fill_value = np.ma.array(np.empty(0, dtype=values.dtype), fill_value=default).fill_value
        try:
            package.fill_value = _fill(package.fill_value, values.dtype)
        except (ValueError, TypeError, OverflowError):
            default = np.ma.array(np.empty(0, dtype=values.dtype)).fill_value
            package.fill_value = np.ma.array(np.empty(0, dtype=values.dtype), fill_value=default).fill_value
        self._stack_owner.replace(package)
        return self

    def subtract(self, background):
        """Subtract a mean-scaled background using an atomic masked stack draft."""
        bg = self[background] if isinstance(background, (int, str)) else background
        bg = bg.to_numpy() if isinstance(bg, ImageFile) else np.ma.asarray(bg)
        included = ~np.ma.getmaskarray(bg)
        with self.edit_numpy() as draft:
            for image in draft:
                image -= bg * image[included].mean() / bg[included].mean()
        return self

    def _reduction(self, values, box, metadata):
        """Wrap a masked reduction with the selected frame metadata."""
        result = ImageFile(values)
        state = self._stack_owner._state
        package = result.export_storage()
        package.dataset.attrs = deepcopy(state.dataset.attrs)
        for variable in ("intensity", "excluded"):
            package.dataset[variable].attrs = deepcopy(state.dataset[variable].attrs)
        result = ImageFile.from_storage(package)
        if set(state.dataset.coords) != {"frame", "y", "x"}:
            first = self[0].export_storage()
            coordinates = first.dataset.coords.to_dataset()
            if any(not coordinates.identical(self[i].export_storage().dataset.coords.to_dataset())
                   for i in range(1, len(self))):
                raise ValueError("Explicitly reconcile frame calibration before reducing the stack")
            first.dataset.intensity.data = np.ma.getdata(values).copy()
            first.dataset.excluded.data = np.ma.getmaskarray(values).copy()
            first.fill_value = np.ma.array(values).fill_value
            result = ImageFile.from_storage(first)
        if metadata == "first":
            result.metadata = self[0].metadata.copy()
        elif metadata == "common":
            result.metadata = self.metadata.common_metadata
        else:
            result.metadata = {}
        if box is not False:
            result.crop(box)
        return result

    def average(self, weights=None, _box=False, _metadata="first"):
        """Average valid pixels, excluding both user masks and ragged padding."""
        values = np.ma.average(self.to_numpy(), axis=0, weights=weights)
        return self._reduction(values, _box, _metadata)

    def stddev(self, weights=None, _box=False, _metadata="first"):
        """Return the population deviation of included pixels in each position."""
        data = self.to_numpy()
        mean = np.ma.average(data, axis=0, weights=weights)
        values = np.ma.sqrt(np.ma.average((data - mean) ** 2, axis=0, weights=weights))
        return self._reduction(values, _box, _metadata)

    def stderr(self, weights=None, _box=False, _metadata="first"):
        """Return deviation divided by the square root of each valid sample count."""
        deviation = self.stddev(weights=weights, _metadata=_metadata).to_numpy()
        values = deviation / np.sqrt(self.to_numpy().count(axis=0))
        return self._reduction(values, _box, _metadata)
