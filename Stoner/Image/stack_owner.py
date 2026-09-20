"""Own stack packages and resolve live image handles by stable frame identity."""

from operator import index as integer_index
from uuid import uuid4

import numpy as np
import xarray as xr

from ..core.storage_owner import _Metadata
from ..core.storage import _copy_metadata
from .storage import Frame, ImageStorage
from .storage_coordinates import _common_attrs
from .storage_owner import ImageOwner


class StackOwner(ImageOwner):
    """Maintain stack values, exclusions and frame records as one validated state.

    Notes:
        Structural edits are atomic. Existing frame handles survive insertion and
        reordering; deleted identities cannot be reused in this owner. Calibrated
        insertion and extraction preserve axes, spatial maps and scalar coordinates;
        incompatible attributes raise before publishing a structural change.
    """

    def __init__(self, package):
        if not isinstance(package, ImageStorage) or package.kind != "stack":
            raise TypeError("StackOwner requires a stack storage package")
        super().__init__(package)
        self._retired = set()
        self._frame_versions = {frame.id: 0 for frame in self._state.frames}

    def _index(self, selector):
        if isinstance(selector, str):
            for index, frame in enumerate(self._state.frames):
                if frame.id == selector:
                    return index
            raise ReferenceError("The selected frame is no longer in this stack")
        if isinstance(selector, (bool, np.bool_)):
            raise TypeError("Frame selection requires an integer position or frame ID")
        index = integer_index(selector)
        if index < 0:
            index += len(self._state.frames)
        if not 0 <= index < len(self._state.frames):
            raise IndexError("Frame position is out of range")
        return index

    def frame(self, selector):
        """Return a stable handle selected by integer position or canonical ID."""
        return FrameOwner(self, self._state.frames[self._index(selector)].id)

    def image(self, selector, image_type=None):
        """Return an ImageFile whose pixel, mask and metadata writes reach a frame."""
        from .core import ImageFile
        from .storage_bridge import bind

        handle = self.frame(selector)
        result = (image_type or ImageFile)()
        bind(result, handle, result.image)
        result.filename = handle.name
        return result

    def clone(self):
        """Return an independent stack owner preserving frame IDs and metadata."""
        return StackOwner(self.export_storage())

    def __deepcopy__(self, memo):
        """Copy committed state without transferring an active transaction lock."""
        result = self.clone()
        memo[id(self)] = result
        return result

    def replace(self, package):
        """Replace a complete stack, retaining live IDs and invalidating crop bounds."""
        self._check_writable()
        if not isinstance(package, ImageStorage) or package.kind != "stack":
            raise TypeError("Replacement requires a stack storage package")
        candidate = package.copy()
        identities = {frame.id for frame in candidate.frames}
        if identities & self._retired:
            raise ValueError("Deleted frame identities cannot be reused")
        removed = set(self._frame_versions) - identities
        versions = {identity: self._frame_versions.get(identity, -1) + 1 for identity in identities}
        self._state = candidate
        self._frame_versions = versions
        self._retired.update(removed)
        self._generation += 1

    def reorder(self, identities):
        """Atomically permute every frame ID, keeping saved handles live."""
        self._check_writable()
        identities = list(identities)
        current = [frame.id for frame in self._state.frames]
        if len(identities) != len(current) or set(identities) != set(current):
            raise ValueError("Reordering requires each existing frame ID exactly once")
        order = [current.index(identity) for identity in identities]
        state = self._state
        candidate = ImageStorage("stack", state.dataset.isel(frame=order).copy(deep=True),
                                 state.metadata, state.fill_value, [state.frames[i] for i in order])
        candidate.validate()
        self._state = candidate

    def delete(self, selector):
        """Remove one frame and its padding, permanently retiring its identity."""
        self._check_writable()
        position = self._index(selector)
        state = self._state
        identity = state.frames[position].id
        order = [i for i in range(len(state.frames)) if i != position]
        dataset = state.dataset.isel(frame=order).copy(deep=True)
        height = int(dataset.valid_height.max()) if order else 0
        width = int(dataset.valid_width.max()) if order else 0
        dataset = dataset.isel(y=slice(0, height), x=slice(0, width))
        candidate = ImageStorage("stack", dataset, state.metadata, state.fill_value,
                                 [state.frames[i] for i in order])
        candidate.validate()
        self._state = candidate
        self._retired.add(identity)
        del self._frame_versions[identity]

    def insert(self, position, package, *, name=""):
        """Insert an image with a new ID, preserving calibration and existing handles.

        Args:
            position (int):
                Insertion position, from zero to the number of frames inclusive.
            package (ImageStorage):
                Detached image state. Dtype promotion must preserve integer values.

        Keyword Arguments:
            name (str):
                Display name for the new frame; names need not be unique.

        Returns:
            FrameOwner:
                A handle for the newly inserted frame.
        """
        self._check_writable()
        if isinstance(position, (bool, np.bool_)):
            raise TypeError("Insertion position must be an integer")
        position = integer_index(position)
        state = self._state
        if not 0 <= position <= len(state.frames):
            raise IndexError("Insertion position is out of range")
        if not isinstance(package, ImageStorage) or package.kind != "image":
            raise TypeError("Insertion requires an image storage package")
        if self._insert_uniform(position, package, name):
            return self.frame(position)
        # from_images validates and detaches each input; exporting here would
        # create a second complete copy of every existing frame.
        images = [self.frame(frame.id)._state for frame in state.frames]
        images.insert(position, package)
        names = [frame.name for frame in state.frames]
        names.insert(position, name)
        candidate = ImageStorage.from_images(images, names=names, metadata=state.metadata)
        new_frame = candidate.frames[position]
        frames = list(state.frames)
        frames.insert(position, new_frame)
        candidate.frames = frames
        candidate.metadata = state.metadata
        if state.frames:
            candidate.fill_value = state.fill_value
        candidate.dataset = candidate.dataset.assign_coords(frame=[frame.id for frame in frames])
        for name in ("frame", "y", "x", "valid_height", "valid_width"):
            candidate.dataset[name].attrs = state.dataset[name].attrs.copy()
        if not state.frames:
            if state.dataset.attrs:
                _common_attrs([state.dataset, candidate.dataset], "empty stack Dataset")
            for name, variable in state.dataset.variables.items():
                if name in {"frame", "y", "x", "valid_height", "valid_width"}:
                    continue
                if name not in candidate.dataset:
                    raise ValueError("Insertion would discard an empty stack calibration coordinate")
                if variable.attrs:
                    _common_attrs([variable, candidate.dataset[name]], name)
        candidate.validate()
        self._state = candidate
        self._frame_versions[new_frame.id] = 0
        return self.frame(new_frame.id)

    def _insert_uniform(self, position, package, name):
        """Pack equal-sized uncalibrated frames without unpacking existing images."""
        state, incoming = self._state, package.dataset
        ds = state.dataset
        if (not state.frames or set(ds.coords) != {"frame", "y", "x"}
                or set(incoming.coords) != {"y", "x"}
                or incoming.intensity.shape != ds.intensity.shape[1:]
                or incoming.intensity.dtype != ds.intensity.dtype):
            return False
        # Native attributes and non-positional axes require the general mapper.
        if (ds.attrs or incoming.attrs or any(var.attrs for var in ds.variables.values())
                or any(var.attrs for var in incoming.variables.values())):
            return False
        height, width = incoming.intensity.shape
        if (not np.all(ds.valid_height.values == height) or not np.all(ds.valid_width.values == width)
                or not np.array_equal(incoming.y.values, np.arange(height))
                or not np.array_equal(incoming.x.values, np.arange(width))):
            return False
        package.validate()
        new_frame = Frame(str(uuid4()), name, _copy_metadata(package.metadata), package.fill_value)
        frames = list(state.frames)
        frames.insert(position, new_frame)
        variables = {
            key: (("frame", "y", "x"), np.insert(ds[key].values, position, incoming[key].values, axis=0))
            for key in ("intensity", "excluded")
        }
        variables.update(valid_height=("frame", np.full(len(frames), height, dtype=int)),
                         valid_width=("frame", np.full(len(frames), width, dtype=int)))
        dataset = xr.Dataset(variables, coords={"frame": [frame.id for frame in frames],
                                               "y": np.arange(height), "x": np.arange(width)})
        candidate = ImageStorage("stack", dataset, state.metadata, state.fill_value, frames)
        candidate.validate()
        self._state = candidate
        self._frame_versions[new_frame.id] = 0
        return True


class _FrameMetadata(_Metadata):
    """Commit imported dictionaries to the frame record, not a temporary image."""

    def __getattr__(self, name):
        if name in {"import_key", "import_all"}:
            def import_metadata(*args, **kwargs):
                self._owner._check_writable()
                draft = self.copy()
                result = getattr(draft, name)(*args, **kwargs)
                self._owner._replace_metadata(draft)
                return result
            return import_metadata
        return super().__getattr__(name)


class FrameOwner(ImageOwner):
    """Resolve a frame by ID for every access, sharing the stack's edit lock."""

    def __init__(self, parent, identity):
        self._root = parent
        self._identity = identity
        self._root._index(identity)

    @property
    def _record(self):
        return self._root._state.frames[self._root._index(self._identity)]

    @property
    def id(self):
        """Return the stable identity after checking that the frame still exists."""
        return self._record.id

    @property
    def name(self):
        """Return the current display name independently of identity."""
        return self._record.name

    @property
    def _generation(self):
        self._root._index(self._identity)
        return self._root._frame_versions[self._identity]

    @property
    def _state(self):
        position = self._root._index(self._identity)
        state = self._root._state
        ds = state.dataset
        height, width = int(ds.valid_height.values[position]), int(ds.valid_width.values[position])
        image = ds[["intensity", "excluded"]].isel(frame=position, y=slice(0, height), x=slice(0, width))
        image = image.drop_vars("frame")
        for axis in ("y", "x"):
            physical = f"index_{axis}" if f"index_{axis}" in image.coords else f"physical_{axis}"
            if physical in image.coords and image[physical].dims == (axis,):
                coordinate = image[physical]
                image = image.drop_vars(physical).assign_coords({axis: (axis, coordinate.values, coordinate.attrs)})
        frame = state.frames[position]
        return ImageStorage("image", image, frame.metadata, frame.fill_value)

    @property
    def _editing(self):
        return self._root._editing

    @_editing.setter
    def _editing(self, value):
        self._root._editing = value

    @property
    def metadata(self):
        """Return a guarded interface to this frame's typed metadata."""
        self._root._index(self._identity)
        return _FrameMetadata(self)

    def _replace_metadata(self, metadata):
        self._check_writable()
        self._record.metadata = metadata

    def _check_writable(self):
        self._root._index(self._identity)
        self._root._check_writable()

    def replace(self, package):
        """Require a detached clone or whole-stack replacement for structural edits."""
        self._check_writable()
        raise ValueError("Clone the frame or replace the parent stack before structural changes")

    def _commit_dataset(self, dataset):
        position = self._root._index(self._identity)
        state = self._root._state
        frame = state.frames[position]
        candidate = ImageStorage("image", dataset, frame.metadata, frame.fill_value)
        candidate.validate()
        height = int(state.dataset.valid_height.values[position])
        width = int(state.dataset.valid_width.values[position])
        if dataset.intensity.shape != (height, width) or dataset.intensity.dtype != state.dataset.intensity.dtype:
            raise ValueError("Frame edits must preserve shape and dtype")
        # Prepare detached buffers before publishing either field. With matching
        # shapes and dtypes these writes cannot require casting or broadcasting.
        buffers = {name: dataset[name].values.copy() for name in ("intensity", "excluded")}
        for name in ("intensity", "excluded"):
            self._root._state.dataset[name].values[position, :height, :width] = buffers[name]
