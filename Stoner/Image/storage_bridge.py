"""Connect ImageFile's public interface to one authoritative image owner."""

from contextlib import contextmanager
from copy import deepcopy
from functools import wraps

import numpy as np
import xarray as xr

from ..core.storage import _copy_metadata
from ..core.base import TypeHintedDict
from ..core.selection import selected_image_values
from .storage import ImageStorage
from .storage_owner import ImageOwner, ImageRegion
from .stack_owner import FrameOwner


def owner_of(image):
    """Find the owner without invoking dynamically generated image attributes."""
    owner = image.__dict__.get("_image_owner")
    if owner is not None:
        owner.validate()
    return owner


def snapshot(image, writable=False):
    """Build a detached NumPy working array, including masks and typed metadata."""
    # Preserve lazy image adapters and avoid owner/accessor import cycles.
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from .numerical import numerical_image

    owner = owner_of(image)
    result = numerical_image(owner.to_numpy())
    metadata = owner._state.metadata
    # Plotting handles belong to the live display. Deep-copying a Figure can open
    # another pyplot window merely while preparing a numerical working array.
    memo = {id(metadata[key]): metadata[key] for key in ("ax", "fig") if key in metadata}
    result.metadata = _copy_metadata(metadata, memo)
    for name, value in image.__dict__.get("_image_attrs", {}).items():
        setattr(result, name, deepcopy(value))
    result.filename = image.filename
    if hasattr(type(image), "tesseractable"):
        result.tesseractable = image.tesseractable
    if not writable:
        result.setflags(write=False)
        result._mask.setflags(write=False)
    return result


def bind(image, owner, array=None):
    """Install a sole owner and retain only non-numerical presentation attributes."""
    # Preserve lazy image adapters and avoid owner/accessor import cycles.
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from .roi import invalidate_roi

    invalidate_roi(image)
    if array is not None:
        image.__dict__["_image_attrs"] = {
            name: deepcopy(getattr(array, name))
            for name in ("filename", "_title", "_mask_color", "_mask_alpha", "debug")
            if hasattr(array, name)
        }
    image.__dict__["_image_owner"] = owner
    image.__dict__.pop("_image", None)
    image.__dict__.pop("_metadata", None)
    for name in ("_image", "_metadata"):
        image._public_attrs.pop(name, None)


def replace_image(image, values):
    """Publish a masked replacement, preserving existing calibration when possible."""
    owner = owner_of(image)
    if isinstance(values, xr.Dataset):
        if owner is None:
            package = ImageStorage.from_xarray(values, metadata=image.metadata)
            if package.kind != "image":
                raise ValueError("ImageFile assignment requires a y/x image Dataset")
            bind(image, ImageOwner(package))
        else:
            owner.replace_xarray(values)
        return
    if hasattr(values, "__dict__") and owner_of(values) is not None:
        values = snapshot(values)
    metadata = _copy_metadata(image.metadata) if isinstance(image.metadata, TypeHintedDict) else image.metadata.copy()
    incoming = getattr(values, "metadata", {})
    for key, value in incoming.items():
        if isinstance(incoming, TypeHintedDict):
            metadata._set_prepared(key, deepcopy(value), incoming.type(key))
        else:
            metadata[key] = deepcopy(value)
    package = ImageStorage.from_numpy(values, metadata=metadata)
    if owner is None:
        bind(image, ImageOwner(package), image.__dict__.get("_image"))
        return
    owner._check_writable()
    old = owner.export_storage()
    if package.dataset.intensity.shape == owner.shape:
        package.dataset = old.dataset.copy(deep=True)
        package.dataset["intensity"].data = np.ma.getdata(values).copy()
        package.dataset["excluded"].data = np.ma.getmaskarray(values).copy()
    elif calibrated(owner):
        raise ValueError("Replace calibrated geometry through an explicit ImageStorage package")
    if isinstance(owner, (ImageRegion, FrameOwner)):
        if package.dataset.intensity.shape != owner.shape or package.dataset.intensity.dtype != owner.dtype:
            raise ValueError("Clone a shared region or frame before changing its shape or dtype")
        with owner.edit_numpy() as draft:
            draft[:] = values
        image.metadata = metadata
    else:
        owner.replace(package)
        if package.dataset.intensity.shape != old.dataset.intensity.shape:
            # Preserve lazy image adapters and avoid owner/accessor import cycles.
            # pylint: disable-next=import-outside-toplevel,cyclic-import
            from .roi import invalidate_roi

            invalidate_roi(image)


def calibrated(owner):
    """Identify explicit coordinates that a numerical transform must preserve."""
    ds = owner._state.dataset
    return set(ds.coords) != {"y", "x"} or any(
        ds[name].attrs or not np.array_equal(ds[name], np.arange(ds.sizes[name])) for name in ("y", "x")
    )


def commit_result(image, values):
    """Publish numerical results without invalidating live regions at fixed geometry."""
    owner = owner_of(image)
    if values.shape != owner.shape or values.dtype != owner.dtype:
        replace_image(image, values)
        return
    with owner.edit_numpy() as draft:
        draft[:] = values
    if hasattr(values, "metadata"):
        image.metadata = values.metadata


def call_method(image, function, args, kwargs, *, setter=False):
    """Run a numerical adapter on a detached array and explicitly publish its result."""
    # Preserve lazy image adapters and avoid owner/accessor import cycles.
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from .storage_interpolation import GEOMETRY, require_alignment

    owner = owner_of(image)
    name = function.__name__
    kwargs = dict(kwargs)
    if "where" in kwargs:
        raise TypeError(f"{name} does not support operation selection; crop explicitly or use a regional reduction")
    reference_grid = kwargs.pop("_reference_grid", None) if name in {"align", "correct_drift"} else None
    force = kwargs.pop("_", bool(getattr(function, "changes_size", False)))
    box = kwargs.pop("_box", False)
    if name == "crop":
        return _call_crop(image, owner, args, kwargs, force)
    if name in {"CW", "CCW", "flip_h", "flip_v", "transpose", "T", "swapaxes", "rot90"}:
        return _call_permutation(image, owner, function, args, kwargs, force, box)
    for operand in (*args, *kwargs.values()):
        if name not in {"align", "correct_drift"} and hasattr(operand, "__dict__"):
            require_alignment(image, operand)
    if calibrated(owner) and name in GEOMETRY:
        return _call_interpolation(image, owner, function, args, kwargs, force, box)
    if calibrated(owner) and name in {"align", "correct_drift"}:
        return _call_alignment(image, owner, function, args, kwargs, force, box, reference_grid)
    return _call_numerical(image, owner, function, args, kwargs, force, box, setter)


def _call_crop(image, owner, args, kwargs, force):
    """Apply an explicit crop and bind its independent or shared region."""
    owner._check_writable()
    if (not args and "box" not in kwargs) or args == (None,):
        # Preserve lazy image adapters and avoid owner/accessor import cycles.
        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from .widgets import RegionSelect

        args = (RegionSelect()(image),)
    bounds = snapshot(image)._box(*args, **kwargs)
    region = owner.region(*bounds)
    target = image.clone if force is None else image
    bind(target, region.clone() if kwargs.get("copy", False) else region)
    return target


def _call_permutation(image, owner, function, args, kwargs, force, box):
    """Permute all scientific state before atomically replacing the selected owner."""
    name = function.__name__
    # Preserve lazy image adapters and avoid owner/accessor import cycles.
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from .storage_transforms import permute

    if box is not False:
        raise ValueError("Crop explicitly before applying an exact image permutation")
    package = permute(owner.export_storage(), name, args, kwargs)
    target = image.clone if force is None or getattr(function, "clones", False) else image
    if package.dataset.intensity.shape != owner.shape and not force:
        target = image.clone
    target_owner = owner_of(target)
    target_owner._check_writable()
    if isinstance(target_owner, (ImageRegion, FrameOwner)):
        raise ValueError("Clone a shared region or frame before permuting its geometry")
    target_owner.replace(package)
    # Preserve lazy image adapters and avoid owner/accessor import cycles.
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from .roi import invalidate_roi

    invalidate_roi(target)
    return target


def _call_interpolation(image, owner, function, args, kwargs, force, box):
    """Resample calibrated geometry and publish a validated independent package."""
    # Preserve lazy image adapters and avoid owner/accessor import cycles.
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from .storage_interpolation import interpolate

    if box is not False:
        raise ValueError("Crop explicitly before interpolating calibrated geometry")
    target = image.clone if force is None or getattr(function, "clones", False) else image
    if isinstance(owner_of(target), (ImageRegion, FrameOwner)):
        raise ValueError("Clone a shared region or frame before interpolating its geometry")
    owner_of(target)._check_writable()
    package = interpolate(owner.export_storage(), function, args, kwargs, snapshot(image, writable=True))
    if package.dataset.intensity.shape != owner.shape and not force:
        target = image.clone
    owner_of(target).replace(package)
    # Preserve lazy image adapters and avoid owner/accessor import cycles.
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from .roi import invalidate_roi

    invalidate_roi(target)
    return target


def _call_alignment(image, owner, function, args, kwargs, force, box, reference_grid):
    """Apply registration to pixels and publish the reference calibration grid."""
    name = function.__name__
    # Preserve lazy image adapters and avoid owner/accessor import cycles.
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from scipy.ndimage import shift
    # Preserve lazy image adapters and avoid owner/accessor import cycles.
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from .storage_interpolation import interpolate

    if box is not False:
        raise ValueError("Use the alignment box argument to select registration pixels")
    target = image.clone if force is None else image
    target_owner = owner_of(target)
    if isinstance(target_owner, (ImageRegion, FrameOwner)):
        raise ValueError("Clone a shared region or frame before aligning its geometry")
    target_owner._check_writable()
    working = snapshot(image, writable=True)
    operands = [snapshot(arg) if hasattr(arg, "__dict__") and owner_of(arg) is not None else arg for arg in args]
    options = {
        key: snapshot(arg) if hasattr(arg, "__dict__") and owner_of(arg) is not None else arg
        for key, arg in kwargs.items()
    }
    numerical = function(working, *operands, **options)
    if name == "correct_drift" and not kwargs.get("do_shift", True):
        commit_result(target, numerical)
        return target
    package = interpolate(owner.export_storage(), shift, (tuple(numerical.metadata["tvec"]),), {}, working)
    package.dataset.intensity.data = np.ma.getdata(numerical).copy()
    package.metadata = numerical.metadata.copy()
    grid = _alignment_grid(owner, args, kwargs, reference_grid, numerical.shape)
    package.dataset = package.dataset.drop_vars(list(package.dataset.coords)).assign_coords(
        grid.coords.to_dataset().copy(deep=True).coords
    )
    target_owner.replace(package)
    # Preserve lazy image adapters and avoid owner/accessor import cycles.
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from .roi import invalidate_roi

    invalidate_roi(target)
    return target


def _call_numerical(image, owner, function, args, kwargs, force, box, setter):
    """Run the compatibility numerical adapter and publish only its supported result."""
    # Preserve lazy image adapters and avoid owner/accessor import cycles.
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from .storage_interpolation import GEOMETRY

    name = function.__name__
    working = snapshot(image, writable=True)
    if box is not False:
        # Preserve lazy image adapters and avoid owner/accessor import cycles.
        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from .numerical import numerical_image

        working = numerical_image(working[working._box(box)], metadata=working.metadata)
    args = [snapshot(arg) if hasattr(arg, "__dict__") and owner_of(arg) is not None else arg for arg in args]
    kwargs = {
        key: snapshot(arg) if hasattr(arg, "__dict__") and owner_of(arg) is not None else arg
        for key, arg in kwargs.items()
    }
    result = function(working, *args, **kwargs)
    if setter:
        if name == "title":
            image.__dict__["_image_attrs"]["_title"] = working._title
            return result
        replace_image(image, working)
        image.__dict__["_image_attrs"].update(
            {key: deepcopy(getattr(working, key)) for key in image.__dict__.get("_image_attrs", {})}
        )
        return result
    if result is None:
        commit_result(image, working)
        return None
    if getattr(function, "keep_class", False):
        return result
    if not isinstance(result, np.ndarray) or result.ndim != 2:
        if name in {"ocr_metadata", "reduce_metadata"}:
            image.metadata = working.metadata
        if name == "imshow":
            for key in ("ax", "fig"):
                if key in working.metadata:
                    image.metadata[key] = working.metadata[key]
        return result
    target = image.clone if force is None or getattr(function, "clones", False) else image
    if result.shape != owner.shape and not force:
        target = image.clone
    commit_result(target, result)
    if name in GEOMETRY or name in {"align", "correct_drift"}:
        # Preserve lazy image adapters and avoid owner/accessor import cycles.
        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from .roi import invalidate_roi

        invalidate_roi(target)
    return target


class OwnerDraw:
    """Run existing drawing operations inside a numerical transaction."""

    def __init__(self, image, mask=False):
        self._image, self._mask = image, mask

    def __dir__(self):
        # Preserve lazy image adapters and avoid owner/accessor import cycles.
        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from .attrs import DrawProxy

        return dir(DrawProxy(snapshot(self._image, writable=True), self._image))

    def __getattr__(self, name):
        # Preserve lazy image adapters and avoid owner/accessor import cycles.
        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from .attrs import DrawProxy

        def draw(*args, **kwargs):
            with owner_of(self._image).edit_numpy() as draft:
                proxy = DrawProxy(draft.mask if self._mask else draft.view(np.ma.MaskedArray), self._image)
                result = getattr(proxy, name)(*args, **kwargs)
            return result

        return draw


def _install_state(cls):
    """Install construction, attribute and metadata owner boundaries."""
    _install_construction(cls)
    _install_attributes(cls)
    _install_metadata(cls)


def _install_construction(cls):
    """Preserve decorated construction before adopting populated image storage."""
    construct = cls.__init__
    allocate = cls.__new__

    def new(class_, *args, **kwargs):
        del kwargs  # Allocation forwards positional arguments; initialisation consumes keywords.
        return allocate(class_, *args)

    @wraps(construct)
    def initialise(self, *args, **kwargs):
        asfloat = kwargs.pop("asfloat", False)
        self.__dict__["_image_initialising"] = True
        try:
            construct(self, *args, **kwargs)
        finally:
            self.__dict__.pop("_image_initialising", None)
        array = self.__dict__["_image"]
        if array.size:
            source = owner_of(args[0]) if args and hasattr(args[0], "__dict__") else None
            package = (
                source.export_storage()
                if source is not None
                else ImageStorage.from_numpy(array, metadata=self.metadata)
            )
            bind(self, ImageOwner(package), array)
            if asfloat:
                self.asfloat()
        elif args and not array.size:
            raise ValueError("Image storage requires non-empty image axes")

    cls.__init__ = initialise
    cls.__new__ = staticmethod(new)


def _install_attributes(cls):
    """Route protected image writes through the current scientific owner."""
    set_attribute = cls.__setattr__

    def getattribute(self, name):
        if name == "_image" and owner_of(self) is not None:
            return snapshot(self)
        return object.__getattribute__(self, name)

    def setattr_(self, name, value):
        if owner_of(self) is None:
            if (
                name == "_image"
                and not self.__dict__.get("_image_initialising", False)
                and type(self).image is cls.image
            ):
                return replace_image(self, value)
            return set_attribute(self, name, value)
        if name == "_image":
            return replace_image(self, value)
        if name == "_metadata":
            self.metadata = value
            return None
        if name not in dir(type(self)) and not name.startswith("_"):
            self._public_attrs[name] = type(value)
        return object.__setattr__(self, name, value)

    cls.__getattribute__ = getattribute
    cls.__setattr__ = setattr_


def _install_metadata(cls):
    """Bind live typed metadata while retaining empty-loader placeholders."""
    empty_metadata = cls.metadata

    def get_metadata(self):
        owner = owner_of(self)
        return owner.metadata if owner is not None else empty_metadata.fget(self)

    def set_metadata(self, value):
        owner = owner_of(self)
        if owner is None:
            return empty_metadata.fset(self, value)
        owner._check_writable()
        if not hasattr(value, "items"):
            raise TypeError("Image metadata must be a mapping")
        copied = value
        copied = _copy_metadata(copied) if isinstance(copied, TypeHintedDict) else TypeHintedDict(copied)
        return owner._replace_metadata(copied)

    cls.metadata = property(get_metadata, set_metadata)


def _install_indexing(cls):
    """Install detached cloning and owner-aware item operations."""
    for method in (_clone, _getitem, _setitem, _delitem):
        method.__name__ = method.__name__.removeprefix("_")
    cls.clone = property(_clone)
    cls.__getitem__, cls.__setitem__ = _getitem, _setitem
    cls.__delitem__ = _delitem


def _install_conversion(cls):
    """Install lossless interchange, numerical exports and regional reductions."""
    for method in (_from_xarray, _to_numpy, _array, _no_array_pointer, _export_storage,
                   _to_xarray, _edit_numpy, _edit_xarray, _mean):
        method.__name__ = method.__name__.removeprefix("_")

    def from_storage(class_, package):
        """Construct an independent image from a lossless image package."""
        if class_.image is not cls.image:
            raise TypeError("Specialised image descriptors require their own storage adapter")
        if not isinstance(package, ImageStorage) or package.kind != "image":
            raise TypeError("ImageFile requires an image storage package")
        result = class_()
        bind(result, ImageOwner(package), result.__dict__["_image"])
        return result

    cls.from_storage, cls.from_xarray = classmethod(from_storage), classmethod(_from_xarray)
    cls.to_numpy, cls.to_xarray, cls.export_storage = _to_numpy, _to_xarray, _export_storage
    cls.__array__ = _array
    cls.__array_interface__ = property(_no_array_pointer)
    cls.__array_struct__ = property(_no_array_pointer)
    cls.edit_numpy, cls.edit_xarray = _edit_numpy, _edit_xarray
    cls.mean = _mean


def _aligned_operation(operation):
    """Bind one arithmetic operation without retaining a loop variable."""
    @wraps(operation)
    def aligned(self, other, _operation=operation):
        # Preserve lazy image adapters and avoid owner/accessor import cycles.
        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from .storage_interpolation import require_alignment

        require_alignment(self, other)
        return _operation(self, other)

    return aligned


def _install_operators(cls):
    """Require calibrated alignment before the existing arithmetic operations."""
    names = ("__add__", "__iadd__", "__sub__", "__isub__", "__truediv__", "__itruediv__", "__floordiv__")
    for name in names:
        setattr(cls, name, _aligned_operation(getattr(cls, name)))


def _install_properties(cls):
    """Install owner-backed geometry properties without retaining numerical arrays."""
    def transposed(self):
        """Return an independent transpose with its calibrated coordinate axes."""
        if owner_of(self) is None:
            return type(self)()
        return self.transpose(_=None)

    cls.T = property(transposed)
    for name in ("shape", "dtype", "size", "ndim"):
        descriptor = getattr(cls, name, None)

        def getter(self, field=name):
            owner = owner_of(self)
            if owner is None:
                return getattr(self.image, field)
            if field == "ndim":
                return 2
            if field == "size":
                return int(np.prod(owner.shape))
            return getattr(owner, field)

        setattr(cls, name, property(getter, getattr(descriptor, "fset", None), doc=f"Return image {name}."))


def install(cls):
    """Install explicit owner boundaries after composing the numerical methods."""
    _install_state(cls)
    _install_indexing(cls)
    _install_conversion(cls)
    _install_operators(cls)
    _install_properties(cls)


def _alignment_grid(owner, args, kwargs, reference_grid, shape):
    """Resolve registration calibration and require the exact numerical output geometry."""
    reference = args[0] if args else kwargs.get("ref")
    reference_owner = owner_of(reference) if hasattr(reference, "__dict__") else None
    grid = reference_owner._state.dataset if reference_owner is not None else owner._state.dataset
    if reference_grid is not None:
        grid = reference_grid
    if tuple(grid.sizes[axis] for axis in ("y", "x")) != shape:
        raise ValueError("Registration requires a reference grid matching the output shape")
    return grid


def _setitem(self, key, value):
    owner = owner_of(self)
    if owner is None:
        if isinstance(key, str):
            self.metadata[key] = value
            return
        raise ValueError("Assign image pixels before indexed edits")
    if isinstance(key, str):
        owner.metadata[key] = value
    else:
        owner[key] = value


def _delitem(self, key):
    if isinstance(key, str):
        del self.metadata[key]
    else:
        raise TypeError("Image pixels cannot be deleted; crop explicitly")


def _clone(self):
    owner = owner_of(self)
    if owner is None:
        result = type(self)()
        result.metadata = _copy_metadata(self.metadata)
        return result
    result = type(self).from_storage(owner.export_storage())
    # Preserve lazy image adapters and avoid owner/accessor import cycles.
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from .roi import copy_roi

    copy_roi(self, result)
    result.__dict__["_image_attrs"] = deepcopy(self.__dict__.get("_image_attrs", {}))
    for name in self._public_attrs:
        if not name.startswith("_") and name not in {"metadata", "image"}:
            setattr(result, name, deepcopy(getattr(self, name)))
    return result


def _getitem(self, key):
    owner = owner_of(self)
    if owner is None:
        return self.metadata[key] if isinstance(key, str) else self.image[key]
    if isinstance(key, str):
        return self.metadata[key]
    result = owner[key]
    if isinstance(result, np.ndarray) and result.ndim == 2:
        bounds = key if isinstance(key, tuple) else (key, slice(None))
        if len(bounds) == 2 and all(isinstance(bound, slice) for bound in bounds):
            package = owner.export_storage()
            package.dataset = package.dataset.isel(y=bounds[0], x=bounds[1])
            selected = type(self).from_storage(package)
            # Preserve lazy image adapters and avoid owner/accessor import cycles.
            # pylint: disable-next=import-outside-toplevel,cyclic-import
            from .roi import copy_roi

            copy_roi(self, selected, stale=True)
            return selected
        if calibrated(owner):
            raise ValueError("Calibrated image selection requires rectangular slices")
        return type(self)(result, metadata=self.metadata.copy())
    return result


def _no_array_pointer(self):
    raise AttributeError("Use the detached __array__ conversion")


def _from_xarray(class_, dataset, *, metadata=None):
    """Import native pixels and coordinates with optional typed metadata."""
    return class_.from_storage(ImageStorage.from_xarray(dataset, metadata=metadata))


def _array(self, dtype=None, copy=None):
    """Supply detached raw values to NumPy without exposing temporary pointers."""
    if copy is False:
        raise ValueError("ImageFile array conversion requires a copy")
    return self.to_numpy(masked=False, dtype=dtype)


def _mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=None):
    """Calculate a mean over selected finite, non-excluded pixels.

    Args:
        axis (int, tuple or None):
            Axes to reduce; None reduces the whole image.
        dtype (numpy.dtype or None):
            Optional accumulation dtype, following NumPy mean semantics.
        out (numpy.ndarray or None):
            Optional destination for the reduced values.
        keepdims (bool):
            Retain reduced dimensions with length one when true.

    Keyword Arguments:
        where (ROI, Boolean array, xarray.DataArray or None):
            Full image selection. True includes a pixel; labelled coordinates
            must exactly match the image. None uses current ROI, or all pixels
            when inactive. ROI.all() bypasses current ROI.

    Returns:
        scalar or numpy.ndarray:
            Mean, with NaN at positions with no eligible observations.

    Raises:
        ValueError:
            The selection dtype, shape or labelled axes do not match.
    """
    owner = owner_of(self)
    if owner is None:
        raise ValueError("A regional mean requires a populated image")
    # Preserve lazy image adapters and avoid owner/accessor import cycles.
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from .roi import image_selection

    values = selected_image_values(owner._state.dataset, image_selection(self, where))
    result = np.ma.asarray(values.mean(axis=axis, dtype=dtype, keepdims=keepdims))
    result = result.astype(np.result_type(result.dtype, float)).filled(np.nan)
    if out is not None:
        out[...] = result
        return out
    return result[()] if result.ndim == 0 else result


@contextmanager
def _edit_numpy(self):
    """Atomically edit fixed-shape values and exclusions in a detached draft."""
    owner = owner_of(self)
    if owner is None:
        raise ValueError("Transactions require a populated standalone image")
    with owner.edit_numpy() as draft:
        yield draft


def _to_numpy(self, *, masked=True, dtype=None):
    """Export detached pixels, optionally including exclusions or a dtype cast."""
    owner = owner_of(self)
    if owner is None:
        return ImageStorage.from_numpy(self.image, metadata=self.metadata).to_numpy(masked=masked, dtype=dtype)
    return owner.to_numpy(masked=masked, dtype=dtype)


def _export_storage(self):
    """Export a detached lossless image package."""
    owner = owner_of(self)
    if owner is not None:
        return owner.export_storage()
    return ImageStorage.from_numpy(self.image, metadata=self.metadata)


def _to_xarray(self):
    """Export detached native state, warning when typed metadata is omitted."""
    return self.export_storage().to_xarray()


@contextmanager
def _edit_xarray(self):
    """Atomically edit values and exclusions while preserving coordinates."""
    owner = owner_of(self)
    if owner is None:
        raise ValueError("Transactions require a populated standalone image")
    with owner.edit_xarray() as draft:
        yield draft
