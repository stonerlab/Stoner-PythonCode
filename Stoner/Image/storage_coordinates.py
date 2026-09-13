"""Pack image axes and coordinate maps into padded stacks without alignment."""

from copy import deepcopy

import numpy as np
import xarray as xr


def _common_attrs(variables, label):
    """Copy matching attributes, rejecting ambiguous shared stack descriptions."""
    first = variables[0].attrs
    if any(not xr.Dataset(attrs=first).identical(xr.Dataset(attrs=item.attrs)) for item in variables[1:]):
        raise ValueError(f"Incompatible {label} attributes; convert units or reconcile attributes before packing")
    return deepcopy(first)


def pack_coordinates(dataset, images):
    """Attach physical axes, coordinate maps, scalar coordinates and native attrs.

    Args:
        dataset (xarray.Dataset):
            Newly allocated positional stack, modified before publication.
        images (list of ImageStorage):
            Validated detached image packages in frame order.

    Notes:
        All inputs must describe the same scalar coordinates and use compatible
        coordinate and variable attributes. Spatial auxiliary coordinates other
        than dimension axes require an explicit two-dimensional y/x map.
        No coordinate alignment or implicit unit conversion takes place.
    """
    if not images:
        return
    sources = [image.dataset for image in images]
    extras = set(sources[0].coords) - {"y", "x"}
    reserved = {"frame", "valid_height", "valid_width", "index_y", "index_x"}
    if extras & reserved:
        raise ValueError("Image coordinates collide with reserved stack names")
    for source in sources:
        if set(source.coords) - {"y", "x"} != extras:
            raise ValueError("Images must supply the same scalar calibration coordinates")
        if any(source[name].dims not in ((), ("y", "x")) for name in extras):
            raise ValueError("Auxiliary spatial coordinates require an explicit y/x map")
    dataset.attrs = _common_attrs(sources, "Dataset")
    for name in ("intensity", "excluded"):
        dataset[name].attrs = _common_attrs([source[name] for source in sources], name)
    for axis in ("y", "x"):
        coordinates = [source[axis] for source in sources]
        attrs = _common_attrs(coordinates, axis)
        if not attrs and all(np.array_equal(coord.values, np.arange(coord.size)) for coord in coordinates):
            continue
        dtype = np.result_type(float, *[coord.dtype for coord in coordinates])
        values = np.full((len(images), dataset.sizes[axis]), np.nan, dtype=dtype)
        for i, coordinate in enumerate(coordinates):
            raw = coordinate.values
            converted = raw.astype(dtype)
            if raw.dtype.kind in "iu" and not np.array_equal(raw.astype(object), converted.astype(object)):
                raise ValueError("Physical coordinate packing would lose integer precision")
            values[i, :raw.size] = converted
        name = f"index_{axis}" if f"physical_{axis}" in extras else f"physical_{axis}"
        dataset.coords[name] = (("frame", axis), values, attrs)
    for name in sorted(extras):
        coordinates = [source[name] for source in sources]
        attrs = _common_attrs(coordinates, name)
        if any(coord.dims != coordinates[0].dims for coord in coordinates):
            raise ValueError("Coordinate representations must match before packing")
        if coordinates[0].dims:
            if name in dataset.coords:
                raise ValueError("Coordinate map collides with a packed physical axis")
            values = np.full(dataset.intensity.shape, np.nan)
            for i, coord in enumerate(coordinates):
                if coord.dtype.kind not in "iuf":
                    raise ValueError("Coordinate maps must be real numeric values")
                converted = coord.values.astype(float)
                if coord.dtype.kind in "iu" and not np.array_equal(
                    coord.values.astype(object), converted.astype(object)
                ):
                    raise ValueError("Coordinate map packing would lose integer precision")
                values[i, :coord.shape[0], :coord.shape[1]] = converted
            dataset.coords[name] = (("frame", "y", "x"), values, attrs)
            continue
        if any(coord.dtype != coordinates[0].dtype for coord in coordinates[1:]):
            raise ValueError(f"Scalar coordinate {name} must use a common dtype before packing")
        dataset.coords[name] = ("frame", np.stack([coord.values for coord in coordinates]), attrs)
