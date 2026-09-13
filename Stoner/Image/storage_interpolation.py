"""Carry physical coordinate maps through numerical image resampling."""

from copy import deepcopy
from inspect import signature

import numpy as np
import xarray as xr
from scipy.ndimage import maximum_filter


GEOMETRY = {"rotate", "resize", "rescale", "warp", "translate", "shift", "zoom", "affine_transform", "gridimage"}


def require_alignment(left, right):
    """Reject image operands with differing coordinates before numerical conversion.

    Raw arrays, scalars and uncalibrated images are positional operands. Two
    calibrated images must have identical coordinates and units, without alignment.
    """
    from .storage_bridge import owner_of, calibrated

    first = owner_of(left) if hasattr(left, "__dict__") else None
    second = owner_of(right) if hasattr(right, "__dict__") else None
    if first is None or second is None:
        return
    if not calibrated(first) or not calibrated(second):
        return
    if not first._state.dataset.coords.to_dataset().identical(second._state.dataset.coords.to_dataset()):
        raise ValueError("Image operands require identical coordinates and units; resample explicitly first")


def interpolate(package, function, args, kwargs, working):
    """Resample pixels, exclusions and physical maps using the same geometry.

    Args:
        package (ImageStorage):
            Source image with its calibration and typed metadata.
        function (callable):
            Existing numerical geometry implementation.
        args (tuple):
            Positional geometry arguments in the existing pixel convention.
        kwargs (dict):
            Numerical geometry options.
        working (numpy.ma.MaskedArray):
            Detached numerical input used by the method adapter.

    Returns:
        ImageStorage:
            Independent result with positional axes and two-dimensional physical
            maps. Coordinates outside the source are NaN and excluded. Coordinate
            maps use linear interpolation; higher-order pixel interpolation uses
            conservatively expanded exclusions around masked samples.
    """
    if function.__name__ == "resize":
        from skimage.transform import resize
        function = resize
    bound = signature(function).bind(working, *args, **kwargs)
    bound.apply_defaults()
    parameters = dict(bound.arguments)
    if parameters.get("output") is not None:
        raise ValueError("Calibrated interpolation does not accept an external output buffer")
    if parameters.get("channel_axis") is not None:
        raise ValueError("Image geometry requires two spatial axes")
    first = next(iter(parameters))

    def run(values, *, field=False):
        options = dict(parameters)
        from .numerical import numerical_image
        array = numerical_image(np.array(values, dtype=float, copy=True)) if field else values
        if field:
            array.metadata = working.metadata.copy()
            overrides = {"order": 1, "mode": "constant", "cval": np.nan,
                         "clip": False, "preserve_range": True, "anti_aliasing": False,
                         "prefilter": False, "add_metadata": False}
            overrides.update(fill_value=np.nan, method="linear")
            for name, value in overrides.items():
                if name in options:
                    options[name] = value
        options[first] = array
        call = signature(function).bind_partial()
        call.arguments.update(options)
        result = function(*call.args, **call.kwargs)
        return result

    numerical = run(working)
    if not isinstance(numerical, np.ndarray) or numerical.ndim != 2:
        raise ValueError("Image geometry must return a two-dimensional array")
    source = package.dataset
    height, width = numerical.shape
    result = package.copy()
    dataset = xr.Dataset(
        {"intensity": (("y", "x"), np.ma.getdata(numerical).copy(), deepcopy(source.intensity.attrs)),
         "excluded": (("y", "x"), np.ma.getmaskarray(numerical).copy(), deepcopy(source.excluded.attrs))},
        coords={"y": np.arange(height), "x": np.arange(width)}, attrs=deepcopy(source.attrs))
    maps = dict(source.coords)
    for axis in ("y", "x"):
        if f"physical_{axis}" not in maps:
            maps[f"physical_{axis}"] = source[axis]
        del maps[axis]
    outside = np.zeros(numerical.shape, bool)
    for name, coordinate in maps.items():
        if not coordinate.dims:
            dataset.coords[name] = coordinate.copy(deep=True)
            continue
        if coordinate.dtype.kind not in "iuf":
            raise ValueError("Interpolated spatial calibration must be real numeric values")
        values = coordinate.broadcast_like(source.intensity).transpose("y", "x").values
        mapped = np.asarray(run(values, field=True))
        outside |= ~np.isfinite(mapped)
        dataset.coords[name] = (("y", "x"), mapped, deepcopy(coordinate.attrs))
    order = parameters.get("order", 3 if parameters.get("method") == "cubic" else 1)
    order = 1 if order is None else order
    exclusions = source.excluded.values
    radius = int(np.ceil(order / 2)) if order > 1 else 0
    if function.__name__ in {"resize", "rescale"} and parameters.get("anti_aliasing") is not False:
        ratio = max(np.array(source.intensity.shape) / np.array(numerical.shape))
        sigma = parameters.get("anti_aliasing_sigma")
        radius += int(np.ceil(4 * np.max(sigma if sigma is not None else max(0, (ratio - 1) / 2))))
    if radius:
        exclusions = maximum_filter(exclusions, size=2 * radius + 1, mode="constant")
    coverage = np.asarray(run(exclusions.astype(float), field=True))
    dataset.excluded.data |= outside | ~np.isfinite(coverage) | (coverage > 0)
    result.dataset = dataset
    if hasattr(numerical, "metadata"):
        result.metadata = numerical.metadata.copy()
    result.validate()
    return result
