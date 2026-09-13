"""Apply exact image permutations to values, exclusions and separable coordinates."""

from operator import index


def permute(package, name, args=(), kwargs=None):
    """Return a detached package with matching pixel and coordinate permutations.

    Args:
        package (ImageStorage):
            Validated two-dimensional source package.
        name (str):
            Transpose, flip or quarter-turn operation.
        args (tuple):
            Positional arguments for NumPy-compatible transpose or swapaxes.
        kwargs (dict or None):
            Keyword arguments for those operations.

    Returns:
        ImageStorage:
            Independent transformed values, mask, coordinates and typed metadata.

    Notes:
        Swapping axes also swaps their names and attributes. Auxiliary coordinate
        dimensions follow their source axis. No resampling or unit conversion occurs.
    """
    kwargs = dict(kwargs or {})
    if name == "rot90":
        k = index(args[0] if args else kwargs.pop("k", 1))
        axes = args[1] if len(args) > 1 else kwargs.pop("axes", (0, 1))
        axes = tuple(index(axis) for axis in axes)
        if len(args) > 2 or len(axes) != 2 or any(axis not in (-2, -1, 0, 1) for axis in axes):
            raise ValueError("rot90 requires two image axes")
        axes = tuple(axis % 2 for axis in axes)
        if axes not in ((0, 1), (1, 0)) or kwargs:
            raise ValueError("Invalid rot90 axes or arguments")
        result = package.copy()
        for _ in range((k if axes == (0, 1) else -k) % 4):
            result = permute(result, "CCW")
        return result
    result = package.copy()
    swap, reverse_y, reverse_x = False, False, False
    if name in {"T", "transpose"}:
        if args and "axes" in kwargs:
            raise TypeError("Transpose axes were supplied twice")
        axes = kwargs.pop("axes", args if len(args) > 1 else (args[0] if args else None))
        if axes is None:
            axes = (1, 0)
        axes = tuple(index(axis) for axis in axes)
        if len(axes) != 2 or any(axis not in (-2, -1, 0, 1) for axis in axes):
            raise ValueError("Transpose requires two image axes")
        axes = tuple(axis % 2 for axis in axes)
        if axes not in ((0, 1), (1, 0)):
            raise ValueError("Transpose axes must be a permutation")
        swap = axes == (1, 0)
    elif name == "swapaxes":
        if args:
            if len(args) != 2:
                raise TypeError("swapaxes requires two axes")
            axes = args
        else:
            axes = (kwargs.pop("axis1"), kwargs.pop("axis2"))
        axes = tuple(index(axis) for axis in axes)
        if any(axis not in (-2, -1, 0, 1) for axis in axes):
            raise ValueError("Axis is outside this two-dimensional image")
        swap = axes[0] % 2 != axes[1] % 2
    elif name in {"CW", "CCW", "flip_h", "flip_v"}:
        if args:
            raise TypeError("This permutation takes no positional arguments")
        swap = name in {"CW", "CCW"}
        reverse_y = name in {"CCW", "flip_v"}
        reverse_x = name in {"CW", "flip_h"}
    else:
        raise ValueError(f"Unsupported exact permutation: {name}")
    if kwargs:
        raise TypeError(f"Unexpected permutation arguments: {', '.join(kwargs)}")
    dataset = result.dataset
    if swap:
        # Simultaneous renaming avoids collisions and preserves auxiliary dimensions.
        dataset = dataset.rename({"y": "x", "x": "y"}).transpose("y", "x")
    dataset = dataset.isel(y=slice(None, None, -1 if reverse_y else 1),
                           x=slice(None, None, -1 if reverse_x else 1))
    result.dataset = dataset.copy(deep=True)
    result.validate()
    return result
