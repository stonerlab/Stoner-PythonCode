"""Prepare and render scalar, vector and volume fields with Matplotlib."""

from dataclasses import dataclass
from numbers import Integral
from re import Pattern

import numpy as np
from matplotlib import colormaps, colors, rcParams
from matplotlib.cm import ScalarMappable
from scipy.interpolate import griddata
from scipy.spatial import QhullError

from .utils import hsl2rgb, joy_division

__all__ = []


@dataclass(frozen=True)
class FieldSamples:
    """Hold independent field samples and validity in original row order."""

    points: np.ndarray
    values: np.ndarray
    coordinate_valid: np.ndarray
    valid: np.ndarray


def prepare_samples(data, columns, dimensions, components):
    """Read only required real columns and combine their exclusions before drawing."""
    positions = [columns.x]
    for name in dimensions[1:]:
        selected = getattr(columns, name)
        if len(selected) != 1:
            raise ValueError(f"This field requires exactly one {name} column.")
        positions.append(selected[0])
    values = []
    for name in components:
        selected = getattr(columns, name)
        if len(selected) != 1:
            raise ValueError(f"This field requires exactly one {name} column.")
        values.append(selected[0])
    arrays = [np.ma.asarray(data.column(position)) for position in positions + values]
    if any(array.dtype.kind not in "biuf" for array in arrays):
        raise TypeError("Field plotting requires real numerical columns.")
    valid = [~np.ma.getmaskarray(array) & np.isfinite(array.data) for array in arrays]
    coordinate_valid = np.logical_and.reduce(valid[:len(positions)])
    complete = np.logical_and.reduce(valid)
    if not complete.any():
        raise ValueError("No finite, unexcluded field samples remain.")
    return FieldSamples(
        np.column_stack([array.data for array in arrays[:len(positions)]]).astype(float),
        (np.column_stack([array.data for array in arrays[len(positions):]]).astype(float)
         if values else np.empty((len(data), 0))),
        coordinate_valid, complete,
    )


def _grid_axes(samples, shape, limits):
    """Resolve inclusive grid centres in X/Y[/Z] order, preserving legacy counts."""
    dimensions = samples.points.shape[1]
    if shape is not None:
        try:
            shape = tuple(shape)
        except TypeError as error:
            raise ValueError(f"shape must contain {dimensions} positive integer counts.") from error
        if len(shape) != dimensions or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in shape):
            raise ValueError(f"shape must contain {dimensions} positive integer counts in X/Y/Z order.")
    axes = []
    for index, limit in enumerate(limits):
        values = np.unique(samples.points[samples.coordinate_valid, index])
        count = len(values) if shape is None else int(shape[index])
        if limit is not None and np.shape(limit) != (2,):
            raise ValueError("Grid limits must be pairs of lower and upper centre coordinates.")
        low, high = (values[0], values[-1]) if limit is None else limit
        if not np.isfinite([low, high]).all() or high < low or (high == low and count > 1):
            raise ValueError("Grid limits must be finite and increasing, with a non-zero span for multiple centres.")
        axes.append(np.linspace(low, high, count))
    return tuple(axes)


def grid_samples(samples, *, shape=None, limits=None, method="linear"):
    """Use the existing SciPy interpolation algorithm on validated field samples.

    Default counts equal the number of distinct coordinates along each axis.
    Two-dimensional arrays have rows along Y and columns along X; volumes use
    X/Y/Z indexing. Excluded measured grid sites stay missing. Complete regular
    lattices are reordered directly, retaining holes without interpolation.
    """
    dimensions = samples.points.shape[1]
    if method not in ("linear", "nearest", "cubic") or method == "cubic" and dimensions != 2:
        raise ValueError("Use linear/nearest interpolation, or cubic for a 2D field.")
    axes = _grid_axes(samples, shape, (None,)*dimensions if limits is None else limits)
    mesh = tuple(np.meshgrid(*axes, indexing="xy" if dimensions == 2 else "ij"))
    points, indices = np.unique(samples.points[samples.valid], axis=0, return_index=True)
    values = samples.values[samples.valid][indices]
    observed = np.unique(samples.points[samples.coordinate_valid], axis=0)
    complete = len(observed) == np.prod([len(axis) for axis in axes])
    complete &= all(np.array_equal(np.unique(observed[:, i]), axis) for i, axis in enumerate(axes))
    if complete:
        result = np.full(mesh[0].shape + (values.shape[1],), np.nan)
        locations = tuple(np.searchsorted(axis, points[:, i]) for i, axis in enumerate(axes))
        result[locations[::-1] if dimensions == 2 else locations] = values
    else:
        try:
            result = griddata(points, values, mesh, method=method)
        except QhullError as error:
            raise ValueError("Cannot interpolate this field geometry; supply non-degenerate coordinates or method='nearest'.") from error
        # Retain exclusions at measured sites even when other samples surround them.
        bad = samples.points[np.isfinite(samples.points).all(axis=1) & ~samples.valid]
        if bad.size:
            good_keys = {tuple(point) for point in points}
            bad = np.array([point for point in bad if tuple(point) not in good_keys]).reshape(-1, dimensions)
            locations = tuple(np.minimum(np.searchsorted(axis, bad[:, i]), len(axis)-1) for i, axis in enumerate(axes))
            matches = np.logical_and.reduce([np.isclose(axis[index], bad[:, i], rtol=1e-10, atol=1e-12)
                                              for i, (axis, index) in enumerate(zip(axes, locations))])
            locations = tuple(index[matches] for index in locations)
            result[locations[::-1] if dimensions == 2 else locations] = np.nan
    result = np.ma.masked_invalid(result)
    if result.count() == 0:
        raise ValueError("No finite field values lie on the requested grid.")
    return axes, mesh, result


def cell_edges(centres):
    """Return cell boundaries halfway between centres, extending the outer cells."""
    centres = np.asarray(centres)
    if len(centres) == 1:
        return np.array([centres[0]-0.5, centres[0]+0.5])
    middle = (centres[1:] + centres[:-1])/2
    return np.r_[centres[0]-(middle[0]-centres[0]), middle, centres[-1]+(centres[-1]-middle[-1])]


def vector_colours(values):
    """Reuse Stoner's HSL direction/magnitude mapping with finite zero-field limits."""
    u, v = values[:, :2].T
    w = values[:, 2] if values.shape[1] == 3 else np.zeros(len(values))
    span = np.ptp(w)
    luminance = (w-w.min())/span if span else np.full(len(values), 0.5)
    magnitude = np.linalg.norm(np.column_stack((u, v, w)), axis=1)
    saturation = magnitude/magnitude.max() if magnitude.max() else np.zeros(len(values))
    hue = 0.5 + np.arctan2(u, v)/(2*np.pi)
    return hsl2rgb(hue, saturation, luminance).astype(float)/255


def _row_options(options, valid):
    """Align explicitly supplied per-sample artist colours and sizes with exclusions."""
    options = dict(options)
    for key in ("c", "s", "color", "colors"):
        value = options.get(key)
        if value is None or isinstance(value, str) or np.ndim(value) == 0:
            continue
        if key in ("color", "colors") and colors.is_color_like(value):
            continue
        array = np.asanyarray(value)
        if len(array) != len(valid):
            raise ValueError(f"Per-sample {key} must have one entry per source row.")
        options[key] = array[valid]
    return options


def _voxel_data(samples, shape, limits, method, visible, filled):
    """Prepare voxel centres, corners and visibility before selecting axes."""
    axes, mesh, values = grid_samples(samples, shape=shape, limits=limits, method=method)
    values = values[..., 0]
    occupied = ~np.ma.getmaskarray(values)
    if visible is not None and filled is not None:
        raise ValueError("Specify visible or filled, not both.")
    if visible is not None:
        if not callable(visible):
            raise TypeError("visible must be a callable of grid-centre X, Y and Z coordinates.")
        selection = np.asarray(visible(*mesh))
    elif filled is not None:
        selection = np.asarray(filled)
    else:
        selection = np.ones(values.shape, dtype=bool)
    if selection.dtype.kind != "b" or selection.shape not in ((), values.shape):
        raise ValueError(f"Voxel visibility must be Boolean with shape {values.shape}, or a scalar Boolean.")
    occupied &= selection
    corners = tuple(np.meshgrid(*(cell_edges(axis) for axis in axes), indexing="ij"))
    return mesh, corners, values, occupied


def draw_field(plot, kind, columns, *, ax=None, shape=None, method="linear", xlim=None, ylim=None, zlim=None,
               colorbar=True, annotations=None, **kwargs):
    """Validate and draw one explicit field request, returning its primary axes."""
    dimensions, components = {
        "surface": ("xy", "z"), "contour": ("xy", "z"), "pcolormesh": ("xy", "z"),
        "image": ("xy", "z"), "scatter3d": ("xyz", ""), "waterfall": ("xy", "z"),
        "vector": ("xy", "uvw" if columns.w else "uv"), "quiver": ("xy", "uv"),
        "vector3d": ("xyz", "uvw"), "voxels": ("xyz", "u"),
    }[kind]
    samples = prepare_samples(plot._data, columns, dimensions, components)
    options = {key: plot._text(value) for key, value in kwargs.items()}
    if kind == "vector" and {"cmap", "norm", "vmin", "vmax"}.intersection(options):
        raise ValueError("vector uses directional RGB colours; scalar colour maps belong to image/pcolormesh.")
    if kind in ("image", "vector") and "extent" in options:
        raise ValueError("Image extents come from coordinate centres; use xlim/ylim to change the grid.")
    projection = "3d" if kind in ("surface", "scatter3d", "vector3d", "voxels") else "rectilinear"
    if kind in ("scatter3d", "quiver", "vector3d", "waterfall") and (shape is not None or method != "linear"):
        raise ValueError(f"{kind} uses ungridded samples; omit shape and method.")
    if zlim is not None and projection != "3d":
        raise ValueError("zlim requires a 3D plotting method.")
    if kind in ("surface", "contour", "pcolormesh", "image"):
        axes, mesh, values = grid_samples(samples, shape=shape, limits=(xlim, ylim), method=method)
        if kind in ("surface", "contour") and any(len(axis) < 2 for axis in axes):
            raise ValueError(f"{kind} requires at least two grid centres along X and Y.")
    if kind == "vector":
        # A 2D pixel and its overlay must use the same first valid vector at a site.
        _, first = np.unique(samples.points[samples.valid], axis=0, return_index=True)
        unique_valid = np.zeros_like(samples.valid)
        unique_valid[np.flatnonzero(samples.valid)[first]] = True
        samples = FieldSamples(samples.points, samples.values, samples.coordinate_valid, unique_valid)
        arrows = options.pop("arrows", True)
        quiver_kwargs = _row_options(options.pop("quiver_kwargs", {}), samples.valid)
        # Interpolate the same directional RGB samples as the existing vector image.
        rgb = np.zeros((len(samples.valid), 3))
        rgb[samples.valid] = vector_colours(samples.values[samples.valid])
        rgb_samples = FieldSamples(samples.points, rgb, samples.coordinate_valid, samples.valid)
        axes, mesh, rgb_grid = grid_samples(rgb_samples, shape=shape, limits=(xlim, ylim), method=method)
        rgb_grid = np.ma.clip(rgb_grid, 0, 1)
    if kind in ("quiver", "vector3d", "scatter3d"):
        options = _row_options(options, samples.valid)
    if kind == "voxels":
        _, corners, values, occupied = _voxel_data(samples, shape, (xlim, ylim, zlim), method,
                                                  options.pop("visible", None), options.pop("filled", None))
        cmap = colormaps.get_cmap(options.pop("cmap", None))
        norm = options.pop("norm", None)
        vmin, vmax = options.pop("vmin", None), options.pop("vmax", None)
        if norm is not None and (vmin is not None or vmax is not None):
            raise ValueError("Supply norm or vmin/vmax, not both.")
        mapper = ScalarMappable(norm=norm or colors.Normalize(vmin, vmax), cmap=cmap)
        rgba = mapper.to_rgba(values.ravel()).reshape(values.shape + (4,))
        face_alpha, edge_alpha = options.pop("face_alpha", 0.5), options.pop("edge_alpha", 0.9)
        face = rgba.copy()
        face[..., 3] *= face_alpha
        edge = rgba.copy()
        edge[..., 3] *= edge_alpha
    ax = plot._target(ax, projection)
    artist = None
    if kind == "surface":
        if "color" not in options and "facecolors" not in options:
            options.setdefault("cmap", rcParams["image.cmap"])
        artist = ax.plot_surface(*mesh, values[..., 0].filled(np.nan), **options)
    elif kind == "contour":
        filled = options.pop("filled", False)
        artist = (ax.contourf if filled else ax.contour)(*mesh, values[..., 0], **options)
    elif kind == "pcolormesh":
        artist = ax.pcolormesh(*mesh, values[..., 0], **options)
    elif kind in ("image", "vector"):
        extent = tuple(edge for axis in axes for edge in cell_edges(axis)[[0, -1]])
        options.setdefault("origin", "lower")
        options.setdefault("aspect", "auto")
        if kind == "vector":
            options.setdefault("alpha", 0.75)
        artist = ax.imshow(values[..., 0] if kind == "image" else rgb_grid, extent=extent, **options)
        if kind == "vector" and arrows:
            _quiver(ax, samples, quiver_kwargs)
    elif kind == "scatter3d":
        artist = ax.scatter(*samples.points[samples.valid].T, **options)
    elif kind == "quiver":
        artist = _quiver(ax, samples, options)
    elif kind == "vector3d":
        if "color" not in options and "colors" not in options:
            options["colors"] = vector_colours(samples.values[samples.valid])
        artist = ax.quiver(*samples.points[samples.valid].T, *samples.values[samples.valid].T, **options)
    elif kind == "voxels":
        ax.voxels(*corners, occupied, facecolors=face, edgecolors=edge, **options)
        artist = mapper
    elif kind == "waterfall":
        joy_division(*samples.points[samples.valid].T, samples.values[samples.valid, 0], ax=ax, **options)
    if (colorbar and kind in ("surface", "contour", "pcolormesh", "image", "voxels")
            and (kind != "surface" or artist.get_array() is not None)):
        label = plot._label(columns.u[0] if kind == "voxels" else columns.z[0])
        ax.figure.colorbar(artist, ax=ax, label=plot._text(label))
    defaults = [plot._label(columns.x), plot._label(columns.y[0])]
    if kind == "waterfall":
        defaults[1] = plot._label(columns.z[0])
    if projection == "3d":
        defaults.append(plot._label(columns.z[0]))
    _finish(plot, ax, defaults, annotations)
    if kind in ("scatter3d", "quiver", "vector3d", "waterfall"):
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
    if projection == "3d" and zlim is not None and kind != "voxels":
        ax.set_zlim(zlim)
    plot._record(ax.figure, ax)
    return ax


def _quiver(ax, samples, options):
    """Draw arrows in data coordinates with lengths in the same coordinate units."""
    defaults = {"pivot": "mid", "angles": "xy", "scale_units": "xy", "scale": 1}
    defaults.update(options)
    return ax.quiver(*samples.points[samples.valid, :2].T, *samples.values[samples.valid, :2].T, **defaults)


def _finish(plot, ax, defaults, annotations):
    """Fill missing field annotations while respecting explicit labels and titles."""
    annotations = {} if annotations is None else annotations
    for name, default in zip(("xlabel", "ylabel", "zlabel"), defaults):
        if name in annotations or not getattr(ax, f"get_{name}")():
            getattr(ax, f"set_{name}")(plot._text(annotations.get(name, default) or ""))
    if "title" in annotations or not ax.get_title():
        ax.set_title(plot._text(annotations.get("title", plot._data.basename) or ""))


def draw_matrix(plot, x=None, y=None, *, rect=None, ax=None, colorbar=True, annotations=None, **kwargs):
    """Render a matrix whose data rows represent X and data columns represent Y."""
    data = plot._data
    if isinstance(x, bool) or isinstance(y, bool):
        raise TypeError("Matrix coordinates must be selectors or vectors, not Booleans.")
    xcolumn = 0 if x is None else None
    if isinstance(x, (Integral, str, Pattern, slice)):
        positions = data.find_col(x, force_list=True)
        if len(positions) != 1:
            raise ValueError("The matrix X selector must identify one column.")
        xcolumn = positions[0]
    if isinstance(y, Integral) and not 0 <= y < len(data):
        raise ValueError("The matrix Y coordinate row must lie within the table.")
    row = int(y)+1 if isinstance(y, Integral) else 0
    column = xcolumn+1 if xcolumn is not None else 0
    if rect is None:
        rect = (row, column)
    rect = tuple(rect)
    if len(rect) not in (2, 4) or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in rect):
        raise ValueError("rect must contain non-negative (row, column[, rows, columns]) integers.")
    row, column = rect[:2]
    rows, cols = rect[2:] if len(rect) == 4 else (len(data)-row, data.shape[1]-column)
    if rows < 2 or cols < 2 or row+rows > len(data) or column+cols > data.shape[1]:
        raise ValueError("The matrix block must contain at least two rows and columns within the table.")
    matrix = np.ma.column_stack([data.column(i)[row:row+rows] for i in range(column, column+cols)])
    if matrix.dtype.kind not in "biuf":
        raise TypeError("Matrix plotting requires real numerical values.")
    matrix = matrix.astype(float)
    xvalues = data.column(xcolumn)[row:row+rows] if xcolumn is not None else np.ma.asarray(x)
    if xcolumn is None and len(xvalues) == len(data):
        xvalues = xvalues[row:row+rows]
    if y is None:
        yvalues = np.array([float(header) for header in data.column_headers[column:column+cols]])
    elif isinstance(y, Integral):
        yvalues = np.ma.concatenate([data.column(i)[y:y+1] for i in range(column, column+cols)])
    else:
        yvalues = np.ma.asarray(y)
        if len(yvalues) == data.shape[1]:
            yvalues = yvalues[column:column+cols]
    for values, count in ((xvalues, rows), (yvalues, cols)):
        if values.shape != (count,) or np.ma.getmaskarray(values).any() or not np.isfinite(values).all():
            raise ValueError("Matrix coordinates must be finite unexcluded vectors matching the selected block.")
    matrix = np.ma.masked_invalid(matrix)
    if matrix.count() == 0:
        raise ValueError("No finite, unexcluded matrix samples remain.")
    mesh = np.meshgrid(xvalues, yvalues)
    ax = plot._target(ax, "3d")
    if "color" not in kwargs and "facecolors" not in kwargs:
        kwargs.setdefault("cmap", rcParams["image.cmap"])
    artist = ax.plot_surface(*mesh, matrix.T.filled(np.nan), **{key: plot._text(value) for key, value in kwargs.items()})
    if colorbar and artist.get_array() is not None:
        ax.figure.colorbar(artist, ax=ax)
    _finish(plot, ax, [plot._label(xcolumn) if xcolumn is not None else "X", "Y", "Value"], annotations)
    plot._record(ax.figure, ax)
    return ax
