"""Draw ImageFile values using the same target and property conventions as Data.plot."""

from numbers import Real

import numpy as np
from matplotlib.colors import ListedColormap, to_rgba

from ..core.selection import resolve_selection, selected_image_values
from ..enums import CoordinateMode
from ..plot.base import PlotAccessor
from .coordinates import resolve_grid
from .storage_bridge import owner_of

__all__ = ["ImagePlotter"]


class ImagePlotter(PlotAccessor):
    """Render the current image without changing pixels, exclusions or metadata.

    Args:
        data (ImageFile):
            Owner of live image values and spatial coordinates.

    Notes:
        Calling the accessor draws an image. Image, imshow, contour, pcolormesh
        and hist return their primary Matplotlib Axes and never show/save them.
        Native coordinates are used by default; pixel mode explicitly requests
        local indices. Image orientation defaults to upper, including mesh and
        contour plots. Non-uniform grids use a mesh on image/callable paths.

        Properties, target inspection, styles and layout contexts follow
        Data.plot. Lookup performs no image reads or figure creation. Legacy
        ImageFile.imshow retains its separate Figure-return contract.

    See Also:
        :py:attr:`Stoner.Image.core.ImageFile.plot`:
            Obtain an accessor bound to an image.
        :py:class:`Stoner.enums.CoordinateMode`:
            Native and pixel coordinate choices.
        :doc:`/UserGuide/image_plotting`:
            Display, calibrated grids, histograms and migration examples.
    """

    def __call__(self, x=None, y=None, *, ax=None, cmap="gray", coordinates=CoordinateMode.NATIVE, **kwargs):
        """Draw the image; accept the same arguments as the image method.

        Args:
            x, y (str or None):
                Existing spatial coordinate names; None uses the dimension axes.

        Keyword Arguments:
            ax (matplotlib.axes.Axes or None):
                Explicit live target, otherwise the current compatible axes.
            cmap (str or matplotlib.colors.Colormap):
                Display colour map, defaulting to gray.
            coordinates (CoordinateMode or str):
                Native coordinates by default, or explicit local pixel indices.
            **kwargs (dict):
                Options documented by image, including origin and display limits.

        Returns:
            matplotlib.axes.Axes:
                Primary axes containing the image or mesh.
        """
        return self.image(x, y, ax=ax, cmap=cmap, coordinates=coordinates, **kwargs)

    def image(self, x=None, y=None, *, ax=None, cmap="gray", coordinates=CoordinateMode.NATIVE,
              origin="upper", aspect="equal", where=None, mask_color=None, mask_alpha=0.5,
              xlabel=None, ylabel=None, title=None, **kwargs):
        """Draw stored intensities, selecting a mesh for a non-uniform grid.

        Args:
            x, y (str or None):
                Existing spatial coordinate names, defaulting to the x/y axes.

        Keyword Arguments:
            ax (matplotlib.axes.Axes or None):
                Target axes; explicit targets do not create spare figures.
            cmap (str or matplotlib.colors.Colormap):
                Colour map; defaults to gray.
            coordinates (CoordinateMode or str):
                Native vectors or local pixel indices; defaults to native.
            origin (str):
                Upper places the first row at the top; lower at the bottom.
            aspect (str or float):
                Matplotlib aspect, defaulting to equal spatial units.
            where (Boolean array, xarray.DataArray or None):
                Temporary pixel inclusion; combined with exclusions/finite values.
            mask_color (colour or None):
                Optional overlay for ineligible pixels; None leaves them transparent.
            mask_alpha (float):
                Overlay opacity between zero and one, defaulting to 0.5.
            xlabel, ylabel, title (str or None):
                Presentation overrides; None uses coordinate labels/image title.
            **kwargs (dict):
                Native imshow or pcolormesh options, including norm/vmin/vmax.
                Extent is a display-only override on uniform grids. Singleton
                coordinate axes have unit pixel width in native units.

        Returns:
            matplotlib.axes.Axes:
                Primary target. Stored values and calibration are unchanged.

        Raises:
            ValueError:
                If coordinates, origin, selection or overlay opacity are invalid.
            TypeError:
                If legacy plotting keywords or wrong-domain enums are supplied.
        """
        return self._field("image", x, y, ax, cmap, coordinates, origin, aspect, where,
                           mask_color, mask_alpha, xlabel, ylabel, title, kwargs)

    def imshow(self, x=None, y=None, **kwargs):
        """Draw a uniformly spaced image with image options, returning Axes.

        Args:
            x, y (str or None):
                Existing spatial coordinate names.

        Keyword Arguments:
            **kwargs (dict):
                Options accepted by image. Non-uniform grids are rejected;
                use image or pcolormesh for those coordinates.

        Returns:
            matplotlib.axes.Axes:
                Axes containing a native Matplotlib image artist.
        """
        return self._named_field("imshow", x, y, kwargs)

    def pcolormesh(self, x=None, y=None, **kwargs):
        """Draw pixel cells with midpoint edges and image presentation options.

        Args:
            x, y (str or None):
                Existing spatial coordinate names.

        Keyword Arguments:
            **kwargs (dict):
                Image presentation/selection options and native pcolormesh options.
                Shading defaults to flat on the calculated cell edges.

        Returns:
            matplotlib.axes.Axes:
                Axes containing a QuadMesh with the image's raw intensity units.
        """
        return self._named_field("pcolormesh", x, y, kwargs)

    def contour(self, x=None, y=None, **kwargs):
        """Draw intensity contours at pixel centres using image presentation options.

        Args:
            x, y (str or None):
                Existing spatial coordinate names.

        Keyword Arguments:
            **kwargs (dict):
                Image presentation/selection options and native contour options,
                including levels. A contour requires at least two rows/columns
                and some eligible intensities. Extent is not a contour option.

        Returns:
            matplotlib.axes.Axes:
                Axes containing the native contour collection.
        """
        return self._named_field("contour", x, y, kwargs)

    def _named_field(self, kind, x, y, options):
        """Route explicitly named field methods through one preparation path."""
        defaults = {"ax": None, "cmap": "gray", "coordinates": CoordinateMode.NATIVE, "origin": "upper",
                    "aspect": "equal", "where": None, "mask_color": None, "mask_alpha": 0.5,
                    "xlabel": None, "ylabel": None, "title": None}
        values = [options.pop(name, default) for name, default in defaults.items()]
        return self._field(kind, x, y, *values, options)

    @staticmethod
    def _reject_legacy(options):
        """Reject ambiguous legacy/table options before reading values or creating targets."""
        legacy = {"figure", "showfig", "show_plot", "save_filename", "template", "xcol", "ycol", "zcol", "z", "kind"}
        found = sorted(legacy.intersection(options))
        if found:
            raise TypeError(f"ImageFile.plot does not accept {', '.join(found)}; use ax and x/y coordinates.")

    def _field(self, kind, x, y, ax, cmap, coordinates, origin, aspect, where,
               mask_color, mask_alpha, xlabel, ylabel, title, options):
        """Prepare a detached masked field before choosing a rendering target."""
        self._reject_legacy(options)
        if origin not in ("upper", "lower"):
            raise ValueError("origin must be 'upper' or 'lower'.")
        grid = resolve_grid(self._data, x, y, coordinates)
        dataset = owner_of(self._data)._state.dataset
        values = selected_image_values(dataset, where)
        if kind == "imshow" and not grid.uniform:
            raise ValueError("imshow requires uniform coordinates; use image or pcolormesh.")
        mesh = kind == "pcolormesh" or (kind == "image" and not grid.uniform)
        if "extent" in options and (mesh or kind == "contour"):
            raise ValueError("extent is only supported by uniform image rendering.")
        if kind == "contour" and (min(values.shape) < 2 or not values.count()):
            raise ValueError("Contours require at least two rows/columns and some eligible pixels.")
        if not isinstance(mask_alpha, Real) or not np.isfinite(mask_alpha) or not 0 <= mask_alpha <= 1:
            raise ValueError("mask_alpha must lie between zero and one.")
        overlay_map = ListedColormap([to_rgba(mask_color, mask_alpha)]) if mask_color is not None else None
        extent = options.pop("extent", (grid.x_edges[0], grid.x_edges[-1],
                                      grid.y_edges[-1] if origin == "upper" else grid.y_edges[0],
                                      grid.y_edges[0] if origin == "upper" else grid.y_edges[-1]))
        ax = self._target(ax)
        if mesh:
            options.setdefault("shading", "flat")
            artist = ax.pcolormesh(grid.x_edges, grid.y_edges, values, cmap=cmap, **options)
        elif kind == "contour":
            artist = ax.contour(grid.x, grid.y, values, cmap=cmap, **options)
        else:
            artist = ax.imshow(values, cmap=cmap, origin=origin, extent=extent, aspect=aspect, **options)
        if overlay_map is not None and np.ma.getmaskarray(values).any():
            overlay = np.ma.array(np.ones(values.shape), mask=~np.ma.getmaskarray(values))
            overlay_options = {"transform": artist.get_transform(), "zorder": artist.get_zorder() + 0.1}
            if mesh or kind == "contour":
                ax.pcolormesh(grid.x_edges, grid.y_edges, overlay, cmap=overlay_map, vmin=0, vmax=1,
                              shading="flat", **overlay_options)
            else:
                ax.imshow(overlay, cmap=overlay_map, vmin=0, vmax=1, origin=origin,
                          extent=extent, aspect=aspect, interpolation="nearest", **overlay_options)
        if mesh or kind == "contour":
            ax.set_xlim(grid.x_edges[0], grid.x_edges[-1])
            ax.set_ylim((grid.y_edges[-1], grid.y_edges[0]) if origin == "upper"
                        else (grid.y_edges[0], grid.y_edges[-1]))
            ax.set_aspect(aspect)
        self._decorate(ax, grid.xlabel if xlabel is None else xlabel, grid.ylabel if ylabel is None else ylabel, title)
        return ax

    def _decorate(self, ax, xlabel, ylabel, title):
        """Apply presentation defaults and record targets outside scientific metadata."""
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is None:
            title = self._data.__dict__.get("_image_attrs", {}).get("_title") or self._data.filename
        ax.set_title(title if title is not False else "")
        self._record(ax.figure, ax)

    def hist(self, *, ax=None, bins=256, where=None, weights=None,
             xlabel=None, ylabel=None, title=None, **kwargs):
        """Draw a histogram of eligible pixels without changing the image.

        Keyword Arguments:
            ax (matplotlib.axes.Axes or None):
                Explicit target, otherwise current compatible axes.
            bins (int, str or array):
                Native Matplotlib histogram bins, defaulting to 256.
            where (Boolean array, xarray.DataArray or None):
                Exact-shape temporary inclusion, True includes.
            weights (array, xarray.DataArray or None):
                Exact-shape weights; labelled inputs must match intensity axes.
                Invalid participating weights are rejected; negative weights are allowed.
            xlabel, ylabel, title (str or None):
                Presentation labels. Default labels are Intensity and Count (or
                Density), exchanged for a horizontal histogram.
            **kwargs (dict):
                Native Axes.hist options, including density, range and histtype.

        Returns:
            matplotlib.axes.Axes:
                Target containing counts computed from raw eligible intensities.

        Raises:
            ValueError:
                If selection/weights are incompatible or no eligible pixels remain.
        """
        self._reject_legacy(kwargs)
        owner = owner_of(self._data)
        if owner is None:
            raise ValueError("Image plotting requires a populated ImageFile owner.")
        dataset = owner._state.dataset
        values = selected_image_values(dataset, where)
        eligible = ~np.ma.getmaskarray(values)
        if not eligible.any():
            raise ValueError("No finite, unexcluded pixels remain for the histogram.")
        if weights is not None:
            if hasattr(weights, "dims"):
                # Reuse labelled-selection alignment validation without treating
                # numerical weights as a Boolean selector.
                resolve_selection(weights.notnull(), dataset.intensity)
                weights = weights.values
            weights = np.ma.asarray(weights)
            if weights.shape != values.shape or weights.dtype.kind not in "biuf":
                raise ValueError("weights must be a real array matching the image shape.")
            if np.any(np.ma.getmaskarray(weights)[eligible]) or not np.isfinite(weights.data[eligible]).all():
                raise ValueError("Participating histogram weights must be finite and unmasked.")
            weights = weights.data[eligible]
        ax = self._target(ax)
        ax.hist(values.data[eligible], bins=bins, weights=weights, **kwargs)
        frequency = "Density" if kwargs.get("density") else "Count"
        default_x, default_y = ((frequency, "Intensity") if kwargs.get("orientation") == "horizontal"
                                else ("Intensity", frequency))
        self._decorate(ax, default_x if xlabel is None else xlabel, default_y if ylabel is None else ylabel, title)
        return ax
