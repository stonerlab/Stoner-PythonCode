"""Manage handle-local ROI state and adapt reusable descriptions to image coordinates."""

from copy import deepcopy

from ..core.roi import ROI, geometry_signature
from ..core.selection import resolve_selection
from ..enums import CoordinateMode
from .coordinates import resolve_grid
from .storage_bridge import owner_of

__all__ = ["ImageROI"]


def invalidate_roi(image):
    """Retain active selection intent but reject use after a geometric operation."""
    state = image.__dict__.get("_roi_state")
    if state is not None:
        image.__dict__["_roi_state"] = (state[0], state[1], True)


def copy_roi(source, destination, *, stale=False):
    """Copy operational state independently, without exporting it to image storage."""
    if "_roi_state" in source.__dict__:
        destination.__dict__["_roi_state"] = deepcopy(source.__dict__["_roi_state"])
        if stale:
            invalidate_roi(destination)


def image_selection(image, where=None, *, current=True):
    """Resolve current/explicit/all precedence at the ImageFile boundary only."""
    owner = owner_of(image)
    if owner is None:
        raise ValueError("ROI selection requires a populated ImageFile.")
    target = owner._state.dataset.intensity
    if where is None and current:
        return image.roi.resolve().values
    if isinstance(where, ROI):
        grid = resolve_grid(image) if where._coordinates is CoordinateMode.NATIVE else None
        return where.resolve(target, grid=grid).values
    return resolve_selection(where, target)


class ImageROI:
    """Manage the current region of interest for one ImageFile handle.

    Args:
        image (ImageFile):
            Owner whose operational ROI state is inspected or changed.

    Notes:
        Setters return the owner. Lookup does not read pixels or create figures.
        ROI membership is independent of exclusions and finite intensities.
        Native constructors capture the owner's x/y unit labels automatically.
        Geometry-changing operations require clear or reselection before reuse.
    """

    def __init__(self, image):
        self._image = image

    def __bool__(self):
        raise TypeError("Use image.roi.active; an active ROI may be empty.")

    @property
    def current(self):
        """Return the immutable current description, or None when inactive."""
        state = self._image.__dict__.get("_roi_state")
        return None if state is None else state[0]

    @property
    def active(self):
        """Return whether a current ROI exists, including empty or stale regions."""
        return self.current is not None

    @property
    def stale(self):
        """Return whether the current description needs explicit clear or reselection."""
        state = self._image.__dict__.get("_roi_state")
        if state is None:
            return False
        return state[2] or state[1] != geometry_signature(self._target())

    def _target(self):
        owner = owner_of(self._image)
        if owner is None:
            raise ValueError("ROI selection requires a populated ImageFile.")
        return owner._state.dataset.intensity

    def set(self, region):
        """Validate a description before atomically replacing the current ROI.

        Args:
            region (ROI):
                Reusable immutable description compatible with this image.

        Returns:
            ImageFile:
                Owner for chaining. Pixels, exclusions and metadata are unchanged.
        """
        if not isinstance(region, ROI):
            raise TypeError("region must be an ROI description; use from_mask for Boolean arrays.")
        target = self._target()
        image_selection(self._image, region)
        self._image.__dict__["_roi_state"] = (region, geometry_signature(target), False)
        return self._image

    def clear(self):
        """Clear current ROI state and return the owner, preserving exclusions."""
        self._image.__dict__.pop("_roi_state", None)
        return self._image

    def all(self):
        """Set an explicit whole-domain ROI and return the owner."""
        return self.set(ROI.all())

    def resolve(self, where=None):
        """Return detached labelled inclusion, using current ROI unless overridden.

        Args:
            where (ROI, Boolean array, xarray.DataArray or None):
                None uses current state; ROI.all() explicitly bypasses it. An
                inactive accessor includes the whole handle's domain.

        Returns:
            xarray.DataArray:
                Independent Boolean membership. True includes, even if excluded
                or non-finite. Consumers apply scientific eligibility separately.

        Raises:
            ValueError:
                The current state is stale or explicit geometry is incompatible.
        """
        target = self._target()
        if where is None:
            if self.stale:
                raise ValueError("Current ROI is stale after a geometry change; clear or reselect it.")
            where = self.current
        included = image_selection(self._image, where, current=False)
        result = target.copy(data=included).rename("included").copy(deep=True)
        result.attrs = {}
        return result

    def box(self, xmin, xmax, ymin, ymax, *, coordinates=CoordinateMode.PIXEL):
        """Set a half-open x/y box and return the owner; native mode captures its units.

        Args:
            xmin, xmax, ymin, ymax (float):
                Finite ordered bounds. Equal bounds select no pixels.

        Keyword Arguments:
            coordinates (CoordinateMode or str):
                Pixel indices by default, or native physical axis values.

        Returns:
            ImageFile:
                Owner with the validated current ROI.
        """
        return self.set(ROI.box(xmin, xmax, ymin, ymax, coordinates=coordinates, reference=self._target()))

    def from_mask(self, included):
        """Set exact Boolean inclusion bound to this grid and return the owner.

        Args:
            included (Boolean array or xarray.DataArray):
                Exact shape/labels; True includes pixels, unlike image.mask.

        Returns:
            ImageFile:
                Owner with independent current membership.
        """
        return self.set(ROI.from_mask(included, reference=self._target()))

    def polygon(self, vertices, *, coordinates=CoordinateMode.PIXEL):
        """Set a closed polygon and return the owner; see :meth:`Stoner.ROI.polygon`.

        Args:
            vertices (array-like):
                At least three non-collinear (x, y) vertices in boundary order.

        Keyword Arguments:
            coordinates (CoordinateMode or str):
                Pixel by default, or native physical coordinates.

        Returns:
            ImageFile:
                Owner with the current polygon.
        """
        return self.set(ROI.polygon(vertices, coordinates=coordinates, reference=self._target()))

    def circle(self, centre, radius, *, coordinates=CoordinateMode.PIXEL):
        """Set a closed circle and return the owner; see :meth:`Stoner.ROI.circle`.

        Args:
            centre (tuple):
                Centre in (x, y) order.
            radius (float):
                Strictly positive radius.

        Keyword Arguments:
            coordinates (CoordinateMode or str):
                Pixel by default, or native axes with equal unit labels.

        Returns:
            ImageFile:
                Owner with the current circle.
        """
        return self.set(ROI.circle(centre, radius, coordinates=coordinates, reference=self._target()))

    def ellipse(self, centre, radii, *, coordinates=CoordinateMode.PIXEL):
        """Set a closed axis-aligned ellipse and return the owner.

        Args:
            centre, radii (tuple):
                Finite (x, y) centre and strictly positive (x, y) radii.

        Keyword Arguments:
            coordinates (CoordinateMode or str):
                Pixel by default, or native axes with equal unit labels.

        Returns:
            ImageFile:
                Owner with the current ellipse.
        """
        return self.set(ROI.ellipse(centre, radii, coordinates=coordinates, reference=self._target()))

    def add(self, region):
        """Union a description with current ROI, or set it when inactive.

        Args:
            region (ROI):
                Compatible description to include in the current selection.

        Returns:
            ImageFile:
                Owner with the combined region; stale current state is rejected.
        """
        if self.stale:
            raise ValueError("Clear or reselect a stale ROI before combining regions.")
        return self.set(region if not self.active else self.current | region)

    def intersect(self, region):
        """Intersect current ROI, using the whole domain when inactive.

        Args:
            region (ROI):
                Compatible description with which to intersect current membership.

        Returns:
            ImageFile:
                Owner with the intersection; stale current state is rejected.
        """
        if self.stale:
            raise ValueError("Clear or reselect a stale ROI before combining regions.")
        return self.set((self.current if self.active else ROI.all()) & region)

    def subtract(self, region):
        """Subtract from current ROI, using the whole domain when inactive.

        Args:
            region (ROI):
                Compatible description to remove from current membership.

        Returns:
            ImageFile:
                Owner with the difference; stale current state is rejected.
        """
        if self.stale:
            raise ValueError("Clear or reselect a stale ROI before combining regions.")
        return self.set((self.current if self.active else ROI.all()) - region)

    def select_rectangle(self, *, ax=None, coordinates=CoordinateMode.PIXEL):
        """Select a box interactively; Enter accepts, Escape/window close cancels.

        Keyword Arguments:
            ax (matplotlib.axes.Axes or None):
                Borrow existing axes, otherwise create and close a temporary figure.
            coordinates (CoordinateMode or str):
                Pixel by default, or native coordinate selection.

        Returns:
            ImageFile:
                Owner; cancellation leaves its previous ROI unchanged. Callbacks
                and preview artists are removed, and borrowed targets are restored.
        """
        from .roi_widgets import select_roi

        return select_roi(self._image, "rectangle", ax=ax, coordinates=coordinates)

    def select_polygon(self, *, ax=None, coordinates=CoordinateMode.PIXEL):
        """Select a polygon interactively; Enter accepts, Escape/window close cancels.

        Keyword Arguments:
            ax (matplotlib.axes.Axes or None):
                Borrow existing axes or use a temporary figure, as for select_rectangle.
            coordinates (CoordinateMode or str):
                Pixel by default, or native coordinate selection.

        Returns:
            ImageFile:
                Owner; previous ROI is retained on cancellation or failed selection.
        """
        from .roi_widgets import select_roi

        return select_roi(self._image, "polygon", ax=ax, coordinates=coordinates)
