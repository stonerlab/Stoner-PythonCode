Select image regions
********************

Use :py:attr:`~Stoner.Image.core.ImageFile.roi` to manage a region of interest
without changing pixels or the persistent image mask::

    import numpy as np
    from Stoner import ImageFile, ROI

    image = ImageFile(np.arange(120.).reshape(10, 12))
    image.roi.box(2, 8, 1, 6)
    selected_mean = image.mean()
    ax = image.plot()
    image.plot.roi(ax=ax, color="red")

Setting and clearing a region returns the image, so
``image.roi.box(2, 8, 1, 6).plot()`` is valid. Merely looking up ``image.roi``
creates no selection, figure or pixel copy. Inspect ``image.roi.active``,
``current`` and ``stale``; do not use ``bool(image.roi)``. An active region
can contain no pixels, which is different from having no current ROI.

.. contents:: On this page
   :local:
   :depth: 1

Membership, exclusions and operation defaults
=============================================

ROI membership uses **True to include**, whereas ``image.mask`` uses True to
exclude. ``image.roi.resolve()`` returns an independent Boolean xarray DataArray
with the image's labels. Membership is purely spatial: excluded pixels and
non-finite intensities can be inside an ROI. Scientific consumers separately
combine membership with exclusions and finite-value checks.

The reviewed consumers in this alpha are ``image.mean()`` and
``image.plot.hist()``. Their selection rules are:

* Omit ``where`` or pass ``None`` to use the current ROI, or the whole domain
  if there is no active ROI.
* Pass ``where=region``, an exact-shape Boolean array or an exactly aligned
  DataArray to override current state for that operation alone.
* Pass ``where=ROI.all()`` to explicitly use the whole domain. Exclusions and
  invalid intensities still do not participate.

An empty current ROI stays empty: mean returns NaN, including empty positions
in axis reductions; histogram raises ValueError when no eligible pixels remain.
Explicit overrides can bypass a stale ROI and never replace it.

Image display, mesh and contour methods still show the whole image by default.
Their explicit ``where`` accepts the same descriptions, but a current ROI never
implicitly crops or hides context. Draw its outline with ``image.plot.roi()``;
it returns Axes, accepts an explicit ``ax``, and preserves existing axis limits.
Outlines follow exact pixel edges, including holes and the domain boundary.
Match their ``coordinates`` choice to the base image.

Other reductions, processing methods, saving and legacy mask selectors retain
their existing semantics. ROI awareness is deliberately limited to the reviewed
consumers. ``Data`` has no ROI accessor and its selection defaults are unchanged.

Reusable descriptions, shapes and combinations
===============================================

:py:class:`Stoner.core.roi.ROI` builds immutable descriptions without modifying
an image. The :py:class:`~Stoner.Image.roi.ImageROI` accessor sets current state::

    left = ROI.box(0, 3, 0, 10)
    right = ROI.box(9, 12, 0, 10)
    background = (left | right) - ROI.circle(centre=(1, 4), radius=1)
    image.roi.set(background)
    image.roi.add(ROI.box(4, 6, 0, 2))
    image.roi.intersect(ROI.box(0, 12, 0, 8))
    image.roi.subtract(ROI.box(0, 1, 0, 1))
    mean_background = image.mean()
    mean_everywhere = image.mean(where=ROI.all())

``|`` is union, ``&`` intersection, ``-`` difference and ``~`` complement against
the target image's domain. Disjoint patches and holes are supported. ``add``
sets its argument when inactive; ``intersect`` and ``subtract`` start with the
whole domain when inactive. Combining a stale current selection is an error.

Builders and accessor setters include ``box``, ``polygon``, ``circle``,
``ellipse``, ``from_mask`` and ``all``. An ellipse is axis-aligned and takes
``centre=(x, y), radii=(rx, ry)``. Circle/ellipse radii must be strictly positive;
polygons require at least three non-collinear vertices. Shape edges are included.
No geometry is rounded to integer coordinates before testing pixel centres.

``image.roi.from_mask(included)`` captures an independent exact Boolean mask.
For a reusable description use
``ROI.from_mask(included, reference=image.image.intensity)``. Numeric masks,
missing values, broadcasting, reversed labels and automatic reindexing are
rejected. A mask description is tied to its captured grid and coordinate labels,
including units; it cannot silently move to another calibration or crop.

Coordinates and boundaries
===========================

Programmatic and interactive ROI constructors default to
:py:class:`~Stoner.enums.CoordinateMode` ``PIXEL``, or ``"pixel"``. Spatial
arguments use ``(x, y) = (column, row)``; array shape is ``(y, x)``.
Pixel indices start at zero for the current handle. Boxes have lower-inclusive,
upper-exclusive bounds: ``ROI.box(2, 8, 1, 6)`` corresponds to membership in
``array[1:6, 2:8]``. Outside bounds intersect the image domain; negative values
never wrap around. Equal bounds select an empty region; reversed and non-finite
bounds are errors. Legacy crop's fractional/negative/metadata-box parser remains
separate and unchanged.

For physical coordinates, request native mode explicitly::

    image.roi.box(-2, 2, -1, 1, coordinates="native")
    physical_region = ROI.box(-2, 2, -1, 1, coordinates="native",
                              reference=image.image.intensity)

The accessor captures its owner's x/y unit labels automatically. Standalone
native builders require a reference DataArray or intensity Dataset. The
description retains units, not the reference image. Reuse on different separable
grids with the same units is supported. Unitless axes are permitted, but unit
labels must match exactly; no automatic conversion occurs. Native circle/ellipse
axes must have equal unit labels. Coordinates are finite and strictly monotonic;
decreasing, anisotropic and non-uniform grids are evaluated by their actual
centre values. Native crop offsets are retained. Two-dimensional coordinate maps
are not supported.

Pixel and native geometric descriptions cannot be combined directly. Resolve
each against an explicit reference and capture it with ``ROI.from_mask`` before
combining if both conventions are required.

An executable calibrated workflow
=================================

This example selects two background patches, replays their description and
compares selected and whole-image histograms without altering the measurement:

.. plot:: samples/image_roi_accessor.py
   :include-source:
   :outname: image_roi_accessor

State, copies, geometry changes and replay
==================================================

``image.roi.clear()`` removes current state. It does not unmask pixels, undo a
crop or restore a corrected image. ``image.roi.all()`` sets an explicit active
whole-domain description. Saved accessors follow current owner state; saved
descriptions retain their geometry.

Value-only edits preserve current ROI. Clone, deepcopy and pickle retain
independent selection state. Parent images, cropped handles and individual stack
frame handles have separate state, even where pixels are shared. Keep the frame
handle if you want to reuse its current ROI; obtaining a new ``stack[index]``
handle does not recover another handle's selection. There is no folder/stack-wide
current ROI in this alpha.

Crops, slices, permutations, rotations, resampling and registration conservatively
mark inherited active ROI state stale. Shape-changing replacement also invalidates
it. ``active`` remains True, and operations that use current ROI fail explicitly
until clear or reselection. No stale region silently becomes the whole image.
Choose an explicit description again only after reviewing the changed geometry.
Independent image-storage imports start without current ROI.

Operational selection is outside scientific metadata and ImageStorage v1.
Ordinary image save/load does not persist it. Explicit JSON replay is available::

    import json

    text = json.dumps(image.roi.current.to_dict())
    restored_region = ROI.from_dict(json.loads(text))
    image.roi.set(restored_region)

Records have version 1; unknown fields, versions and inconsistent geometry are
rejected. Mask records retain exact labelled geometry. Coordinate metadata used
for ROI state/replay must be JSON-compatible. Replay restores a description, not
GUI objects or an implicit change to an image's exclusions.

Interactive selection
======================

Interactive methods are explicit and intended for a GUI Matplotlib backend::

    image.roi.select_rectangle()
    image.roi.select_polygon(coordinates="native")
    # An existing live axes can be borrowed:
    image.roi.select_rectangle(ax=ax)

Drag a rectangle, or complete a polygon using Matplotlib's polygon selector, then
press Enter to accept. Escape or closing the window cancels and keeps the old
ROI. Temporary figures and owned preview artists/callbacks are removed; supplied
axes, presentation and the previous plotting target are restored. Native and
pixel previews use the shared image coordinate service. An invalid selection
never replaces current state. Existing ``RegionSelect``, ``ShapeSelect`` and
``image.mask.select`` retain their legacy behaviour.

The regression suite simulates acceptance, cancellation, closing and backend
failures. Live GUI interaction and backend-specific behaviour still require
manual verification; headless checks are not a claim of manual GUI coverage.

Next: image fitting
====================

The subsequent ImageFile.fit alpha will use this same resolver for fitting plane
and quadratic backgrounds to reference regions, then applying corrections over
the whole image. It is not part of this ROI alpha; there is no released
``image.fit`` API yet. General processing within an ROI and rectangular ROI
extraction conveniences remain separate future work; existing explicit crop
continues to work.
