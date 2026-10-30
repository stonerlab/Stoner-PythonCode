Plot an image
**************

Use :py:attr:`~Stoner.Image.core.ImageFile.plot` to display an image and return
its Matplotlib Axes, consistently with ``Data.plot``::

    import numpy as np
    from Stoner import ImageFile

    image = ImageFile(np.arange(120).reshape(10, 12))
    ax = image.plot(cmap="viridis")
    image.plot.xlabel = "Column"
    image.plot.ylabel = "Row"
    image.plot.title("Measurement")
    image.plot.show()

Calling ``image.plot()`` is equivalent to ``image.plot.image()``. Rendering
does not show or save automatically and does not change intensities, exclusions
or scientific metadata. Use ``ax.figure.savefig(...)`` or
``image.plot.SavedFigure`` for saving. Inspect ``image.plot.ax``, ``figure`` and
``axes`` for recent live targets without creating or reviving a figure.

.. contents:: On this page
   :local:
   :depth: 1

Coordinates, contours and histograms
====================================

The :py:class:`~Stoner.Image.plotting.ImagePlotter` accessor supplies ``image``,
``imshow``, ``pcolormesh``, ``contour`` and ``hist``. All return the primary Axes.
Here is a complete example with stored physical coordinates and excluded pixels:

.. plot:: samples/image_plot_accessor.py
   :include-source:
   :outname: image_plot_accessor

By default, :py:class:`~Stoner.enums.CoordinateMode` ``NATIVE`` uses the image's
stored coordinate vectors. Coordinate ``long_name`` and ``units`` attributes
supply axis labels. Uncalibrated images have positional vectors. Pass
``coordinates=CoordinateMode.PIXEL`` (or ``"pixel"``) for local column/row indices.
Legacy ``MicronsPerPixel`` metadata does not silently override stored coordinates.

The optional ``x`` and ``y`` arguments name existing spatial coordinate vectors;
they are not table columns or raw arrays. Native coordinates preserve crop
offsets. Pixel mode starts at zero for the current image handle and rejects
explicit coordinate names. Array order is ``(y, x)``; spatial arguments use
``(x, y)``. Two-dimensional coordinate maps are not supported in this first
accessor version: select compatible one-dimensional vectors or use pixel mode.

The first image row appears at the top by default (``origin="upper"``); use
``origin="lower"`` for the opposite orientation. Image, mesh and contour plots
use the same coordinate locations and orientation, including decreasing axes.
Coordinates must be finite and strictly monotonic. Singleton axes use unit
pixel width in the selected coordinate units.

Uniform coordinates are drawn with ``Axes.imshow``, with extent calculated from
pixel centres and their edges. Non-uniform coordinates use ``Axes.pcolormesh``
without resampling measured pixels. Explicit ``image.plot.imshow()`` requires a
uniform grid. ``image.plot.pcolormesh()`` uses midpoint cell edges on either kind
of grid. An ``extent`` override changes only uniform-image display, never stored
calibration; mesh/contour methods reject it.

Exclusions and temporary selection
==================================

Excluded and non-finite intensities are transparent and do not determine colour
limits. Request a coloured overlay with ``mask_color="red", mask_alpha=0.4``.
The overlay shares the field's coordinates, orientation and display transform.
``cmap``, ``norm``, ``vmin`` and ``vmax`` affect display only; no stored values are
normalised or clipped. A fully excluded image can be displayed, but a histogram
or contour without eligible values raises ``ValueError``.

``where`` accepts an exact-shape Boolean array or a precisely aligned xarray
DataArray. True includes a pixel; persistent mask True excludes it. Temporary
selection combines with exclusions/finite-value checks without changing the
owner. There is no broadcasting or silent coordinate alignment::

    included = np.ones(image.shape, dtype=bool)
    included[:2, :] = False
    ax = image.plot.hist(where=included, bins=20)

The histogram counts raw eligible intensities, with native options such as
``density``, ``range``, ``histtype`` and ``orientation``. Optional weights must
match the image shape and labels, with finite unmasked participating values.
This is distinct from ``image.hist()``, which still returns a numerical ``Data``
table. Profile extraction also remains independent: a returned profile can use
``image.profile_line(...).plot()``.

Shared contexts and properties
==============================

Use the same style/layout/formatter interfaces as ``Data.plot``, including
``style``, ``new_figure``, ``MultiPanel``, ``StackVertical``, ``InsetPlot`` and
``SavedFigure``. See :doc:`plot_layouts` and :doc:`../plot-contexts`.
Supplying ``ax=`` draws on that live compatible target without creating a spare
figure. No image columns or ``labels`` list are introduced; use coordinate
metadata and explicit labels instead.

Properties resolve on the recent live Axes, then its Figure. Reads give live
callable getter/setter proxies, just as for Data::

    image.plot.xlabel = "Position / um"
    label = image.plot.xlabel()
    image.plot.title("Panel title", loc="left")
    image.plot.title.get(loc="left")
    image.plot.dpi = 120

Contexts restore their documented caller state on exit. Plot target state is
transient: cloning/pickling an image does not capture live Figure/Axes objects.
The source image remains unchanged by rendering, including hidden raw values.

Migrating from legacy display
=============================

``image.imshow()`` retains its existing Figure return and options. The new
``image.plot()`` and ``image.plot.imshow()`` return Axes. Translate a Figure
consumer explicitly::

    # Legacy: figure = image.imshow()
    ax = image.plot()
    figure = ax.figure

Legacy ``figure=``/``showfig`` and table ``xcol``/``ycol`` options are rejected on
the accessor. Existing widgets and Figure-dependent image display helpers retain
their legacy paths. ImageFolder/ImageStack bulk plotting has not been extended
by this first accessor implementation; an individual frame/region ImageFile can
use its own accessor. ROI management and image fitting remain planned additions,
not methods currently available on ImageFile.
