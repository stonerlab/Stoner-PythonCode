Images, surfaces and vector fields
**********************************

.. currentmodule:: Stoner

Use these methods when each measured value belongs to a position in two or
three dimensions. Start by choosing how you want to display that value::

    d.plot.image(x="X", y="Y", z="Signal")      # A colour image.
    d.plot.surface(x="X", y="Y", z="Signal")    # A raised 3D surface.
    d.plot.contour(x="X", y="Y", z="Signal")    # Lines of equal value.

These examples need a table with X, Y and Signal columns. Each row describes
one measured position. Use your own column names; the two-column starter
measurement in :doc:`plotfile` is intended for line plots instead.

The worked examples below show how to build suitable tables. Read the gridding
section when your measurement positions do not already form a regular grid.

.. contents:: On this page
   :local:
   :depth: 1

.. _plotting-fields:

Scalar, vector and volume fields
================================

Field methods return their **primary Matplotlib Axes**, accept explicit ``ax=``
and inherit the caller's style. They do not display or save figures. An explicit
target must have the correct 2D/3D projection; an incompatible implicit current
axes causes a new figure to be created. No field method imports or selects Mayavi.

.. list-table::
    :header-rows: 1
    :widths: 25 30 45

    * - Method
      - Inputs
      - Rendering
    * - ``surface``
      - X, Y, scalar Z
      - Gridded 3D surface.
    * - ``contour``
      - X, Y, scalar Z
      - 2D contours; ``filled=True`` selects filled contours.
    * - ``pcolormesh`` / ``image``
      - X, Y, scalar Z
      - Pseudocolour cells / an image with physical pixel-centre coordinates.
    * - ``scatter3d``
      - X, Y, Z
      - Ungridded 3D points, including per-sample ``c`` and ``s`` arrays.
    * - ``vector``
      - X, Y, U, V, optional W
      - Directional RGB image with optional U/V arrows.
    * - ``quiver``
      - X, Y, U, V
      - Ungridded 2D arrows; unused Z/W roles do not affect validity.
    * - ``vector3d``
      - X, Y, Z, U, V, W
      - Matplotlib 3D arrows coloured by vector direction by default.
    * - ``voxels``
      - X, Y, Z, scalar U
      - A scalar volume with physical cell corners and visibility controls.
    * - ``matrix``
      - Table block and X/Y coordinate vectors
      - 3D surface; data rows index X and data columns index Y.
    * - ``waterfall``
      - X, grouping Y, response Z
      - Ungridded grouped traces using the Joy Division utility.

A named method and ``d.plot(kind="surface", ...)`` use the same implementation.
The automatic call retains the scientific role/default choices: XYZ selects a
surface, XYUV[W] a vector image, XYZUVW a 3D vector plot, and XYZU voxels. Unassigned
three-column automatic calls still infer XY with Y uncertainty. Explicit field
methods on entirely unassigned tables instead take their required inputs in the
listed order: ``d.plot.surface()`` uses columns 0/1/2 and ``d.plot.quiver()`` uses
0/1/2/3. Explicit selectors override those defaults; ``no_guess=True`` requires
roles or explicit selectors. For assigned tables, supply missing field selectors
explicitly. Named methods can take their listed coordinates positionally, while
the automatic call still accepts only X/Y positionally::

    ax = d.plot.surface(x="Field", y="Temperature", z="Response", cmap="viridis")
    ax = d.plot.scatter3d(x=0, y=1, z=2, color="black")
    other.plot.surface(ax=ax, alpha=0.4, colorbar=False)

Gridding and exclusions
-----------------------

Scalar grids use SciPy's existing ``griddata`` interpolation engine, with
``method="linear"`` by default; ``nearest`` and 2D ``cubic`` are also available.
``shape=(nx, ny)`` gives inclusive centre counts in X/Y order; volumes use
``(nx, ny, nz)``. Default counts equal the number of distinct coordinates along
each axis, matching the previous grid-count policy. ``xlim``, ``ylim`` and volume
``zlim`` select increasing grid-centre bounds. Ungridded methods instead use limits
as ordinary axes limits and reject gridding arguments.

Measurements already on a complete regular grid are used without interpolation. Two-dimensional renderer arrays
have Y rows and X columns; voxel arrays use X/Y/Z indexing. A sample is used only when all the columns needed for that plot are unmasked
and finite. Excluded/non-finite scalar or vector samples do not
participate in interpolation; an excluded measured site coinciding with an output
grid centre stays missing. Other unsampled sites can be interpolated, and
linear/cubic values outside the convex hull remain missing. Duplicate positions
use their first valid sample. Measurements, exclusions and roles are unchanged.
Unlike the old surface helper, missing values are never replaced with zero.

``image`` retains scalar values for normalisation; it does not clip them by
pre-converting them to RGB. Its default ``origin="lower"`` maps the first row to
the smallest Y. Image extents extend half a cell beyond the outer centres. A
single centre spans one coordinate unit. Scalar fields use ``cmap``/``norm`` or
``vmin``/``vmax`` and add a colour bar by default. Use ``colorbar=False`` for an
overlay or when managing the colour bar yourself. A solid-colour surface has no
scalar colour bar. Surface and matrix colour maps otherwise follow the current
Matplotlib ``image.cmap`` setting.

.. plot:: samples/field_plots.py
    :include-source:
    :outname: fields

.. plot:: samples/3D_plot.py
    :include-source:
    :outname: 3D

Vector colours and arrows
-------------------------

``vector`` reuses the established HSL mapping: hue follows ``atan2(U,V)``,
saturation follows relative vector magnitude, and optional W controls luminance
across its range. Constant W has neutral luminance; an all-zero field is finite
neutral grey. The hue boundary now correctly renders red. These directional RGB
colours have no scalar colour bar; use a scalar plotting method for a numerical
colour scale. ``arrows=False`` suppresses the overlay; ``quiver_kwargs={...}``
supplies arrow styling separately from image options.

To smooth the displayed colour background, pass ``interpolation="bilinear"``
or ``"bicubic"`` to the underlying image renderer. For a finer colour grid,
combine this with ``shape=(nx, ny)`` and ``method="linear"`` or ``"cubic"``::

    ax = d.plot.vector(shape=(300, 300), method="cubic", interpolation="bicubic")

``shape`` and ``method`` interpolate the **RGB colours**, not the vector
components. A complete grid is otherwise used directly, so changing ``method``
alone need not change it. ``interpolation`` controls how those colour pixels
are displayed. Neither option is a statistical smoothing filter. The arrows
retain the original valid sample locations and components, and the measurements
are unchanged. Use ``interpolation="nearest"`` to show the colour-grid pixels.

At duplicate positions, the vector image and its overlay use the same first
valid vector. The ungridded ``quiver`` and ``vector3d`` methods retain every valid
sample. ``waterfall`` intentionally applies the Joy Division helper's black
background and white traces; use ``bg_color``, ``color`` and ``axes_color`` to
change these colours.

``quiver`` and the vector overlay default to ``pivot="mid"``, ``angles="xy"``,
``scale_units="xy"`` and ``scale=1``: arrow directions and lengths are expressed
in data coordinates. Override these native Matplotlib options when the vector
components have different units from the spatial coordinates. ``vector3d`` passes
native 3D quiver options such as ``length``, ``normalize`` and
``arrow_length_ratio``. All selected component masks follow the rendered arrows.

.. plot:: samples/Vectorfield.py
    :include-source:
    :outname: vector_field

.. plot:: samples/3D_vector_plot.py
    :include-source:
    :outname: vector3d

Matrices and voxels
-------------------

``matrix`` has a separate interpretation from a row-wise XYZ table. By default,
the first column supplies X coordinates and the remaining numeric column headers
supply Y. The selected data block has shape ``(nx, ny)`` and is transposed once
for Matplotlib's Y/X mesh. This works for non-square blocks and preserves missing
cells. ``x`` can instead be a selector or explicit vector, and ``y`` a coordinate
row index or explicit vector. A coordinate row starts the default block below it.
``rect=(row, column[, rows, columns])`` crops the block, with corresponding
coordinate slices. Coordinate vectors must be finite and unexcluded.

Unlike ``vector``/``image``, ``matrix`` renders a 3D surface rather than an image.
It does not accept image ``interpolation`` or the field-gridding ``shape`` and
``method`` options. Its native ``rcount`` and ``ccount`` options control how many
existing samples Matplotlib renders, rather than interpolating new values.
Set ``rcount=ny, ccount=nx`` to draw the whole ``(nx, ny)`` matrix without
downsampling. ``linewidth=0, antialiased=True`` removes mesh lines and antialiases
polygon edges, but does not smooth or resample the measurements. For numerical
interpolation, represent the data as a row-wise XYZ table and use
``plot.surface(shape=(nx, ny), method="cubic")`` instead.

.. plot:: samples/matrix_plot.py
    :include-source:
    :outname: matrix

``voxels`` treats X/Y/Z as cell centres and U as intensity. Cell corners lie
halfway between centres, extending the outer cells by half a spacing. A singleton
coordinate spans one unit. ``visible=lambda x, y, z: ...`` evaluates a Boolean
predicate on **grid centres**; alternatively, ``filled`` supplies a Boolean array
with the gridded shape. Both combine with finite/unexcluded values. The colour
bar and voxel faces share one scalar normalisation. ``face_alpha`` and
``edge_alpha`` set the mapped opacity (defaults 0.5 and 0.9).

.. plot:: samples/voxel_plot.py
    :include-source:
    :outname: voxel2

Next steps
==========

See :doc:`plot_formatting` for labels and saving, or :doc:`plot_layouts`
for panels. Return to :doc:`plotfile` to choose another plotting task.
