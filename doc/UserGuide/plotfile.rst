*************
Plotting Data
*************

.. currentmodule:: Stoner

The :py:attr:`~Stoner.core.data.Data.plot` property returns a callable
:py:class:`~Stoner.plot.accessor.DataPlotter`. Use it to infer a scientific plot,
or select a particular plot through its methods. Only the normal
Stoner import is needed::

    from Stoner import Data

    d = Data("sample.txt", setas="xye")
    ax = d.plot()
    ax.set_title("My measurement")
    ax.figure.savefig("measurement.pdf")

Drawing returns the Matplotlib ``Axes`` and does not show or save it. Notebook
display follows Matplotlib's normal behaviour; elsewhere use ``d.plot.show()``.
The removed ``showfig`` setting has no replacement: returns are consistent.
``Options.no_figs`` does not affect accessor returns or suppress drawing.

.. autoattribute:: Stoner.core.data.Data.plot

.. plot:: samples/single_plot.py
    :include-source:
    :outname: single

Automatic and explicit plotting
===============================

Explicit selectors override assigned :ref:`setas` roles. Unassigned tables retain
these column-count guesses: two columns mean XY, three mean XY and Y uncertainty,
and four mean X, X uncertainty, Y and Y uncertainty. Five/six columns retain the
existing XYUVW/XYZUVW vector-field guesses. Three unassigned columns do not mean
a surface. Use ``no_guess=True`` to disable column-count guesses.

The ordinary methods are:

* ``d.plot()``: infer lines or error bars from resolved coordinates/uncertainties;
* ``d.plot.line()``: plain lines, ignoring assigned uncertainties;
* ``d.plot.scatter()``: points drawn with Matplotlib's scatter method;
* ``d.plot.errorbar()``: data and inferred/explicit uncertainties;
* ``d.plot.errorfill()``: lines with shaded uncertainty in one direction.

``d.plot(kind="scatter", ...)`` and ``d.plot.scatter(...)`` share an
implementation. Named ordinary methods override field roles, so
``d.plot.line(x=0, y=1)`` works on an XYZ table. Scalar, vector and volume fields
also have named accessor methods, as do histogram, box, KDE and hexbin plots.
These choices are described below; the automatic call keeps its scientific
role/count dispatch.

.. plot:: samples/plot_accessor.py
    :include-source:
    :outname: accessor

Columns, uncertainties and exclusions
=====================================

Use ``x`` and ``y`` for selectors. Y can select several columns sharing one axes.
Exact headers, compiled regular expressions, slices, lists and integer positions
follow Stoner's existing rules, including duplicate-header handling. ``startx``
selects the assigned X group at or after that column position::

    ax = d.plot(x="Field", y=["Signal A", "Signal B"])
    ax = d.plot.errorbar(y=[3, 1])  # errors follow their assigned Y columns
    ax = d.plot.line(0, 1, fmt="r-")

Only X and Y are positional in the automatic call. Supply ``fmt`` as a keyword:
``d.plot(0, 1, "r-")`` is no longer accepted. ``fmt`` and ``label`` can be lists
with one entry per selected Y column. Scatter uses ``marker`` and ``color``
instead of ``fmt``.

Error specifications have these meanings:

* ``None`` infers assigned/guessed errors; ``False`` disables them.
* An integer, string or compiled pattern selects one uncertainty column.
* A floating scalar supplies constant uncertainty, for example ``yerr=0.2``.
* NumPy arrays provide values of shape ``(N,)`` or asymmetric errors ``(2, N)``.
* A Y list/tuple matching the trace count supplies specifications per trace. Use
  an array to avoid ambiguity with a list of column selectors.

Explicit coordinates can still inherit column-count uncertainty guesses; use
``kind="line"`` for a plain line. Plain line/scatter calls reject explicit errors
and direct you to an uncertainty method. For example::

    ax = d.plot.errorbar(xerr=False, yerr=0.2, fmt="o", capsize=3)
    ax = d.plot.errorfill(yerr=0.2, xerr=False, color="orange", alpha_fill=0.3)

Shaded errors support one direction per trace. Disable the other explicitly when
both roles are assigned; the accessor does not silently discard one direction.
Exclusions and non-finite values are applied independently to each trace and its
errors. Excluded points are removed, so lines connect remaining points. Unrelated
column masks do not remove samples from a selected trace. Invalid error shapes,
negative uncertainties at plotted samples, and empty/all-excluded requests raise
an error before drawing. Measurements, masks, roles and metadata are unchanged.

The raw ``d.data.plot()`` interface uses pandas semantics and does not apply
Stoner's separate exclusions. Use the accessor's statistical methods for
mask-aware distributions and binning.

.. _plotting-statistics:

Distributions and hexagonal binning
===================================

``hist``, ``box``, ``kde`` and ``hexbin`` use Matplotlib directly, with SciPy's
``gaussian_kde`` for density estimation. Each returns its primary Axes, respects
``ax=`` and current styling, and does not show/save. Direct drawing keeps the
selection/mask handling small and avoids an intermediate DataFrame or pandas
backend. Changing pandas' global plotting backend does not affect these methods.

For ``hist``, ``box`` and ``kde``, ``y`` selects sample columns rather than XY
pairs. Omitting it selects assigned Y columns in the current ``startx`` group;
if none exist, all real numeric columns are selected. A single-column table
therefore works. ``no_guess=True`` disables that numeric fallback. Explicit
selectors use Stoner's normal positional/header/pattern/slice rules and preserve
selection order, including duplicate-header behaviour::

    ax = d.plot.hist(y=[1, 3], bins=30, histtype="step")
    ax = d.plot(kind="hist", y=[1, 3], bins=30, histtype="step")
    ax = d.plot.box(y=[1, 3], orientation="horizontal")
    ax = d.plot.kde(y=1, bw_method="silverman", ind=300)

Each sample column removes its own excluded/non-finite values independently.
Masks in unselected X/error columns do not remove distribution samples;
no separate X coordinate is read for alignment. Every selected series must retain
samples, otherwise the call raises before creating axes. Integer samples are filtered without inserting
NaNs into the measurement table. All methods leave values, exclusions, metadata,
roles and storage ownership unchanged.

``hist`` forwards native options such as ``bins``, ``range``, ``density``,
``cumulative``, ``histtype`` and ``stacked``. Selected series share bin edges.
``stacked=True`` adds histogram values; use ``StackVertical`` to arrange panels.
For histograms and KDEs, ``weights`` is one column selector or an array with one
entry per original row. Invalid weights combine with each series' exclusions,
preserving alignment independently. Histograms retain native support for signed
weights. KDE weights must be non-negative with positive total weight.

``box`` passes independent sample arrays to Matplotlib; it has no weights or
group-by forwarding. ``label`` supplies category labels and ``orientation`` is
``"vertical"`` or ``"horizontal"``. Native controls such as ``whis``,
``showfliers``, ``notch``, ``positions`` and artist-property dictionaries remain
available. The accessor handles the native orientation/tick-label keyword names
across the supported Matplotlib versions.

``kde`` requires at least two distinct samples with positive weight per series.
``bw_method`` uses SciPy's factor/name/callable convention; None uses Scott's
rule. ``ind`` is an integer of at least two points or a finite evaluation vector.
By default each series is evaluated at 1000 points extending half its observed
range beyond each end; zero-weight samples do not determine that range. Gaussian
kernels extend beyond the measured range, including physical bounds. Remaining
styling arguments go to ``Axes.plot``.

Distribution ``label`` values default to column labels; explicit ``None``
suppresses legend/category labels. Histograms and KDEs update their legend by
default; ``legend=False`` disables that update. ``xlabel``, ``ylabel`` and
``title`` retain source/metadata interpolation and existing text unless explicitly
overridden; ``None`` clears an annotation. Unsupported ``backend``, ``by``,
``subplots``, ``layout`` and ``secondary_y`` options raise an error; use the
stonerplots contexts for layout and axes management.

``hexbin`` uses one X and one Y column, with the ordinary scientific coordinate
defaults. ``C=None`` counts jointly valid points; a column selector or source-row
array supplies values to aggregate with ``reduce_C_function`` (mean by default).
Only X/Y and optional C determine joint validity. Assigned uncertainties and
unrelated masks are ignored. Native ``gridsize``, ``mincnt``, ``extent``,
``bins``, ``cmap``, ``norm`` and ``xscale``/``yscale`` controls are supported,
including logarithmic binning. ``colorbar=True`` adds a colour scale without
replacing the primary recent target::

    ax = d.plot.hexbin(x=0, y=1, gridsize=25, mincnt=1)
    ax = d.plot.hexbin(x=0, y=1, C=3, reduce_C_function=sum, colorbar=False)

.. plot:: samples/statistical_plots.py
    :include-source:
    :outname: statistics

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

Complete regular lattices are reordered directly. Two-dimensional renderer arrays
have Y rows and X columns; voxel arrays use X/Y/Z indexing. Only required columns
contribute to joint validity. Excluded/non-finite scalar or vector samples do not
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

Figures, axes and formatting
============================

Ordinary plots use explicit ``ax=`` or Matplotlib's current compatible axes. They
create a figure only when needed; a new Data object does not imply a new figure.
An incompatible implicit projection creates a suitable figure, while an
incompatible explicit axes raises an error::

    ax = d.plot()
    other.plot.line(x=0, y=1, ax=ax)
    figure = d.plot.new_figure(figsize=(7, 4))
    ax = d.plot.line()

``d.plot.ax``, ``d.plot.figure`` and ``d.plot.axes`` inspect recent live targets
without selecting or creating a figure. Obtain the Figure from ``ax.figure``.
Copy, deepcopy, clone, construction from another Data, and pickle round trips
preserve label configuration while detaching live figures/axes.

Drawing inherits current rcParams and existing axes styling. It does not reset
styles or apply a legacy template. Use the returned Axes for formatting::

    ax = d.plot(fmt="o", xlabel="Field / T", ylabel="Signal", title="Run 1")
    ax.set_xscale("log")
    ax.set_ylim(0, 10)
    ax.legend(loc="upper left")

``xscale`` and ``yscale`` are also accessor arguments. ``d.plot.labels`` provides
mutable column-label overrides, defaulting to column headers. Plain text labels,
titles and styling strings retain metadata ``{name}`` interpolation; strings
containing ``$`` retain their mathematical markup.

Omitted annotations fill empty labels/titles; subsequent traces do not overwrite
existing text. Explicit values override it; ``None`` clears it. ``label=None``
suppresses that series' legend entry and ``legend=False`` stops automatic legend
updates. User formatters and locators are left alone.

.. _plotting-contexts:

Layouts and styles with one Stoner import
=========================================

Stoner requires stonerplots 1.9.6 or newer. Its context classes are available
directly from ``d.plot`` and :py:mod:`Stoner.plot`. They have the upstream
signatures and return their native handles; Stoner does not wrap their behaviour.

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Context
      - Purpose
    * - ``SavedFigure``
      - Apply styles and save figures created in the block, or an existing Figure with ``use=figure``.
    * - ``MultiPanel``
      - Regular or irregular panels; iterate its native sequence or index its axes.
    * - ``StackVertical``
      - Stacked panels with shared X coordinates and optional joined boundaries.
    * - ``DoubleYAxis``
      - Independent right-hand Y scale with combined legend handling.
    * - ``InsetPlot``
      - Inset axes positioned within their parent.
    * - ``CentredAxes``
      - Position axes through a specified origin.
    * - ``PlotLabeller``
      - Apply tick formatters/locators to axes created inside the block.

Use explicit ``ax=`` to make each target clear. Indexing/iterating panel handles,
insets and double-axis blocks also select pyplot's current axes, so ordinary
``d.plot()`` calls work in those contexts. Start a new Figure before constructing
a layout when the current figure already contains plots::

    from Stoner import Data

    d = Data("sample.txt", setas="xyy")
    with d.plot.SavedFigure("measurement.pdf", style=["stoner", "aps"], formats="pdf"):
        figure = d.plot.new_figure()
        with d.plot.StackVertical(2, figure=figure) as panels:
            d.plot.line(y=1, ax=panels[0], title=None)
            d.plot.line(y=2, ax=panels[1], title=None)

``SavedFigure(filename=False, ...)`` applies style/formatting without saving.
For an existing figure use ``SavedFigure(..., use=ax.figure)``; set
``formats="png"`` or another format explicitly when needed. Figures stay open
unless ``autoclose=True``. Saving is an explicit context action; ordinary drawing
still does not save or display anything.

``d.plot.style(...)`` is Matplotlib's native style context. It accepts a style
name, list, style file or rcParams dictionary and restores rcParams on exit,
including when the block raises an exception. Stonerplots registers the ``stoner``
base style and journal modifiers such as ``aps``, ``aip`` and ``nature``. Styles
affect artists as they are created; configure existing axes explicitly when
changing their properties. No style is applied merely by calling ``d.plot()``.

``PlotLabeller`` formats newly created axes by default. Use
``include_open=True`` to include existing axes in all open figures, or configure
a selected existing Axes directly. Supply formatter/locator classes or configured instances: stonerplots
creates independent tick objects per axis. ``d.plot.TexFormatter`` and
``d.plot.TexEngFormatter`` expose the upstream formatter classes::

    with d.plot.PlotLabeller():
        ax = d.plot.line(y=1)
    ax.yaxis.set_major_formatter(d.plot.TexEngFormatter(unit="V"))

Inset and double-axis contexts restore the previous target, and the inset handle
remains valid after repositioning. Layout contexts restore their entry target.
``SavedFigure`` and ``PlotLabeller`` scan tracked figures at entry/exit and can change
pyplot's current figure; pass ``ax=`` on subsequent draws when that matters.
They do not create an extra figure merely to perform formatting.

.. plot:: samples/plot_contexts.py
    :include-source:
    :outname: contexts

Migration of ordinary calls
===========================

.. list-table::
    :header-rows: 1
    :widths: 45 55

    * - Previous call
      - Current accessor call
    * - ``d.plot_xy(0, 1, "r-")``
      - ``d.plot.line(0, 1, fmt="r-")`` or automatic ``d.plot(0, 1, fmt="r-")``
    * - ``d.plot(xcol=0, ycol=1)``
      - ``d.plot(x=0, y=1)``
    * - ``d.plot(figure=False)``
      - ``d.plot.new_figure(); d.plot()``
    * - ``d.plot(figure=existing_figure)``
      - ``d.plot(ax=existing_axes)``
    * - ``d.plot(plotter=errorfill, ...)``
      - ``d.plot.errorfill(...)``
    * - ``d.plot(plotter=pyplot.semilogy, ...)``
      - ``d.plot(yscale="log", ...)``
    * - conditional Figure/Data/None return
      - always the primary Axes; use ``ax.figure`` or retain ``d`` explicitly

The ordinary accessor rejects ``figure``, ``template``, ``plotter``, ``multiple``,
``show_plot``, ``save_filename`` and old ``*col`` keywords. The former top-level
plotting methods have been removed.

Layout and style migration examples
====================================

Ordinary multiple-Y plotting replaces ``multiple="common"``. Use
``StackVertical`` for ``multiple="panels"``, ``MultiPanel`` for
``multiple="subplots"``, and ``DoubleYAxis`` for ``multiple="y2"``. Replace
``d.inset(...)`` with ``d.plot.InsetPlot(ax=parent, ...)``. For an independent
second X scale use the returned Axes' ``twiny()``; stonerplots 1.9.6 does not
provide a ``DoubleXAxis`` context. Transformed units use Matplotlib's
``secondary_xaxis``/``secondary_yaxis``.

.. plot:: samples/common_y_plot.py
    :include-source:
    :outname: common_y

.. plot:: samples/double_y_plot.py
    :include-source:
    :outname: double_y

.. plot:: samples/inset_plot.py
    :include-source:
    :outname: inset

.. plot:: samples/multiple_y2_plot.py
    :include-source:
    :outname: multiple_y

.. plot:: samples/multiple_panels_plot.py
    :include-source:
    :outname: panels

.. plot:: samples/subplot_plot.py
    :include-source:
    :outname: subplots

.. plot:: samples/template.py
    :include-source:
    :outname: template

Template classes are replaced by rcParams/style contexts and explicit Axes
formatting. The following examples use sketch settings and colour/line cycles.
The old PRB template and stonerplots ``aps`` style have different appearances;
``aps`` is the supported replacement rather than an equivalent alias. The GB/JTB
examples now show explicit centred axes and tick/line configuration; Seaborn uses
its native style and plotting contexts.

.. plot:: samples/template2.py
    :include-source:
    :outname: template2

.. plot:: samples/plotstyles/customising.py
    :include-source:
    :outname: customising

Removed helpers and field migration
====================================

Use ``d.plot.surface`` instead of ``plot_xyz``; ``scatter3d`` replaces
``plot_xyz(plotter="scatter", griddata=False)``. Use ``contour``, ``pcolormesh``,
``image``, ``matrix``, ``vector``, ``quiver``, ``vector3d`` and ``voxels`` in place
of their old top-level field helpers. ``waterfall`` replaces the custom
``plotter=joy_division`` path. Old ``griddata``, ``projection``, ``plotter`` and
backend options are rejected by the accessor: select the required named method.
The unused Mayavi installation extra has been removed.

The top-level plotting methods, PlotMixin, template classes and broad pyplot/
Figure/Axes proxies have been removed. Replace ``d.fig`` with ``ax.figure``,
``d.labels`` with ``d.plot.labels``, and formatting attributes such as ``d.title``
with ``ax.set_title(...)``. Use the Figure's ``subplots``/``add_subplot`` methods
or stonerplots contexts for layouts. An independent X scale uses ``ax.twiny()``;
an independent Y scale uses ``d.plot.DoubleYAxis``. Styles now come from the
required stonerplots dependency, rather than bundled template stylesheets.

Scientific ``d.griddata`` and shared image utilities remain independently callable;
the numerical gridding API and image-display classes retain their separate rules.

.. _plotting-folders:

Folder pages and bulk plotting
==============================

``PlotFolder.plot`` draws members serially on independent pages arranged with
stonerplots ``MultiPanel``. It returns one primary Axes per member, in folder
order, and ``folder.figures`` lists recent live pages. ``plots_per_page`` must be
a positive integer; an empty folder returns an empty list and creates no figure.
Unused page slots are removed. Single-panel, portrait and landscape layouts work,
and each target uses its member's required 2D/3D projection. Auxiliary colour-bar
axes are separate from the returned primary axes.

If drawing or a callback fails, the failed page is closed and its members' prior
plotting targets are restored. Completed earlier pages remain in ``folder.figures``.
Callbacks can deliberately change measurements; those changes are not rolled back.

``figsize``, ``dpi``, ``facecolor``, ``edgecolor``, ``frameon``, ``FigureClass`` and
``layout`` configure page figures. Other arguments go to each ``Data.plot`` call,
including ``kind``. ``folder.figure(**kwargs)`` creates an independent figure and
remembers its figure options for later pages, without assigning member targets.
``plot_extra(i, j, data)`` runs after each draw; both member index ``i`` and page
slot ``j`` are zero-based, and the target is ``data.plot.ax``.

``folder.each.plot(...)`` calls each member's automatic accessor;
``folder.each.plot.line(...)`` and the other named drawing methods are supported.
The equivalent dotted string form is ``folder.each("plot.line", ...)``. All these
plotting calls run serially even when global multiprocessing/threading is enabled,
and return results in folder order. Default calls do not write diagnostic
function/argument metadata or collect accessors. Explicit ``_return="name"``
retains the normal each-result storage contract. Generic user callables that draw
plots should pass ``_serial=True`` themselves.

Use ``ax=`` on bulk calls to overlay members on one explicit target. A paginated
``folder.plot`` chooses its own targets and rejects ``ax``/``figure`` overrides.
Configure layouts/styles outside bulk calls; properties, configuration and context
classes such as ``folder.each.plot.labels`` or ``.SavedFigure`` are deliberately
not exposed. Set individual ``data.plot.labels`` or use a surrounding style context.

.. plot:: samples/folder_plotting.py
    :include-source:
    :outname: folder_pages
