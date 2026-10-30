Release notes
=============

0.12.0a6 (2026-10-30)
---------------------

``ImageFile.plot`` now provides callable image display, explicit ``image``,
``imshow``, ``pcolormesh``, ``contour`` and ``hist`` methods with Axes returns,
shared styles/layout contexts and live property forwarding consistent with
``Data.plot``. Native calibrated coordinates are preserved; non-uniform grids
use meshes. Exclusions, finite-value checks and explicit Boolean/labelled
selections affect rendering without changing scientific pixels or metadata.
``CoordinateMode`` supplies native/pixel choices alongside supported strings.
Legacy ``ImageFile.imshow`` retains its Figure return. ROI management and image
fitting remain future milestones. See :doc:`UserGuide/image_plotting`.

0.12.0a5 (2026-10-30)
---------------------

``folder.each.fit`` now supports the callable lmfit default, explicit algorithms
and registered-model shortcuts. Calls preserve member order, result collection,
metadata storage and existing serial/thread/process controls. Custom shortcut
definitions are transferred to workers without changing their global registries.
Unsupported definition transfer has an explicit diagnostic and cleans up the
operation's pool. See :doc:`UserGuide/fitting_folders`.

Internal extrapolation and hysteresis helpers now use ``Data.fit``. Shared
fitting preparation also accepts ordinary integer Y-column selectors instead
of treating them as array containers.

``Data.fit`` now provides owner-bound fitting with ``x`` and ``y`` column
selectors. ``d.fit(model, ...)`` defaults to lmfit; registered-model shortcuts
such as ``d.fit.linear(...)`` use the same algorithm. Explicit ``lmfit``,
``curve_fit``, ``odr``, ``differential_evolution`` and ``polyfit`` methods preserve
their existing options and returns. Direct fitting methods remain available.

Documentation, runnable examples and analysis scripts now use ``Data.fit`` and
``folder.each.fit``. Fitting coordinate keywords on these accessors are ``x``
and ``y``. Model/decorator examples also demonstrate registered shortcuts.

The accessor provides model-name completion, a live read-only ``models`` view,
and fresh model state for each fit. Saved shortcuts resolve the current
registration. See :doc:`UserGuide/fitting_registry` and :doc:`fit-shortcuts`.

Differential evolution now resolves explicitly selected coordinate columns before
calculating fit statistics, fixing a failure for named X-column selectors.

Fitting models now have a process-local registry with snake_case keys. The
existing ``make_model`` decorator registers generated classes and supports named
registration and explicit replacement; ``register_model`` supports handwritten
model subclasses and named factories. Existing model imports and fitting
metadata prefixes are retained. The registry supplies the ``Data.fit`` shortcuts.

``Stoner.enums.FitOutput`` provides string-valued choices for the six common
fitting outputs. Existing strings, defaults and algorithm-specific returns
remain supported. See :doc:`UserGuide/fitting_registry`.

Fit annotations now belong to the plotting accessor: replace
``data.annotate_fit(model, ...)`` with ``data.plot.annotate_fit(model, ...)``.
The method still returns the same ``Data`` object after drawing and returns
formatted parameter text with ``text_only=True``. It uses the accessor's recent
live axes or an explicit ``ax=`` target and never creates a figure.

Column selectors now return all matches at the first successful matching stage:
exact header equality, then case-sensitive whole-header glob matching when the
string contains glob syntax, otherwise literal substring matching, and finally
regular-expression search. Compiled patterns explicitly request regex matching
and preserve their flags. Exact duplicate names select every occurrence.

Use ``data.find_col(selector, first=True)`` to deliberately choose the first
resolved column. Numeric strings remain names, and ``force_list=True`` retains
list results. Methods requiring one coordinate or observation reject ambiguous
explicit selections; general curve fitting retains multiple predictors and
independent dependent channels. Renaming and mapping-row append apply to all
selected columns. The native ``data.data`` DataFrame retains pandas lookup and
alignment semantics.

Column descriptions now use one immutable owner schema, with shared role parsing
and grouping for bound and detached ``setas`` objects. Saved role/header views
follow structural edits; cloned descriptions remain independent. Role changes
validate before commit, and uncertainty columns in later x groups use absolute
positions consistently. Intelligent defaults do not become stored assignments.
Fitted results now re-enter the same owner boundary, keeping native DataFrame
headers consistent with the column schema.

Column ownership retains the composed schema and ordinary pandas columns.
The custom Index prototype remains isolated: native pandas transforms do not
automatically transport Stoner IDs, roles, masks or typed metadata. Use Stoner
structural operations or lossless ``DataStorage`` interchange. The column
selection guide now includes a runnable duplicate-channel example with
explicit first selection, saved views and mask-preserving repetition.

Version 0.12.0a4 (2026-10-26)
--------------------------------

This fourth alpha adds convenient Matplotlib properties to the plotting
accessor, simplifies the User Guide and fixes CSV configuration for fitting.
Python and dependency requirements are unchanged from 0.12.0a3.

After drawing, properties such as ``data.plot.xlabel = "Field / T"`` resolve
against the recent live Axes, then its Figure. Reading a property returns a
callable proxy: ``data.plot.xlabel()`` retrieves its value, while
``data.plot.title("Run", loc="left")`` supports additional setter arguments.
Explicit ``.get(...)`` and ``.set(...)`` calls handle more complex signatures.
Existing accessor attributes retain precedence. Property access does not create
a figure, and the plotting methods retain their Axes return values.

The data, plotting and curve-fitting guides now start with practical examples
and link to shorter pages organised by task. Outdated embedded video tutorials
have been removed. Documentation samples and analysis scripts use the plotting
properties where appropriate, retaining explicit Axes for individual panels and
artist-specific controls. Cached example figures are retained.

Fitting INI files now forward CSV layout settings to the registered CSV loader,
fixing `issue #41 <https://github.com/stonerlab/Stoner-PythonCode/issues/41>`_.
``header_line``/``data_line`` use zero-based row numbers, while configurations
using historical ``start_line`` without ``data_line`` retain one-based
``header_line``/``start_line`` numbering. ``separator`` remains a fallback for
both delimiters; explicit modern settings and Python loader keywords take
precedence. Omitted settings retain the loader defaults. See
:doc:`UserGuide/fitting_configuration` for headerless input, delimiter detection,
legacy options and examples.

Version 0.12.0a3 (2026-10-25)
--------------------------------

This third alpha extends the plotting accessor and requires stonerplots 1.9.8
or newer. Python, pandas/xarray storage and the Axes return contracts are
unchanged from the previous alpha.

Boolean TIFF saving now converts pixels explicitly for Pillow 13's stricter
mode validation, preserving 0/1 values and Boolean dtype metadata. Test
environments require modern pytest-xdist/pytest-forked releases to avoid
pre-collection failures with current Python/pytest combinations.

``Data.plot.style`` now accepts comma-separated style names and ``extra``
rcParams overrides, with the same precedence and default styles as
``SavedFigure``. Native dictionaries, mixed style lists and ``after_reset``
remain supported. Settings are restored on exit without figure lifecycle actions.

``Data.plot.OffsetPlots`` and ``Stoner.plot.OffsetPlots`` directly expose the
upstream context for additive linear offsets or multiplicative logarithmic
offsets, including complete error bars. Source measurements remain unchanged.
See :ref:`plotting-offsets`.

``Data.plot.DoubleXAxis`` and ``DoubleXYAxis`` (also exported by ``Stoner.plot``)
complement ``DoubleYAxis``. They yield a secondary Axes with an independent top
X scale sharing Y, or independent top X/right Y scales respectively. The
required stonerplots version is now 1.9.8; see :ref:`plotting-double-axes`.

The repository analysis scripts now use the plotting accessor, explicit
Matplotlib axes and mask-preserving numerical edits. Standalone scripts have
import-safe callable entry points and documented input requirements. Regression
coverage includes the bundled transport, PCAR, VSM, BNL, diffraction and image
data, with controlled fringe/XMCD/GenX inputs. The GenX loader's host interface
is exercised separately from the external application's interactive UI.
The PCAR INI files retain their CSV layout reference. Forwarding those entries
through the configuration helper remains deferred under
`issue #41 <https://github.com/stonerlab/Stoner-PythonCode/issues/41>`_;
direct CSV loader layout controls remain available.

Version 0.12.0a2 (2026-10-18)
--------------------------------

This second alpha introduces a breaking redesign of the plotting API and keeps
the pandas/xarray storage model from 0.12.0a1. Existing scripts using the legacy
plotting methods and templates need the migration changes below. Python 3.12 or
newer and stonerplots 1.9.6 or newer are required.

Plotting API (breaking changes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Data.plot`` is now a callable accessor: ``ax = data.plot()`` retains scientific
column-role and uncertainty inference, while named methods such as
``data.plot.line()``, ``surface()``, ``vector3d()``, ``hist()`` and ``kde()`` select
an explicit plot type. Drawing always returns the primary Matplotlib Axes.
Showing and saving are explicit; returned Axes provide native formatting methods.
Selections honour Stoner exclusions and leave measurements, roles and metadata
unchanged. Direct Matplotlib/SciPy implementations keep these scientific rules
independent of pandas' plotting backend.

Stoner requires stonerplots 1.9.6 or newer for styles, tick formatters, panels,
stacked plots, double Y axes, insets and saving contexts. These are available from
the accessor with the normal ``from Stoner import Data`` import. Drawing inherits
the caller's style; no legacy template is applied implicitly. Journal styles can
differ visibly from the removed template stylesheets.
Fitting examples use Matplotlib's bundled DejaVu Sans for portable mathematical
annotations instead of relying on the platform's preferred sans-serif font.

``PlotFolder.plot`` now paginates with stonerplots and returns primary Axes in
member order. ``folder.figures`` inspects recent live pages. Named bulk drawing
is available through ``folder.each.plot.line(...)`` and runs serially even when
multiprocessing/threading is enabled. Ordinary bulk plotting does not add
diagnostic metadata; explicit ``_return`` retains result storage.
Failed pages are closed and their members' previous plotting targets restored;
earlier completed pages remain available.

The old top-level drawing/layout methods, PlotMixin, template classes and broad
pyplot/Axes/Figure proxies have been removed. Replace ``data.fig`` with
``ax.figure``, ``data.labels`` with ``data.plot.labels`` and formatting assignments
with native Axes methods. Independent numerical ``Data.griddata`` and image
display APIs remain available. The unused Mayavi extra has been removed.

See :doc:`UserGuide/plotfile` for current examples, selection/mask semantics and
the call-by-call migration guide. Supported Python versions remain 3.12–3.14;
the storage migration's pandas/xarray requirements are unchanged.

Scan frame access
~~~~~~~~~~~~~~~~~~~~

Accessing Attocube or MAXIMUS stack frames no longer exports their pixels and
exclusion masks merely to expose scan metadata. Saved frame handles retain their
identity through reordering, live common-header defaults and local overrides,
presentation attributes, masks, fill values and spatial calibration.

Documentation and maintenance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The plotting guide, runnable examples and retained plot cache describe the new
accessor and stonerplots contexts. The Read the Docs configuration initialises
headless Qt and Matplotlib before package imports. Repository maintenance and
storage handovers record the scientific contracts and validation evidence.
