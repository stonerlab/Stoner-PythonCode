Release notes
=============

Unreleased
----------

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
