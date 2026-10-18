Release notes
=============

Unreleased: plotting redesign
-----------------------------

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
