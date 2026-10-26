Plotting collections of measurements
************************************

.. currentmodule:: Stoner

When you have several measurements, use ``PlotFolder`` to give each its own
panel or page. To draw them together, use a shared plotting target::

    from Stoner.folders import PlotFolder

    folder = PlotFolder("measurements")  # Replace with your data directory.
    axes = folder.plot(plots_per_page=4)

The complete example below constructs a folder without needing any files and
shows both pagination and an overlay. See :doc:`datafolder` for loading and
managing collections, and :doc:`plotfile` for plotting a single measurement.

.. contents:: On this page
   :local:
   :depth: 1

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

Next steps
==========

See :doc:`plot_layouts` for layout/style contexts and
:doc:`plot_formatting` for labels and saving. Return to :doc:`plotfile`.
