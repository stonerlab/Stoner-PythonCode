Updating older plotting scripts
*******************************

.. currentmodule:: Stoner

Use this page when a script written for an older Stoner release no longer
plots as expected. For a new script, start with :doc:`plotfile`; the older
names below are included only to help you translate existing code.

Drawing now consistently returns a Matplotlib Axes and does not show or save
the picture automatically. Use ``d.plot.show()`` for display and
``d.plot.figure.savefig(...)`` for saving. The removed ``showfig`` setting has
no replacement, and ``Options.no_figs`` does not change drawing or its return.

.. contents:: On this page
   :local:
   :depth: 1

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
    * - ``d.annotate_fit(model, ...)``
      - ``d.plot.annotate_fit(model, ...)``
    * - conditional Figure/Data/None return
      - always the primary Axes; use ``ax.figure`` or retain ``d`` explicitly

The ordinary accessor rejects ``figure``, ``template``, ``plotter``, ``multiple``,
``show_plot``, ``save_filename`` and old ``*col`` keywords. The former top-level
plotting methods have been removed.

Fit annotations return the same ``Data`` object after drawing. Use
``d.plot.annotate_fit(model, text_only=True)`` for formatted text without drawing;
other annotation calls require a recent live plot or an explicit ``ax=`` target.

Removed helpers and field migration
===================================

Use ``d.plot.surface`` instead of ``plot_xyz``; ``scatter3d`` replaces
``plot_xyz(plotter="scatter", griddata=False)``. Use ``contour``, ``pcolormesh``,
``image``, ``matrix``, ``vector``, ``quiver``, ``vector3d`` and ``voxels`` in place
of their old top-level field helpers. ``waterfall`` replaces the custom
``plotter=joy_division`` path. Old ``griddata``, ``projection``, ``plotter`` and
backend options are rejected by the accessor: select the required named method.
The unused Mayavi installation extra has been removed.

The top-level plotting methods, PlotMixin, template classes and broad pyplot/
Figure/Axes proxies on Data have been removed. Replace ``d.fig`` with ``ax.figure``,
``d.labels`` with ``d.plot.labels``, and formatting attributes such as ``d.title``
with ``d.plot.title`` or ``ax.set_title(...)``. The accessor provides the callable
property forwarding in :doc:`plot_formatting`. Use the Figure's ``subplots``/``add_subplot`` methods
or stonerplots contexts for layouts. Independent X, Y or both scales use
``d.plot.DoubleXAxis``, ``DoubleYAxis`` or ``DoubleXYAxis``. Styles now come from the
required stonerplots dependency, rather than bundled template stylesheets.

Scientific ``d.griddata`` and shared image utilities remain independently callable;
the numerical gridding API and image-display classes retain their separate rules.

Next steps
==========

See :doc:`plot_recipes` for translated layout examples,
:doc:`plot_formatting` for properties, or :doc:`upgrade` for other changes.
Return to :doc:`plotfile` for the current learning path.
