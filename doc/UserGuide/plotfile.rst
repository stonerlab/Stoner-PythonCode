*************
Plotting Data
*************

.. currentmodule:: Stoner

The :py:attr:`~Stoner.core.data.Data.plot` property returns a callable
:py:class:`~Stoner.plot.accessor.DataPlotter`. Use it to infer a scientific plot,
or select a particular ordinary plot through its methods. Only the normal
Stoner import is needed::

    from Stoner import Data

    d = Data("sample.txt", setas="xye")
    ax = d.plot()
    ax.set_title("My measurement")
    ax.figure.savefig("measurement.pdf")

Drawing returns the Matplotlib ``Axes`` and does not show or save it. Notebook
display follows Matplotlib's normal behaviour; elsewhere use ``d.plot.show()``.
Legacy ``showfig`` and ``Options.no_figs`` settings do not affect accessor returns
or suppress drawing.

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

The ordinary methods available in this stage are:

* ``d.plot()``: infer lines or error bars from resolved coordinates/uncertainties;
* ``d.plot.line()``: plain lines, ignoring assigned uncertainties;
* ``d.plot.scatter()``: points drawn with Matplotlib's scatter method;
* ``d.plot.errorbar()``: data and inferred/explicit uncertainties;
* ``d.plot.errorfill()``: lines with shaded uncertainty in one direction.

``d.plot(kind="scatter", ...)`` and ``d.plot.scatter(...)`` share an
implementation. Named ordinary methods override field roles, so
``d.plot.line(x=0, y=1)`` works on an XYZ table. Automatic field plots remain
available through retained scientific helpers. Named field methods and pandas
conveniences are later stages and are not yet accessor methods.

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
Stoner's separate exclusions. Mask-aware pandas-backed plotting is a later stage.

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

Explicit ``ax=`` works with externally imported Matplotlib/stonerplots contexts.
The one-import stonerplots context attributes, styles and replacement template
examples are the next stage and are not yet exposed on the accessor.

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
``show_plot``, ``save_filename`` and old ``*col`` keywords. Retained top-level
methods are temporary migration support, not aliases on the new accessor.

Retained field, layout and template examples
============================================

These families await later stages. Their templates, figure selection and return
policies are distinct from ordinary accessor drawing. Scientific gridding remains
available separately through ``d.griddata``.

* ``plot_xyz``, ``contour_xyz`` and ``image_plot`` handle scalar fields.
* ``plot_xyuv``, ``plot_xyzuvw`` and ``plot_voxels`` handle vector/volume plots.
* ``plot_xy(multiple="panels"/"subplots"/"y2")``, inset/twin-axis helpers and
  template classes remain the temporary layout interface.

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

.. plot:: samples/template2.py
    :include-source:
    :outname: template2

.. plot:: samples/plotstyles/customising.py
    :include-source:
    :outname: customising
