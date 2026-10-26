Lines, scatter and uncertainty plots
************************************

.. currentmodule:: Stoner

Start with ``d.plot()`` to draw a measurement. To choose the appearance
explicitly, use one of these methods::

    d.plot.line()               # Join measurements with a line.
    d.plot.scatter()            # Draw individual points.
    d.plot.errorbar()           # Show assigned measurement uncertainties.
    d.plot.errorfill()          # Show uncertainty as a shaded band.

Here ``d`` is a Data object. Use the small example in :doc:`plotfile` or load
your own measurement as described in :doc:`data_loading`. If your table has
uncertainties, assign their roles as described in :doc:`data_columns`.

The detailed examples illustrate different table layouts: substitute the
column names or positions in your own measurement.

.. contents:: On this page
   :local:
   :depth: 1

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
See :doc:`plot_fields` and :doc:`plot_statistics` for those choices. The
automatic call keeps its scientific role/count dispatch.

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

A complete file-based example
=============================

.. plot:: samples/single_plot.py
    :include-source:
    :outname: single

Next steps
==========

Use :doc:`plot_formatting` to change labels, limits or save your plot.
Return to :doc:`plotfile` to choose another plotting task.
