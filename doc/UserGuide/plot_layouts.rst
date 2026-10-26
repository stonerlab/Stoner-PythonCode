Panels, styles and independent scales
*************************************

.. currentmodule:: Stoner

A layout arranges several plotting areas in one picture. A style chooses
defaults such as fonts, line widths and colours. Begin with a ``with`` block
to apply a style temporarily::

    with d.plot.style("stoner"):
        d.plot()

For two panels, create a new Figure and draw explicitly into each panel::

    figure = d.plot.new_figure()
    with d.plot.MultiPanel(2, figure=figure) as panels:
        d.plot.line(ax=panels[0])
        d.plot.scatter(ax=panels[1])
    d.plot.show()

These examples use ``d`` from :doc:`plotfile`. The ``with`` blocks apply their
settings while the indented code runs and restore the previous context on exit.
The later examples cover independent scales and offsets between datasets.

.. contents:: On this page
   :local:
   :depth: 1

.. _plotting-contexts:

Layouts and styles with one Stoner import
=========================================

Stoner requires stonerplots 1.9.8 or newer. Its context classes are available
directly from ``d.plot`` and :py:mod:`Stoner.plot`. They accept the same arguments as in stonerplots. Each is used in a ``with``
block to apply its layout or style; the examples below show how to draw within it.

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
    * - ``OffsetPlots``
      - Separate datasets vertically on each axes, including their error bars.
    * - ``DoubleYAxis``
      - Independent right-hand Y scale with combined legend handling.
    * - ``DoubleXAxis``
      - Independent top X scale sharing the primary Y scale.
    * - ``DoubleXYAxis``
      - Independent top X and right Y scales on one overlaid Axes.
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

``d.plot.style(style=None, extra=None)`` accepts the same style names and
``extra`` rcParams overrides as ``SavedFigure``. Pass a comma-separated string
such as ``"stoner, aps"`` or a list such as ``["stoner", "aps"]``; whitespace
and empty comma-separated entries are ignored. Styles are applied in order,
then ``extra`` overrides their values. Supply ``extra`` as a keyword argument::

    with d.plot.style("stoner, aps", extra={"lines.linewidth": 2}):
        ax = d.plot()

Omitting ``style`` or passing ``None`` uses ``stonerplots.default.style``, just
like ``SavedFigure``. Use ``style=[]`` or ``style=""`` for overrides alone.
Style files, rcParams dictionaries and mixed lists of names/dictionaries remain
supported. ``after_reset=True`` first resets to Matplotlib defaults. The context
restores rcParams on exit, including exceptions; it does not select, create,
save or close figures. Stonerplots registers the ``stoner``
base style and journal modifiers such as ``aps``, ``aip`` and ``nature``. Styles
affect artists as they are created; configure existing axes explicitly when
changing their properties. No style is applied merely by calling ``d.plot()``.

System font selection varies across platforms. The fitting examples select
Matplotlib's bundled DejaVu Sans for portable small mathematical annotations::

    with d.plot.style("stoner", extra={"font.sans-serif": ["DejaVu Sans"]}):
        ax = d.plot()

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

.. _plotting-double-axes:

Independent X, Y and XY scales
------------------------------

``DoubleXAxis`` and ``DoubleXYAxis`` complement ``DoubleYAxis``. All three
contexts yield one secondary Matplotlib Axes and accept ``ax``, ``legend``,
``loc`` and ``colours``. ``DoubleXAxis`` shares Y with its parent while keeping
X independent; ``DoubleYAxis`` shares X while keeping Y independent.
``DoubleXYAxis`` keeps both coordinates independent, with ticks/labels on the
top and right. Its single overlaid Axes follows its parent's plotting rectangle
during layout and resizing.

The returned Axes works with explicit ``ax=`` and is selected as the current
target by default. Each context restores the previously selected figure/axes on
exit, including exceptions. To keep the entry target current inside the block,
use ``switch_to_x2=False``, ``switch_to_y2=False`` or ``switch_to_xy2=False``
respectively, and draw with explicit ``ax=``.

``legend=True`` merges legend entries onto the primary axes on exit; use
``legend=False`` to manage legends yourself. ``colours=("black", "red")`` colours
primary/secondary spines, ticks and labels for the independent coordinates.
Set series colours separately in drawing calls. These contexts plot independent
datasets; use Matplotlib's ``secondary_xaxis``/``secondary_yaxis`` for a coordinate
conversion such as frequency to period instead.

.. plot:: samples/double_axes.py
    :include-source:
    :outname: double_axes
    :width: 100%

.. _plotting-offsets:

Offset datasets on one axes
----------------------------

``d.plot.OffsetPlots(offset=2)`` shifts dataset ``i`` by ``i * 2`` in Y data
units on successful exit; the first dataset stays in place. On logarithmic Y
axes, ``factor=10`` instead multiplies dataset ``i`` by ``10 ** i``. Offsets
change the rendered artists only: source values, exclusions, roles and metadata
remain unchanged. Each axes starts its own dataset count.

Create the axes inside the block, as in the example below. To process an
existing axes use ``include_open=True``; this includes existing artists on
**all open axes**, so close unrelated figures first if necessary. The context
restores the figure/axes selected at entry, including on exceptions. A failed
block leaves its artists unshifted.

By default, equal nonzero numbers of lines and scatter collections are paired
by their order within each type and move together. Use ``pair=False`` to shift
every dataset separately in creation order. Error-bar data lines, caps and
segments move as one dataset, including ``fmt="none"`` plots. ``lines``,
``scatter`` and ``errorbars`` control which artists participate. Filled bands
(including ``errorfill``), images and reference lines in axes coordinates are
not shifted; use ``errorbar`` when uncertainties must follow their trace.
The upstream context handles ordinary 2D Cartesian axes with linear or log Y
scales and recalculates limits while respecting explicit limits.

.. plot:: samples/offset_plots.py
    :include-source:
    :outname: offset_plots

Next steps
==========

See :doc:`plot_recipes` for more worked layout and style examples, or
:doc:`plot_folders` for collections of measurements. Return to :doc:`plotfile`.
