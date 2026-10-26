Worked layout and style recipes
*******************************

.. currentmodule:: Stoner

These complete examples show alternative ways to arrange or style a plot.
Choose the recipe that matches your task. They also illustrate replacements
for older layout and template APIs; see :doc:`plot_migration` for the call table.
Read :doc:`plot_layouts` first if ``with`` blocks and panel handles are new to you.

.. contents:: On this page
   :local:
   :depth: 1

Panels, insets and style recipes
================================

Ordinary multiple-Y plotting replaces ``multiple="common"``. Use
``StackVertical`` for ``multiple="panels"``, ``MultiPanel`` for
``multiple="subplots"``, and ``DoubleYAxis`` for ``multiple="y2"``. Replace
``d.inset(...)`` with ``d.plot.InsetPlot(ax=parent, ...)``. For an independent
second X scale use ``d.plot.DoubleXAxis``; for both coordinates use
``d.plot.DoubleXYAxis``. Transformed units use Matplotlib's
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

Next steps
==========

Return to :doc:`plot_layouts` for context options or :doc:`plotfile`
to choose another plotting task.
