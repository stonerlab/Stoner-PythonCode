Plotting contexts and tick formatters
========================================

The :py:class:`~Stoner.plot.accessor.DataPlotter` accessor exposes the following
stonerplots classes directly. For example, ``data.plot.MultiPanel`` and
:py:class:`Stoner.plot.MultiPanel` are the same class, with the same constructor
arguments and context handle. No separate stonerplots import is needed::

    from Stoner import Data
    import numpy as np

    data = Data(np.arange(5.0), np.arange(5.0) ** 2, setas="xy")
    figure = data.plot.new_figure()
    with data.plot.MultiPanel(2, figure=figure) as panels:
        data.plot.line(ax=panels[0])
        data.plot.scatter(ax=panels[1])

Each class page below includes the installed dependency's constructor signature
and parameter documentation. See :doc:`UserGuide/plot_layouts` for worked
examples, or the :py:class:`~Stoner.plot.accessor.DataPlotter` API for drawing
methods and temporary styling with :py:meth:`~Stoner.plot.accessor.DataPlotter.style`.

Context managers
----------------

.. autosummary::
   :toctree: classes
   :template: plot-context.rst

   Stoner.plot.SavedFigure
   Stoner.plot.MultiPanel
   Stoner.plot.StackVertical
   Stoner.plot.OffsetPlots
   Stoner.plot.DoubleYAxis
   Stoner.plot.DoubleXAxis
   Stoner.plot.DoubleXYAxis
   Stoner.plot.InsetPlot
   Stoner.plot.CentredAxes
   Stoner.plot.PlotLabeller

Tick formatters
---------------

These classes configure individual Matplotlib tick axes; construct them and pass
them to ``ax.xaxis.set_major_formatter(...)`` or the corresponding Y/Z method.
They are also available as ``data.plot.TexFormatter`` and
``data.plot.TexEngFormatter``.

.. autosummary::
   :toctree: classes
   :template: plot-context.rst

   Stoner.plot.TexFormatter
   Stoner.plot.TexEngFormatter
