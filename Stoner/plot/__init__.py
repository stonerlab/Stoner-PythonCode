"""Provide the callable Data plotting accessor and stonerplots layout contexts.

Use :py:attr:`Stoner.core.data.Data.plot` for automatic or explicit scientific
plotting through :py:class:`Stoner.plot.accessor.DataPlotter`. Numerical
``Data.griddata`` remains independently callable in :mod:`Stoner.plot.functions`.
"""

from stonerplots import (
    CentredAxes,
    DoubleXAxis,
    DoubleXYAxis,
    DoubleYAxis,
    InsetPlot,
    MultiPanel,
    OffsetPlots,
    PlotLabeller,
    SavedFigure,
    StackVertical,
    TexEngFormatter,
    TexFormatter,
)

from .accessor import DataPlotter

__all__ = [
    "CentredAxes",
    "DataPlotter",
    "DoubleXAxis",
    "DoubleXYAxis",
    "DoubleYAxis",
    "InsetPlot",
    "MultiPanel",
    "OffsetPlots",
    "PlotLabeller",
    "SavedFigure",
    "StackVertical",
    "TexEngFormatter",
    "TexFormatter",
    "utils",
]
