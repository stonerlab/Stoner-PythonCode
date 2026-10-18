"""Provide the callable Data plotting accessor, fields and layout contexts.

Use :py:attr:`Stoner.core.data.Data.plot` for automatic or explicit scientific
plotting through :py:class:`Stoner.plot.accessor.DataPlotter`. The legacy
:py:class:`Stoner.plot.core.PlotMixin` and template classes remain temporarily
available for folder helpers and the final removal stage.
"""

from stonerplots import (
    CentredAxes,
    DoubleYAxis,
    InsetPlot,
    MultiPanel,
    PlotLabeller,
    SavedFigure,
    StackVertical,
    TexEngFormatter,
    TexFormatter,
)

from .accessor import DataPlotter
from .core import PlotMixin

__all__ = [
    "CentredAxes",
    "DataPlotter",
    "DoubleYAxis",
    "InsetPlot",
    "MultiPanel",
    "PlotLabeller",
    "PlotMixin",
    "SavedFigure",
    "StackVertical",
    "TexEngFormatter",
    "TexFormatter",
    "formats",
    "utils",
]
