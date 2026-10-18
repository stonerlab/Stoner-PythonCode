"""Provide the callable Data plotting accessor and retained legacy field helpers.

Use :py:attr:`Stoner.core.data.Data.plot` for automatic or explicit ordinary
plotting through :py:class:`Stoner.plot.accessor.DataPlotter`. The legacy
:py:class:`Stoner.plot.core.PlotMixin` and template classes remain available for
field and layout examples awaiting the later refactor stages.
"""

from .core import PlotMixin
from .accessor import DataPlotter

__all__ = ["DataPlotter", "PlotMixin", "formats", "utils"]
