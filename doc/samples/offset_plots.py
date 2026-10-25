"""Separate related traces and their uncertainties without editing measurements."""

import numpy as np

from Stoner import Data

x = np.linspace(0, 2 * np.pi, 41)
d = Data(x, 2 + np.sin(x), 2 + np.cos(x), np.full(x.size, 0.12),
         setas="xyye", column_headers=["Phase (rad)", "Sine", "Cosine", "Uncertainty"])

# OffsetPlots tracks axes created inside its block by default.
with d.plot.SavedFigure(filename=False, style="stoner"), d.plot.OffsetPlots(offset=2, factor=4):
    figure = d.plot.new_figure(figsize=(8, 3.5))
    with d.plot.MultiPanel((1, 2), figure=figure, adjust_figsize=False) as panels:
        for y in (1, 2):
            d.plot.errorbar(y=y, yerr=3, ax=panels[0], capsize=3, title=None)
            d.plot.line(y=y, ax=panels[1], yscale="log", title=None)
        panels[0].set(title="Additive offsets", ylabel="Signal + offset")
        panels[1].set(title="Multiplicative offsets", ylabel="Signal × factor")
