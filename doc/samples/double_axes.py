"""Compare an independent top X scale with independent top X/right Y scales."""

import numpy as np

from Stoner import Data

x = np.linspace(0, 2 * np.pi, 61)
primary = Data(x, np.sin(x), setas="xy", column_headers=["Time (s)", "Signal (V)"])
shared_y = Data(10*x, np.cos(x), setas="xy", column_headers=["Distance (mm)", "Signal (V)"])
independent = Data(10*x, 300 + 20*np.cos(x), setas="xy",
                   column_headers=["Distance (mm)", "Temperature (K)"])

with primary.plot.SavedFigure(filename=False, style="stoner"):
    figure = primary.plot.new_figure(figsize=(9, 4), layout="none")
    with primary.plot.MultiPanel((1, 2), figure=figure, adjust_figsize=False) as panels:
        left, right = list(panels)
        for ax in (left, right):
            primary.plot.line(ax=ax, color="black", label="Primary", title=None)
        with primary.plot.DoubleXAxis(ax=left, colours=("black", "red"), loc="lower left") as top:
            shared_y.plot.line(ax=top, color="red", label="Shared Y", title=None)
        with primary.plot.DoubleXYAxis(ax=right, colours=("black", "red"), loc="lower left") as overlay:
            independent.plot.line(ax=overlay, color="red", label="Independent X/Y", title=None)
    figure.subplots_adjust(left=0.09, right=0.9, bottom=0.18, top=0.8, wspace=0.65)
