"""Add an inset to a plot."""

# pylint: disable=invalid-name
from Stoner import Data

p = Data("sample.txt", setas="xy")
with p.plot.style("stoner"):
    ax = p.plot()
    with p.plot.InsetPlot(ax=ax, loc="upper right", width=0.5, height=0.5) as inset:
        p.plot.line(y=2, ax=inset, title=None, legend=False)
