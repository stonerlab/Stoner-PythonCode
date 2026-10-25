"""Combine layouts, an inset and independent double axes with one Stoner import."""

from Stoner import Data

d = Data("sample.txt", setas="xyy")
with d.plot.SavedFigure(filename=False, style="stoner"):
    figure = d.plot.new_figure(figsize=(8, 3.5))
    with d.plot.PlotLabeller():
        with d.plot.MultiPanel(
            (1, 2), figure=figure, adjust_figsize=False
        ) as panels:
            left = panels[0]
            d.plot.line(y=1, ax=left, title=None)
            with d.plot.InsetPlot(
                ax=left, loc="upper right", width=0.4, height=0.4
            ) as inset:
                d.plot.scatter(y=2, ax=inset, title=None, legend=False, s=8)
            right = panels[1]
            d.plot.line(y=1, ax=right, title=None, legend=False)
            with d.plot.DoubleYAxis(ax=right, colours=["black", "red"]) as twin:
                d.plot.line(y=2, ax=twin, color="red", legend=False, title=None)
