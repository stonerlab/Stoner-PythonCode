"""Double y axis plot."""

# pylint: disable=invalid-name
from Stoner import Data

p = Data("sample.txt", setas="xyy")
with p.plot.SavedFigure(filename=False, style="stoner"):
    ax = p.plot.line(y=1, legend=False)
    with p.plot.DoubleYAxis(ax=ax) as right:
        p.plot.line(y=2, ax=right, color="red", legend=False)
