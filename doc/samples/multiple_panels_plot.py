"""Plot data using multiple sub-plots."""

# pylint: disable=invalid-name
from Stoner import Data

p = Data("sample.txt", setas="xyy")
with p.plot.style("stoner"):
    with p.plot.StackVertical(2) as panels:
        p.plot.line(y=1, ax=panels[0], title=None)
        p.plot.line(y=2, ax=panels[1], title=None)
