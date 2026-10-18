"""Plot data using two y-axes."""

# pylint: disable=invalid-name
from Stoner import Data

p = Data("sample.txt", setas="xyy")
with p.plot.style("stoner"):
    ax = p.plot.line(0, 1, fmt="k-", legend=False)
    with p.plot.DoubleYAxis(ax=ax, colours=["black", "red"]) as right:
        p.plot.line(0, 2, ax=right, fmt="r-", legend=False)
