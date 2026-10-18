"""Independent sub plots for multiple y data."""

# pylint: disable=invalid-name
from Stoner import Data

p = Data("sample.txt", setas="xyy")
with p.plot.style("stoner"):
    with p.plot.MultiPanel((1, 2)) as panels:
        for column, ax in zip([1, 2], panels):
            p.plot.line(y=column, ax=ax, title=None)
