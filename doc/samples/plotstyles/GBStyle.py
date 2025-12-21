"""Example plot using experimental GBStyle."""

# pylint: disable=invalid-name
import os.path as path

from Stoner import Data, __home__
from Stoner.plot.formats import GBPlotStyle

filename = path.realpath(
    path.join(__home__, "..", "doc", "samples", "sample.txt")
)
d = Data(filename, setas="xy", template=GBPlotStyle)
d.y = d.y - (max(d.y) / 2)
d.plot()
