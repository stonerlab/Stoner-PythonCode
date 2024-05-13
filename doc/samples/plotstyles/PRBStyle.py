"""Example plot using a style similar to Physical Review B."""

# pylint: disable=invalid-name
import os.path as path

from Stoner import Data, __home__
from Stoner.plot.formats import PRBPlotStyle

filename = path.realpath(
    path.join(__home__, "..", "doc", "samples", "sample.txt")
)
d = Data(filename, setas="xy", template=PRBPlotStyle)
d.plot()
