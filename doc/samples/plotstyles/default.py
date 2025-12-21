"""Example plot using default style."""

# pylint: disable=invalid-name
import os.path as path

from Stoner import Data, __home__
from Stoner.plot.formats import DefaultPlotStyle

filename = path.realpath(
    path.join(__home__, "..", "doc", "samples", "sample.txt")
)
d = Data(filename, setas="xy", template=DefaultPlotStyle)
d.plot()
