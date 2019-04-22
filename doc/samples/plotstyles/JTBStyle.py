"""Example plot using Joe Batley's plot style."""
from Stoner import Data, __home__
from Stoner.plot.formats import JTBPlotStyle
import os.path as path

filename = path.realpath(path.join(__home__, "..", "doc", "samples", "sample.txt"))
d = Data(filename, setas="xy", template=JTBPlotStyle)
d.plot()
