"""Example plot in XKCD comic style SketchPlot template."""
from Stoner import Data, __home__
from Stoner.plot.formats import SketchPlot
import os.path as path

filename = path.realpath(path.join(__home__, "..", "doc", "samples", "sample.txt"))
d = Data(filename, setas="xy", template=SketchPlot)
d.plot()
