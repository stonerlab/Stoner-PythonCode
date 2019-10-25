"""Example plot using experimental GBStyle"""
from Stoner import Data, __home__
from Stoner.plot.formats import GBPlotStyle
import os.path as path

filename = path.realpath(path.join(__home__, "..", "doc", "samples", "sample.txt"))
d = Data(filename, setas="xy", template=GBPlotStyle)
d.y = d.y - (max(d.y) / 2)
d.plot()
