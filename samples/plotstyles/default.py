"""Example plot using default style."""
from Stoner import Data
from Stoner.plot.formats import DefaultPlotStyle
import os.path as path
filename=path.realpath(path.join(path.dirname(__file__),"..","sample.txt"))
d=Data(filename,setas="xy",template=DefaultPlotStyle)
d.plot()

