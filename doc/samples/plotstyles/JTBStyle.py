"""Example plot using Joe Batley's plot style."""
from Stoner import Data
from Stoner.plot.formats import JTBPlotStyle
import os.path as path
filename=path.realpath(path.join(path.dirname(__file__),"..","sample.txt"))
d=Data(filename,setas="xy",template=JTBPlotStyle)
d.plot()

