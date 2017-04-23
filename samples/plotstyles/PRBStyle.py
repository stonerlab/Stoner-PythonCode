"""Example plot using a style similar to Physical Review B."""
from Stoner import Data
from Stoner.plot.formats import PRBPlotStyle
import os.path as path
filename=path.realpath(path.join(path.dirname(__file__),"..","sample.txt"))
d=Data(filename,setas="xy",template=PRBPlotStyle)
d.plot()

