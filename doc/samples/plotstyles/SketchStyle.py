"""Example plot in XKCD comic style SketchPlot template."""
from Stoner import Data
from Stoner.plot.formats import SketchPlot    
import os.path as path
filename=path.realpath(path.join(path.dirname(__file__),"..","sample.txt"))
d=Data(filename,setas="xy",template=SketchPlot)
d.plot()

