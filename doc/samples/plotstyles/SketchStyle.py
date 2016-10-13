from Stoner import Data
from Stoner.plot.formats import SketchPlot    

d=Data("../sample.txt",setas="xy",template=SketchPlot)
d.plot()

