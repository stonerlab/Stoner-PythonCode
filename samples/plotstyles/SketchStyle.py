from Stoner import Data
from Stoner.PlotFormats import SketchPlot    

d=Data("../sample.txt",setas="xy",template=SketchPlot)
d.plot()

