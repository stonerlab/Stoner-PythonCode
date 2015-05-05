from Stoner import Data
from Stoner.PlotFormats import JTBPlotStyle

d=Data("../sample.txt",setas="xy",template=JTBPlotStyle)
d.plot()

