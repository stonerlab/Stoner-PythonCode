from Stoner import Data
from Stoner.PlotFormats import GBPlotStyle

d=Data("../sample.txt",setas="xy",template=GBPlotStyle)
d.y=d.y-(max(d.y)/2)
d.plot()

