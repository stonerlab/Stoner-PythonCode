from Stoner import Data
from Stoner.PlotFormats import PRBPlotStyle

d=Data("../sample.txt",setas="xy",template=PRBPlotStyle)
d.plot()

