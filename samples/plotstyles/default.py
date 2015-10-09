from Stoner import Data
from Stoner.PlotFormats import DefaultPlotStyle

d=Data("../sample.txt",setas="xy",template=DefaultPlotStyle)
d.plot()

