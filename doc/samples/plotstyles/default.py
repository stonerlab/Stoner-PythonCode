from Stoner import Data
from Stoner.plot.formats import DefaultPlotStyle

d=Data("../sample.txt",setas="xy",template=DefaultPlotStyle)
d.plot()

