from Stoner import Data
from Stoner.plot.formats import JTBPlotStyle

d=Data("../sample.txt",setas="xy",template=JTBPlotStyle)
d.plot()

