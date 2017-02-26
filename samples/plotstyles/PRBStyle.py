from Stoner import Data
from Stoner.plot.formats import PRBPlotStyle

d=Data("../sample.txt",setas="xy",template=PRBPlotStyle)
d.plot()

