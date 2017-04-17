"Example plot using Joe Batley's plot style."
from Stoner import Data
from Stoner.plot.formats import JTBPlotStyle

d=Data("../sample.txt",setas="xy",template=JTBPlotStyle)
d.plot()

