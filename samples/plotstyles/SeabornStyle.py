from Stoner import Data
from Stoner.PlotFormats import SeabornPlotStyle

d=Data("../sample.txt",setas="xyy",template=SeabornPlotStyle(stylename="dark",context="talk",palette="muted"))
d.plot(multiple="y2")

