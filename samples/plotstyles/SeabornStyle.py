"""Example plot style using Seaborn plot styling template."""
from Stoner import Data
from Stoner.plot.formats import SeabornPlotStyle

d=Data("../sample.txt",setas="xyy",template=SeabornPlotStyle(stylename="dark",context="talk",palette="muted"))
d.plot(multiple="y2")

