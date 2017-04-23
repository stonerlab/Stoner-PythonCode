"""Example plot style using Seaborn plot styling template."""
from Stoner import Data
from Stoner.plot.formats import SeabornPlotStyle
import os.path as path
filename=path.realpath(path.join(path.dirname(__file__),"..","sample.txt"))
d=Data(filename,setas="xyy",template=SeabornPlotStyle(stylename="dark",context="talk",palette="muted"))
d.plot(multiple="y2")

