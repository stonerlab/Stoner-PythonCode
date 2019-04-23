"""Example plot style using Seaborn plot styling template."""
from Stoner import Data, __home__
from Stoner.plot.formats import SeabornPlotStyle
import os.path as path

filename = path.realpath(path.join(__home__, "..", "doc", "samples", "sample.txt"))
d = Data(
    filename,
    setas="xyy",
    template=SeabornPlotStyle(stylename="dark", context="talk", palette="muted"),
)
d.plot(multiple="y2")
