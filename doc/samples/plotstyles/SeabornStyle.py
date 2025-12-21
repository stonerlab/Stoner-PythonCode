"""Example plot style using Seaborn plot styling template."""

# pylint: disable=invalid-name
import os.path as path

from Stoner import Data, __home__
from Stoner.plot.formats import SeabornPlotStyle

filename = path.realpath(
    path.join(__home__, "..", "doc", "samples", "sample.txt")
)
d = Data(
    filename,
    setas="xyy",
    template=SeabornPlotStyle(
        stylename="dark", context="talk", palette="muted"
    ),
)
d.plot(multiple="y2")
