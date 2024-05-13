"""Customising a template for plotting."""

# pylint: disable=invalid-name
from Stoner import Data
from Stoner.plot.formats import SketchPlot

p = Data("sample.txt", setas="xy", template=SketchPlot)
p.plot()
