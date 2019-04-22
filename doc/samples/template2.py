"""Customising a template for plotting."""
from Stoner import Data
from Stoner.plot.formats import SketchPlot

p = Data("sample.txt", setas="xy", template=SketchPlot)
p.plot()
