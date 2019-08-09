"""Simple ploting with a template."""
from Stoner import Data
from Stoner.plot.formats import DefaultPlotStyle
from cycler import cycler

p = Data("sample.txt", setas="xy", template=DefaultPlotStyle())
p.template(
    axes__prop_cycle=cycler("color", ["r", "g", "b"])
)  # pylint: disable=not-callable
p.plot()
p.y += 1.0
p.plot()
p.y += 1.0
p.plot()
