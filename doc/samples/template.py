"""Simple plotting with a template."""

# pylint: disable=invalid-name, no-member
from cycler import cycler

from Stoner import Data
from Stoner.plot.formats import DefaultPlotStyle

p = Data("sample.txt", setas="xy", template=DefaultPlotStyle())
p.template(  # pylint: disable=not-callable
    axes__prop_cycle=cycler("color", ["r", "g", "b"])
)  # pylint: disable=not-callable
p.plot()
p.y += 1.0
p.plot()
p.y += 1.0
p.plot()
