"""Add an inset to a plot."""

# pylint: disable=invalid-name
from Stoner import Data

p = Data("sample.txt", setas="xy")
p.plot()
p.inset(loc=1, width="50%", height="50%")
p.setas = "x.y"
p.plot()
p.title = ""  # Turn off the inset title
