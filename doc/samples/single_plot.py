"""Simple plot in 2 lines."""

# pylint: disable=invalid-name
from Stoner import Data

p = Data("sample.txt", setas="xy")
# Quick plot
p.plot()
