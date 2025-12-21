"""Plot data on a single y-axis."""

# pylint: disable=invalid-name
from Stoner import Data

p = Data("sample.txt", setas="xyy")
# Quick plot
p.plot()
