"""Plot data on a single y-axis."""
from Stoner import Data

p = Data("sample.txt", setas="xyy")
# Quick plot
p.plot()
