"""Double y axis plot."""
from Stoner import Data

p = Data("sample.txt", setas="xyy")
# Quick plot
p.plot(multiple="y2")
