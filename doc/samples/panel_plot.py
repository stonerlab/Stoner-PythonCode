"""Plot multiple y data using separate sub-plots."""
from Stoner import Data

p = Data("sample.txt", setas="xyy")
# Quick plot
p.plot(multiple="panels")
