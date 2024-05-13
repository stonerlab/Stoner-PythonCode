"""Plot data using multiple sub-plots."""

# pylint: disable=invalid-name
from Stoner import Data

p = Data("sample.txt", setas="xyy")
# Quick plot
p.plot(multiple="panels")
# Helps to fix layout !
p.set_layout_engine("tight")
p.draw()
