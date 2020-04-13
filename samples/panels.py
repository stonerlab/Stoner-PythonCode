"""Plot data using mutiple sub-plots."""
# pylint: disable=invalid-name
from Stoner import Data

p = Data("sample.txt", setas="xyy")
# Quick plot
p.plot(multiple="panels")
# Helps to fix layout !
p.tight_layout()
