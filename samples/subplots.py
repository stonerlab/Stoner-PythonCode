"""Independent sub plots for multiple y data."""
# pylint: disable=invalid-name
from Stoner import Data

p = Data("sample.txt", setas="xyy")
# Quick plot
p.plot(multiple="subplots")
# Helps to fix layout !
p.tight_layout()
