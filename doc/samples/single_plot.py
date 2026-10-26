"""Draw a measurement and set its title through the plotting accessor."""

# pylint: disable=invalid-name
from Stoner import Data

p = Data("sample.txt", setas="xy")
# Quick plot
p.plot()
# Format the recent plot; showing and saving remain explicit.
p.plot.title = "A quick plot"
