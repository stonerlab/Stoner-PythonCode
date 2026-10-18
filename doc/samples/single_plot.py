"""Simple plot in 2 lines."""

# pylint: disable=invalid-name
from Stoner import Data

p = Data("sample.txt", setas="xy")
# Quick plot
ax = p.plot()
# Drawing returns the Matplotlib Axes; showing/saving is explicit.
ax.set_title("A quick plot")
