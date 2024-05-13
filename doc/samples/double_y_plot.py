"""Plot data using two y-axes."""

# pylint: disable=invalid-name
from Stoner import Data

p = Data("sample.txt", setas="xyy")
p.plot_xy(0, 1, "k-")
p.y2()
p.plot_xy(0, 2, "r-")
