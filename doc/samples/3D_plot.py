"""3D surface plot example."""

# pylint: disable=invalid-name
import numpy as np
import matplotlib.cm

from Stoner import Data

x, y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
x = x.ravel()
y = y.ravel()
z = np.cos(4 * np.pi * np.sqrt(x**2 + y**2)) * np.exp(-np.sqrt(x**2 + y**2))

p = Data()
p = p & x & y & z
p.column_headers = ["X", "Y", "Z"]
p.setas = "xyz"

p.plot_xyz(cmap=matplotlib.cm.jet)
p.title = "Surface plot"
