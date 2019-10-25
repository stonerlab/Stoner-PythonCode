"""Plot 3d fdata as a colourmap."""
from Stoner import Data
import numpy as np

x, y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
x = x.ravel()
y = y.ravel()
z = np.cos(4 * np.pi * np.sqrt(x ** 2 + y ** 2)) * np.exp(-np.sqrt(x ** 2 + y ** 2))

p = Data(np.column_stack((x, y, z)), column_headers=["X", "Y", "Z"])
p.setas = "xyz"

p.colormap_xyz()
p.title = "Colourmap plot"
