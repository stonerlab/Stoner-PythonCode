"""3D surface plot example."""

# pylint: disable=invalid-name
import numpy as np
import matplotlib.cm

from Stoner import Data

x, y, z = np.meshgrid(
    np.linspace(-2, 2, 21), np.linspace(-2, 2, 21), np.linspace(-2, 2, 21)
)
x = x.ravel()
y = y.ravel()
z = z.ravel()
u = np.sin(x * y * z)

p = Data(x, y, z, u, setas="xyzu", column_headers=["X", "Y", "Z"])

p.plot_voxels(cmap=matplotlib.cm.jet, visible=lambda x, y, z: x - y + z < 2.0)
p.set_box_aspect((1, 1, 1.0))  # Passing through to the current axes
p.title = "Voxel plot"
