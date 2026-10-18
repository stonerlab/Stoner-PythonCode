"""Draw a rotating three-component vector field with Matplotlib."""

import numpy as np

from Stoner import Data

x, y, z = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4), [0., 1.], indexing="ij")
d = Data(x.ravel(), y.ravel(), z.ravel(), -y.ravel(), x.ravel(), np.full(x.size, 0.5),
         setas="xyzuvw", column_headers=["X", "Y", "Z", "U", "V", "W"])
with d.plot.style("stoner"):
    d.plot.new_figure(figsize=(5.2, 5.0))
    ax = d.plot.vector3d(length=0.3, title="Rotating vector field")
    ax.set_box_aspect((1, 1, 0.7))
