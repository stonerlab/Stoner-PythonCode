"""Compare scalar images, pseudocolour and contours with an excluded sample."""

import numpy as np

from Stoner import Data

x, y = np.meshgrid(np.linspace(-2, 2, 17), np.linspace(-1, 1, 9))
z = np.cos(x)*np.exp(-y*y)
d = Data(x.ravel(), y.ravel(), z.ravel(), setas="xyz", column_headers=["X", "Y", "Signal"])
d.mask[(x.ravel() == 0) & (y.ravel() == 0), 2] = True
with d.plot.style("stoner"):
    figure = d.plot.new_figure(figsize=(10, 3))
    with d.plot.MultiPanel(3, figure=figure, adjust_figsize=False) as panels:
        d.plot.image(ax=panels[0], title="Scalar image", vmin=-0.5, vmax=1)
        d.plot.pcolormesh(ax=panels[1], title="Pseudocolour", vmin=-0.5, vmax=1)
        d.plot.contour(ax=panels[2], title="Filled contours", filled=True, levels=np.linspace(-0.5, 1, 16))
