"""Compare independently masked distributions and joint hexagonal binning."""

import numpy as np

from Stoner import Data

rng = np.random.default_rng(2026)
x = rng.normal(size=500)
a = 0.7*x + rng.normal(scale=0.5, size=x.size)
b = rng.normal(loc=0.6, scale=0.7, size=x.size)
d = Data(x, a, b, setas="xyy", column_headers=["Input", "Response A", "Response B"])
d.mask[::13, 1] = True
d.mask[::17, 2] = True

with d.plot.style("stoner"):
    figure = d.plot.new_figure(figsize=(9, 7))
    with d.plot.MultiPanel((2, 2), figure=figure, adjust_figsize=False) as panels:
        d.plot.hist(ax=panels[0, 0], bins=25, histtype="step", density=True, title="Histograms")
        d.plot.kde(ax=panels[0, 1], ind=300, title="Gaussian density estimates")
        d.plot.box(ax=panels[1, 0], label=["A", "B"], orientation="horizontal", title="Independent samples")
        d.plot.hexbin(y=1, ax=panels[1, 1], gridsize=20, mincnt=1, title="Joint input/response counts")
