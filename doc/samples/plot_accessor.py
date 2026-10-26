"""Compare automatic, plain-line, scatter and shaded-error plotting."""

import numpy as np

from Stoner import Data

x = np.linspace(0, 2 * np.pi, 31)
d = Data(
    x,
    np.sin(x),
    np.full_like(x, 0.12),
    setas="xye",
    column_headers=["Phase / rad", "Signal", "Uncertainty"],
)

figure = d.plot.new_figure(figsize=(8, 6), layout="constrained")
axes = figure.subplots(2, 2)
d.plot(ax=axes[0, 0], fmt="o", title="Automatic: inferred error bars")
d.plot.line(ax=axes[0, 1], title="Explicit plain line")
d.plot.scatter(ax=axes[1, 0], marker="o", title="Explicit scatter")
d.plot.errorfill(ax=axes[1, 1], title="Explicit shaded uncertainty")

# Property forwarding uses the most recent Axes, then its Figure.
d.plot.xlabel = "Phase / rad"
d.plot.title("Explicit shaded uncertainty", loc="center", fontsize=10)
assert d.plot.title.get(loc="center") == "Explicit shaded uncertainty"
d.plot.dpi = 100
