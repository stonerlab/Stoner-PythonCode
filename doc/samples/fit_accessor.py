"""Fit a registered model with the callable Data fitting accessor."""

import numpy as np

from Stoner import Data
from Stoner.analysis.fitting.models.generic import Linear
from Stoner.enums import FitOutput

x = np.linspace(0, 3, 25)
y = 2*x + 1 + 0.04*np.sin(5*x)
d = Data(x, y, column_headers=["Field", "Signal"], setas="xy")

report = d.fit.linear(x="Field", y="Signal", result=True,
                      header="Linear fit", output=FitOutput.REPORT)
assert d.fit.models["linear"] is Linear
assert abs(report.params["slope"].value - 2) < 0.05

ax = d.plot.line(x="Field", y=["Signal", "Linear fit"], fmt=["o", "-"])
d.plot.annotate_fit(Linear, ax=ax, x=0.05, y=0.75)
ax.figure.canvas.draw()
print(report.fit_report())
