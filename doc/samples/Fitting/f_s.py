"""Test Weak-localisation fitting."""

# pylint: disable=invalid-name
from numpy import logspace, ones_like, log10
from numpy.random import normal

from Stoner import Data
from Stoner.analysis.fitting.models.e_transport import (
    fluchsSondheimer,
    FluchsSondheimer,
)

B = logspace(log10(2), 2, 26)
params = [12.5, 0.75, 1e3]
G = fluchsSondheimer(B, *params) + normal(size=len(B), scale=5e-5)
dG = ones_like(B) * 5e-5
d = Data(
    B,
    G,
    dG,
    setas="xye",
    column_headers=["Thickness (nm)", "Conductance", "dConductance"],
)

d.curve_fit(fluchsSondheimer, p0=params, result=True, header="curve_fit")

d.setas = "xye"
d.lmfit(FluchsSondheimer, result=True, header="lmfit")

d.setas = "xyeyy"
d.plot(fmt=["r.", "b-", "g-"])

d.annotate_fit(
    fluchsSondheimer,
    x=0.2,
    y=0.6,
    fontdict={"size": "x-small", "color": "blue"},
)
d.annotate_fit(
    FluchsSondheimer,
    x=0.2,
    y=0.4,
    fontdict={"size": "x-small", "color": "green"},
    prefix="FluchsSondheimer",
)
d.title = "Fluchs-Sondheimer Fit"
