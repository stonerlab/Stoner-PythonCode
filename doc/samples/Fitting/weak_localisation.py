"""Test Weak-localisation fitting."""

# pylint: disable=invalid-name
from copy import copy

from numpy import linspace, ones_like
from numpy.random import normal

from Stoner import Data
from Stoner.analysis.fitting.models.e_transport import WLfit, wlfit

B = linspace(-8, 8, 201)
params = [1e-3, 2.0, 0.25, 1.4]
G = wlfit(B, *params) + normal(size=len(B), scale=5e-7)
dG = ones_like(B) * 5e-7
d = Data(
    B,
    G,
    dG,
    setas="xye",
    column_headers=["Field $\\mu_0H (T)$", "Conductance", "dConductance"],
)

d.fit.curve_fit(wlfit, p0=copy(params), result=True, header="curve_fit")

d.setas = "xye"
d.fit.lmfit(WLfit, result=True, header="lmfit")

d.setas = "xyeyy"
d.plot(fmt=["r.", "b-", "g-"])

d.plot.annotate_fit(
    wlfit, x=0.05, y=0.75, fontdict={"size": "x-small", "color": "blue"}
)
d.plot.annotate_fit(
    WLfit,
    x=0.05,
    y=0.5,
    fontdict={"size": "x-small", "color": "green"},
    prefix="WLfit",
)
d.plot.title = "Weak Localisation Fit"
