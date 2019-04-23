"""Test Weak-localisation fitting."""
from Stoner import Data
import Stoner.Fit as SF
from numpy import linspace, ones_like
from numpy.random import normal
from copy import copy

B = linspace(-8, 8, 201)
params = [1e-3, 2.0, 0.25, 1.4]
G = SF.wlfit(B, *params) + normal(size=len(B), scale=5e-7)
dG = ones_like(B) * 5e-7
d = Data(
    B,
    G,
    dG,
    setas="xye",
    column_headers=["Field $\\mu_0H (T)$", "Conductance", "dConductance"],
)

d.curve_fit(SF.wlfit, p0=copy(params), result=True, header="curve_fit")

d.setas = "xye"
d.lmfit(SF.WLfit, p0=copy(params), result=True, header="lmfit")

d.setas = "xyeyy"
d.plot(fmt=["r.", "b-", "g-"])

d.annotate_fit(SF.wlfit, x=0.05, y=0.75, fontdict={"size": "x-small", "color": "blue"})
d.annotate_fit(
    SF.WLfit,
    x=0.05,
    y=0.5,
    fontdict={"size": "x-small", "color": "green"},
    prefix="WLfit",
)
d.title = "Weak Localisation Fit"
d.tight_layout()
