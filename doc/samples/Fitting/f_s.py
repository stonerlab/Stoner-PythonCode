"""Test Weak-localisation fitting."""
from Stoner import Data
import Stoner.Fit as SF
from numpy import linspace, ones_like
from numpy.random import normal

B = linspace(2, 100, 26)
params = [12.5, 0.75, 1e3]
G = SF.fluchsSondheimer(B, *params) + normal(size=len(B), scale=5e-5)
dG = ones_like(B) * 5e-5
d = Data(
    B,
    G,
    dG,
    setas="xye",
    column_headers=["Thickness (nm)", "Conductance", "dConductance"],
)

d.curve_fit(SF.fluchsSondheimer, p0=params, result=True, header="curve_fit")

d.setas = "xye"
d.lmfit(SF.FluchsSondheimer, p0=params, result=True, header="lmfit")

d.setas = "xyeyy"
d.plot(fmt=["r.", "b-", "g-"])

d.annotate_fit(
    SF.fluchsSondheimer, x=0.2, y=0.6, fontdict={"size": "x-small", "color": "blue"}
)
d.annotate_fit(
    SF.FluchsSondheimer,
    x=0.2,
    y=0.4,
    fontdict={"size": "x-small", "color": "green"},
    prefix="FluchsSondheimer",
)
d.title = "Fluchs-Sondheimer Fit"
d.tight_layout()
