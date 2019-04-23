"""Test Weak-localisation fitting."""
from Stoner import Data
import Stoner.Fit as SF
from numpy import linspace, ones_like
from numpy.random import normal
from copy import deepcopy

T = linspace(4.2, 300, 101)
params = [265, 65, 1.0, 5]
params2 = deepcopy(params)
G = SF.blochGrueneisen(T, *params) + normal(size=len(T), scale=5e-5)
dG = ones_like(T) * 5e-5
d = Data(T, G, dG, setas="xye", column_headers=["Temperature (K)", "Resistivity", "dR"])

d.curve_fit(SF.blochGrueneisen, p0=params, result=True, header="curve_fit")

d.setas = "xy"
d.lmfit(SF.BlochGrueneisen, p0=params2, result=True, header="lmfit")

d.setas = "xyeyy"
d.plot(fmt=["r.", "b-", "g-"])

d.annotate_fit(SF.blochGrueneisen, x=0.65, y=0.35, fontdict={"size": "x-small"})
d.annotate_fit(
    SF.BlochGrueneisen,
    x=0.65,
    y=0.05,
    fontdict={"size": "x-small"},
    prefix="BlochGrueneisen",
)
d.title = "Bloch-Grueneisen Fit"
d.tight_layout()
