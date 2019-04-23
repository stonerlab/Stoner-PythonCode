"""Test Weak-localisation fitting."""
from Stoner import Data
import Stoner.Fit as SF
from numpy import linspace, ones_like
from numpy.random import normal
from copy import copy

B = linspace(-0.01, 0.01, 100)
params = [1, 1.0e-11, 250]
G = SF.langevin(B, *params) + normal(size=len(B), scale=5e-3)
dG = ones_like(B) * 5e-3
d = Data(B, G, dG, setas="xye", column_headers=["Field $\\mu_0H (T)$", "Moment", "dM"])

func = lambda H, M_s, m: SF.langevin(H, M_s, m, 250)

d.curve_fit(func, p0=copy(params)[0:2], result=True, header="curve_fit")

d.setas = "xye"
fit = SF.Langevin()
p0 = fit.guess(G, x=B)
for p, v in zip(p0, params):
    p0[p].set(v)
    p0[p].max = v * 5
    p0[p].min = 0
    p0[p].vary = p != "T"

d.lmfit(fit, p0=p0, result=True, header="lmfit")


d.setas = "xyeyy"
d.plot(fmt=["r.", "b-", "g-"])

d.annotate_fit(
    func, x=0.1, y=0.5, fontdict={"size": "x-small", "color": "blue"}, mode="eng"
)
d.annotate_fit(
    SF.Langevin,
    x=0.1,
    y=0.25,
    fontdict={"size": "x-small", "color": "green"},
    prefix="Langevin",
    mode="eng",
)
d.title = "langevin Fit"
