"""Test langevin fitting."""

# pylint: disable=invalid-name
from copy import copy

from numpy import linspace, ones_like
from numpy.random import normal

from Stoner import Data
from Stoner.analysis.fitting.models.magnetism import langevin, Langevin

B = linspace(-0.01, 0.01, 100)
params = [1, 1.0e-11, 250]
G = langevin(B, *params) + normal(size=len(B), scale=5e-3)
dG = ones_like(B) * 5e-3
d = Data(
    B,
    G,
    dG,
    setas="xye",
    column_headers=["Field $\\mu_0H (T)$", "Moment", "dM"],
)

func = lambda H, M_s, m: langevin(H, M_s, m, 250)

d.curve_fit(func, p0=copy(params)[0:2], result=True, header="curve_fit")

d.setas = "xye"
fit = Langevin()
fit.params = fit.guess(G, x=B)
fit.params["T"].value = 250
fit.params["T"].vary = False

d.lmfit(fit, p0=fit.params, result=True, header="lmfit")


d.setas = "xyeyy"
d.plot(fmt=["r.", "b-", "g-"])

d.annotate_fit(
    func,
    x=0.1,
    y=0.5,
    fontdict={"size": "x-small", "color": "blue"},
    mode="eng",
)
d.annotate_fit(
    Langevin,
    x=0.1,
    y=0.25,
    fontdict={"size": "x-small", "color": "green"},
    prefix="Langevin",
    mode="eng",
)
d.title = "langevin Fit"
