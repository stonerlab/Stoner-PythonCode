"""Test Weak-localisation fitting."""

# pylint: disable=invalid-name
from copy import copy

from numpy import linspace, ones_like
from numpy.random import normal

from Stoner import Data
from Stoner.analysis.fitting.models.magnetism import (
    KittelEquation,
    kittelEquation,
)
from Stoner.plot.formats import TexEngFormatter

B = linspace(1e3, 5e4, 51)
params = [2.2, 1e5, 2e2]
G = kittelEquation(B, *params) + normal(size=len(B), scale=5e7)
dG = ones_like(B) * 5e7

d = Data(
    B,
    G,
    dG,
    setas="xye",
    column_headers=["Field $Oe$", r"$\nu (Hz)$", r"\delta $\nu (Hz)$"],
)

d.curve_fit(kittelEquation, p0=copy(params), result=True, header="curve_fit")

fit = KittelEquation()
p0 = fit.guess(G, x=B)

d.lmfit(fit, p0=p0, result=True, header="lmfit")

d.setas = "xyeyy"
d.plot(fmt=["r.", "b-", "g-"])

d.annotate_fit(
    kittelEquation,
    x=0.5,
    y=0.25,
    fontdict={"size": "x-small", "color": "blue"},
    mode="eng",
)
d.annotate_fit(
    KittelEquation,
    x=0.5,
    y=0.05,
    fontdict={"size": "x-small", "color": "green"},
    prefix="KittelEquation",
    mode="eng",
)
d.title = "Kittel Fit"
d.fig.gca().xaxis.set_major_formatter(TexEngFormatter())
d.fig.gca().yaxis.set_major_formatter(TexEngFormatter())
