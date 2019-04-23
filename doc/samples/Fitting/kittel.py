"""Test Weak-localisation fitting."""
from Stoner import Data
import Stoner.Fit as SF
from Stoner.plot.formats import TexEngFormatter

from numpy import linspace, ones_like
from numpy.random import normal
from copy import copy

B = linspace(1e3, 5e4, 51)
params = [2.2, 1e5, 2e2]
G = SF.kittelEquation(B, *params) + normal(size=len(B), scale=5e7)
dG = ones_like(B) * 5e7

d = Data(
    B,
    G,
    dG,
    setas="xye",
    column_headers=["Field $Oe$", r"$\nu (Hz)$", r"\delta $\nu (Hz)$"],
)

d.curve_fit(SF.kittelEquation, p0=copy(params), result=True, header="curve_fit")

fit = SF.KittelEquation()
p0 = fit.guess(G, x=B)

d.lmfit(fit, p0=p0, result=True, header="lmfit")

d.setas = "xyeyy"
d.plot(fmt=["r.", "b-", "g-"])

d.annotate_fit(
    SF.kittelEquation,
    x=0.5,
    y=0.25,
    fontdict={"size": "x-small", "color": "blue"},
    mode="eng",
)
d.annotate_fit(
    SF.KittelEquation,
    x=0.5,
    y=0.05,
    fontdict={"size": "x-small", "color": "green"},
    prefix="KittelEquation",
    mode="eng",
)
d.title = "Kittel Fit"
d.fig.gca().xaxis.set_major_formatter(TexEngFormatter())
d.fig.gca().yaxis.set_major_formatter(TexEngFormatter())
d.tight_layout()
