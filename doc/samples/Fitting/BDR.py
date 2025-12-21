"""Example of nDimArrhenius Fit."""

# pylint: disable=invalid-name
from numpy import linspace, ones_like
from numpy.random import normal

from Stoner import Data
from Stoner.analysis.fitting.models.tunnelling import BDR, bdr

# Make some data
V = linspace(-10, 10, 1000)
I = bdr(V, 2.5, 3.2, 0.3, 15.0, 1.0) + normal(size=len(V), scale=1.0e-3)
dI = ones_like(V) * 1.0e-3

# Curve fit
d = Data(V, I, dI, setas="xye", column_headers=["Bias", "Current", "Noise"])

d.curve_fit(bdr, p0=[2.5, 3.2, 0.3, 15.0, 1.0], result=True, header="curve_fit")
d.setas = "xyey"
d.plot(fmt=["r.", "b-"])
d.annotate_fit(
    bdr,
    x=0.6,
    y=0.05,
    prefix="bdr",
    fontdict={"size": "x-small", "color": "blue"},
)

# lmfit
d.setas = "xy"
fit = BDR(missing="drop")
p0 = fit.guess(I, x=V)
for p, v, mi, mx in zip(
    ["A", "phi", "dphi", "d", "mass"],
    [2.500, 3.2, 0.3, 15.0, 1.0],
    [0.100, 1.0, 0.05, 5.0, 0.5],
    [10, 10.0, 2.0, 30.0, 5.0],
):
    p0[p].value = v
    p0[p].bounds = [mi, mx]
d.lmfit(fit, p0=p0, result=True, header="lmfit")
d.setas = "x...y"
d.plot(fmt="g-")
d.annotate_fit(
    fit,
    x=0.2,
    y=0.05,
    prefix="BDR",
    fontdict={"size": "x-small", "color": "green"},
)

d.ylabel = "Current"
d.title = "BDR Model test"
