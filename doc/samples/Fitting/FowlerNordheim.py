"""Example of nDimArrhenius Fit."""

# pylint: disable=invalid-name
from numpy import linspace, ones_like
from numpy.random import normal

from Stoner import Data
from Stoner.analysis.fitting.models.tunnelling import (
    fowlerNordheim,
    FowlerNordheim,
)

# Make some data
V = linspace(-4, 4, 1000)
I = fowlerNordheim(V, 2500, 3.2, 15.0) + normal(size=len(V), scale=1e-6)
dI = ones_like(V) * 10e-6

d = Data(V, I, dI, setas="xye", column_headers=["Bias", "Current", "Noise"])

d.curve_fit(
    fowlerNordheim, p0=[2500, 3.2, 15.0], result=True, header="curve_fit"
)
d.setas = "xyey"
d.plot(fmt=["r.", "b-"])
d.annotate_fit(
    fowlerNordheim,
    x=0.2,
    y=0.6,
    prefix="fowlerNordheim",
    fontdict={"size": "x-small", "color": "blue"},
)

d.setas = "xye"
fit = FowlerNordheim()
p0 = [2500, 5.2, 15.0]
p0 = fit.guess(I, x=V)
for p, v, mi, mx in zip(
    ["A", "phi", "d"], [2500, 3.2, 15.0], [100, 1, 5], [1e4, 20.0, 30.0]
):
    p0[p].value = v
    p0[p].bounds = [mi, mx]
d.lmfit(FowlerNordheim, p0=p0, result=True, header="lmfit")
d.setas = "x...y"
d.plot(fmt="g-")
d.annotate_fit(
    fit,
    x=0.2,
    y=0.2,
    prefix="FowlerNordheim",
    fontdict={"size": "x-small", "color": "green"},
)

d.ylabel = "Current"
d.title = "Fowler-Nordheim Model test"
