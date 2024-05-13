"""Test langevin fitting."""

# pylint: disable=invalid-name
from numpy import linspace, ones_like, where, nan, isnan
from numpy.random import normal

from Stoner import Data
from Stoner.analysis.fitting.models.magnetism import blochLaw, BlochLaw

T = linspace(10, 1000, 100)
Ms = 1714
Tc = 1046
M = where(
    T > 300, nan, blochLaw(T, Ms, Tc) + normal(scale=Ms / 200, size=len(T))
)
dM = ones_like(T) * Ms / 200

d = Data(
    T, M, dM, column_headers=["Temperature", "Magnetization", "dM"], setas="xye"
)

d.curve_fit(
    blochLaw,
    p0=[1500, 1500],
    result=True,
    header="curve_fit",
    prefix="curve_fit",
    bounds=lambda x, r: not isnan(r.y),
)

model = BlochLaw()

fit = d.lmfit(
    model,
    result=True,
    header="lmfit",
    prefix="lmfit",
    output="report",
    bounds=lambda x, r: not isnan(r.y),
    g=2.01,
)


d.setas = "xyeyy"
d.plot(fmt=["r.", "b-", "g-"])

d.annotate_fit(
    blochLaw,
    x=0.1,
    y=0.5,
    fontdict={"size": "x-small", "color": "blue"},
    mode="eng",
    prefix="curve_fit",
)
d.annotate_fit(
    BlochLaw,
    x=0.1,
    y=0.25,
    fontdict={"size": "x-small", "color": "green"},
    prefix="lmfit",
    mode="eng",
)
d.title = "BlochLaw Fit"
