"""Example of nDimArrhenius Fit."""

# pylint: disable=invalid-name
from numpy import linspace
from numpy.random import normal

from Stoner import Data
from Stoner.analysis.fitting.models.thermal import nDimArrhenius, NDimArrhenius

# Make some data
T = linspace(50, 500, 101)
R = nDimArrhenius(T + normal(size=len(T), scale=5.0, loc=1.0), 1e6, 0.5, 2)
d = Data(T, R, setas="xy", column_headers=["T", "Rate"])

# Curve_fit on its own
d.curve_fit(nDimArrhenius, p0=[1e6, 0.5, 2], result=True, header="curve_fit")
d.setas = "xyy"
d.plot(fmt=["r.", "b-"])
d.annotate_fit(nDimArrhenius, x=0.25, y=0.3, mode="eng")

# lmfit using lmfit guesses
fit = NDimArrhenius()
p0 = fit.guess(R, x=T)
d.lmfit(fit, result=True, header="lmfit")
d.setas = "x..y"
d.plot(fmt="g-")
d.annotate_fit(
    NDimArrhenius, x=0.25, y=0.05, prefix="NDimArrhenius", mode="eng"
)

d.title = "n-D Arrhenius Test Fit"
d.ylabel = "Rate"
d.xlabel = "Temperature (K)"
