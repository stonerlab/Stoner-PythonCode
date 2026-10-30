"""Example of nDimArrhenius Fit."""

# pylint: disable=invalid-name
from numpy import linspace
from numpy.random import normal

from Stoner import Data
from Stoner.analysis.fitting.models.thermal import ModArrhenius, modArrhenius

# Make some data
T = linspace(200, 350, 101)
R = modArrhenius(T, 1e6, 0.5, 1.5) * normal(scale=0.00005, loc=1.0, size=len(T))
d = Data(T, R, setas="xy", column_headers=["T", "Rate"])

# Curve_fit on its own
d.fit.curve_fit(modArrhenius, p0=[1e6, 0.5, 1.5], result=True, header="curve_fit")
d.setas = "xyy"
d.plot(fmt=["r.", "b-"])
d.plot.annotate_fit(modArrhenius, x=0.2, y=0.5, mode="eng")

# lmfit using lmfit guesses
fit = ModArrhenius()
p0 = [1e6, 0.5, 1.5]
d.setas = "xy"
d.fit.lmfit(fit, result=True, header="lmfit")
d.setas = "x..y"
d.plot()
d.plot.annotate_fit(ModArrhenius, x=0.2, y=0.25, prefix="ModArrhenius", mode="eng")

d.plot.title = "Modified Arrhenius Test Fit"
d.plot.ylabel = "Rate"
d.plot.xlabel = "Temperature (K)"
