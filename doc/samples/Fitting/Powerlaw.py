"""Example of PowerLaw Fit."""

# pylint: disable=invalid-name
from numpy import linspace
from numpy.random import normal

from Stoner import Data
from Stoner.analysis.fitting.models.generic import PowerLaw, powerLaw

# Make some data
T = linspace(50, 500, 101)
R = powerLaw(T, 1e-2, 0.6666666) * normal(size=len(T), scale=0.1, loc=1.0)
d = Data(T, R, setas="xy", column_headers=["T", "Rate"])

# Curve_fit on its own
d.curve_fit(powerLaw, p0=[1, 0.5], result=True, header="curve_fit")
d.setas = "xyy"
d.plot(fmt=["r.", "b-"])
d.annotate_fit(powerLaw, x=0.5, y=0.25)

# lmfit using lmfit guesses
fit = PowerLaw()
p0 = fit.guess(R, x=T)
d.lmfit(fit, p0=p0, result=True, header="lmfit")
d.setas = "x..y"
d.plot(fmt="g-")
d.annotate_fit(PowerLaw, x=0.5, y=0.05, prefix="PowerLaw")

d.title = "Powerlaw Test Fit"
d.ylabel = "Rate"
d.xlabel = "Temperature (K)"
