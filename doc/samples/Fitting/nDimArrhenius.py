"""Example of nDimArrhenius Fit."""
from Stoner import Data
import Stoner.Fit as SF
from numpy import linspace
from numpy.random import normal

# Make some data
T = linspace(50, 500, 101)
R = SF.nDimArrhenius(T + normal(size=len(T), scale=5.0, loc=1.0), 1e6, 0.5, 2)
d = Data(T, R, setas="xy", column_headers=["T", "Rate"])

# Curve_fit on its own
d.curve_fit(SF.nDimArrhenius, p0=[1e6, 0.5, 2], result=True, header="curve_fit")
d.setas = "xyy"
d.plot(fmt=["r.", "b-"])
d.annotate_fit(SF.nDimArrhenius, x=0.25, y=0.3)

# lmfit using lmfit guesses
fit = SF.NDimArrhenius()
p0 = fit.guess(R, x=T)
d.lmfit(fit, p0=p0, result=True, header="lmfit")
d.setas = "x..y"
d.plot(fmt="g-")
d.annotate_fit(SF.NDimArrhenius, x=0.25, y=0.05, prefix="NDimArrhenius")

d.title = "n-D Arrhenius Test Fit"
d.ylabel = "Rate"
d.xlabel = "Temperature (K)"
