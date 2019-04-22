"""Example of nDimArrhenius Fit."""
from Stoner import Data
import Stoner.Fit as SF
from numpy import linspace, ones_like
from numpy.random import normal

# Make some data
V = linspace(-4, 4, 1001)
I = SF.simmons(V, 2500, 5.2, 15.0) + normal(size=len(V), scale=100e-9)
dI = ones_like(V) * 100e-9

d = Data(V, I, dI, setas="xye", column_headers=["Bias", "Current", "Noise"])

d.curve_fit(SF.simmons, p0=[2500, 5.2, 15.0], result=True, header="curve_fit")
d.setas = "xyey"
d.plot(fmt=["r.", "b-"])
d.annotate_fit(SF.simmons, x=0.25, y=0.25, prefix="simmons", fontdict={"size": "x-small", "color": "blue"})

d.setas = "xye"
fit = SF.Simmons()
p0 = [2500, 5.2, 15.0]
d.lmfit(SF.Simmons, p0=p0, result=True, header="lmfit")
d.setas = "x...y"
d.plot(fmt="g-")
d.annotate_fit(fit, x=0.65, y=0.25, prefix="Simmons", fontdict={"size": "x-small", "color": "green"})

d.ylabel = "Current"
d.title = "Simmons Model test"
d.tight_layout()
