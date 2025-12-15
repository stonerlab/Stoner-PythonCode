"""Simple use of lmfit to fit data."""

# pylint: disable=invalid-name
from numpy import exp, linspace, random

from Stoner import Data
from Stoner.plot.utils import errorfill

random.seed(12345)  # Ensure consistent random numbers!


# Make some data
def func(x, A, B, C):
    """Simple fitting function."""
    return A + B * exp(-x / C)


x = linspace(0, 10.0, 101)
y = func(x, 2, 4, 1.5) + random.normal(scale=0.2, size=101)
x += +random.normal(scale=0.1, size=101)

d = Data(x, y, column_headers=["Time", "Signal"], setas="xy")


# Do the fitting and plot the result
fit = d.odr(
    func,
    result=True,
    header="Fit",
    A=1,
    B=3,
    C=1.2,
    prefix="Model",
    residuals=True,
)

# Reset labels
d.labels = []

# Make nice two panel plot layout
ax = d.subplot2grid((3, 1), (2, 0))
d.setas = "x..y"
d.plot(fmt="g+")
d.title = ""

# Plot up the data
ax = d.subplot2grid((3, 1), (0, 0), rowspan=2)
d.setas = "xy"
d.plot(fmt="ro")

d.setas = "x.y"
d.plot(plotter=errorfill, yerr=0.2, color="orange")
d.plot(plotter=errorfill, xerr=0.1, color="orange", label=None)
d.xticklabels = [[]]
d.xlabel = ""

# Annotate plot with fitting parameters
d.annotate_fit(func, prefix="Model", x=0.7, y=0.3, fontdict={"size": "x-small"})
text = r"$y=A+Be^{-x/C}$" + "\n\n"
d.text(7.2, 3.9, text, fontdict={"size": "x-small"})
d.title = "Orthogonal Distance Regression  Fit"
