"""Extrapolate data example."""
from Stoner import Data
from numpy import linspace, ones_like, column_stack
from numpy.random import normal
import matplotlib.pyplot as plt
from Stoner.plot.utils import errorfill

x = linspace(1, 10, 101)
d = Data(
    column_stack(
        [x, (2 * x ** 2 - x + 2) + normal(size=len(x), scale=2.0), ones_like(x) * 2]
    ),
    setas="xye",
    column_headers=["x", "y"],
)

extra_x = linspace(8, 15, 11)

y3 = d.extrapolate(extra_x, overlap=80, kind="cubic")
y2 = d.extrapolate(extra_x, overlap=3.0, kind="quadratic")
y1 = d.extrapolate(extra_x, overlap=1.0, kind="linear")

d.plot(fmt="k.")
d.title = "Extrapolation Demo"

errorfill(
    extra_x, y2[:, 0], color="green", yerr=y2[:, 1], label="Quadratic Extrapolation"
)
errorfill(extra_x, y3[:, 0], color="red", yerr=y3[:, 1], label="Cubic Extrapolation")
errorfill(extra_x, y1[:, 0], color="blue", yerr=y1[:, 1], label="Linear Extrapolation")
plt.legend(loc=2)
