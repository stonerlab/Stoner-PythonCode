"""Extrapolate data example."""

# pylint: disable=invalid-name
from numpy import linspace, ones_like, column_stack, exp, sqrt
from numpy.random import normal, seed
import matplotlib.pyplot as plt

from Stoner import Data
from Stoner.plot.utils import errorfill

seed(1245)  # Ensure consistent random numbers
x = linspace(1, 10, 101)
d = Data(
    column_stack(
        [
            x,
            (2 * x**2 - x + 2) + normal(size=len(x), scale=2.0),
            ones_like(x) * 2,
        ]
    ),
    setas="xye",
    column_headers=["x", "y", "errors"],
)

extra_x = linspace(8, 15, 11)

y4 = d.extrapolate(
    extra_x,
    overlap=200,
    kind=lambda x, A, C: A * exp(x / 10) + C,
    errors=lambda x, Aerr, Cerr, popt: sqrt(
        (Aerr * exp(x / popt[1])) ** 2 + Cerr**2
    ),
)
y3 = d.extrapolate(extra_x, overlap=80, kind="cubic")
y2 = d.extrapolate(extra_x, overlap=3.0, kind="quadratic")
y1 = d.extrapolate(extra_x, overlap=1.0, kind="linear")

d.plot(fmt="k.", capsize=2.0)
d.title = "Extrapolation Demo"

errorfill(
    extra_x, y4[:, 0], color="orange", yerr=y4[:, 1], label="Arg. Function"
)

errorfill(
    extra_x, y3[:, 0], color="red", yerr=y3[:, 1], label="Cubic Extrapolation"
)

errorfill(
    extra_x,
    y2[:, 0],
    color="green",
    yerr=y2[:, 1],
    label="Quadratic Extrapolation",
)
errorfill(
    extra_x, y1[:, 0], color="blue", yerr=y1[:, 1], label="Linear Extrapolation"
)
plt.legend(loc=2)
