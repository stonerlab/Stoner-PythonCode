"""Demonstrate Channel math operations."""

# pylint: disable=invalid-name
from numpy import linspace, ones_like, sin, cos, pi
from numpy.random import normal, seed

from Stoner import Data
from Stoner.plot.utils import errorfill

seed(12345)  # Ensure consistent random numbers!
x = linspace(0, 10 * pi, 101) + 1e-9
e = 0.01 * ones_like(x)
y = 0.1 * sin(x) + normal(size=len(x), scale=0.01) + 0.1
e2 = 0.01 * cos(x)
y2 = 0.1 * ones_like(x)
d = Data(
    x,
    y,
    e,
    y2,
    e2,
    column_headers=["$X$", "$Y_+$", r"$\delta Y_+$", "$Y_-$", r"$\delta Y_-$"],
    setas="xyeye",
)

a = tuple(d.column_headers[1:3])
b = tuple(d.column_headers[3:5])

d.add(a, b, replace=False)
d.subtract(a, b, replace=False)
d.multiply(a, b, replace=False)
d.divide(a, b, replace=False)
d.diffsum(a, b, replace=False)
d.setas = "xyeyeyeyeyeyeye"
d.plot(
    multiple="panels",
    plotter=errorfill,
    color="red",
    alpha_fill=0.2,
    figsize=(5, 8),
)
