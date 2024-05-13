"""Example of using scale to overlap data."""

# pylint: disable=invalid-name, no-member
from numpy import linspace, sin, exp, pi, column_stack
from numpy.random import normal, seed
import matplotlib as mpl
from tabulate import tabulate

from Stoner import Data

seed(3)  # Just fix the random numbers to stop optimizer warnings
mpl.rc("text", usetex=True)


x = linspace(0, 10 * pi, 201)
x2 = x * 1.5 + 0.23
y = 10 * exp(-x / (2 * pi)) * sin(x) + normal(size=len(x), scale=0.1)
y2 = 3 * exp(-x / (2 * pi)) * sin(x) - 1 + normal(size=len(x), scale=0.1)

d = Data(x, y, column_headers=["Time", "Signal 1"], setas="xy")
d2 = Data(x2, y2, column_headers=["Time", "Signal 2"], setas="xy")

d.plot(label="1$^\\mathrm{st}$ signal")
d2.plot(figure=d.fig, label="2$^\\mathrm{nd}$ signal")
d3 = d2.scale(d, header="Signal 2 scaled", xmode="affine")
d3.plot(figure=d.fig, label="1$^\\mathrm{st}$ scaled signals")
d3["test"] = linspace(1, 10, 10)
txt = tabulate(d3["Transform"], floatfmt=".2f", tablefmt="grid")
d3.text(10, 4, "Transform\n{}".format(txt), fontdict={"size": "x-small"})

np_data = column_stack((x2, y2))
d4 = d.scale(
    np_data, header="Signal 2 scaled", xmode="affine", use_estimate=True
)
d4.plot(figure=d.fig, label="2$^\\mathrm{nd}$ scaled signal")
d4.ylim = (-7, 9)
txt = tabulate(d4["Transform"], floatfmt=".2f", tablefmt="grid")
d4.text(10, -7, "Transform\n{}".format(txt), fontdict={"size": "x-small"})

d4.title = "Scaling Example"
