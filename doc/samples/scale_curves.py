"""Example of using scale to overlap data"""
from Stoner import Data
from numpy import linspace, sin, exp, pi
from numpy.random import normal
import matplotlib as mpl
from tabulate import tabulate

mpl.rc("text", usetex=True)

x = linspace(0, 10 * pi, 201)
x2 = x * 1.5 + 0.23
y = 10 * exp(-x / (2 * pi)) * sin(x) + normal(size=len(x), scale=0.1)
y2 = 3 * exp(-x / (2 * pi)) * sin(x) - 1 + normal(size=len(x), scale=0.1)

d = Data(x, y, column_headers=["Time", "Signal 1"], setas="xy")
d2 = Data(x2, y2, column_headers=["Time", "Signal 2"], setas="xy")

d.plot()
d2.plot(figure=d.fig)
d3 = d2.scale(d, header="Signal 2 scaled", xmode="affine")
d3.plot(figure=d.fig)
d3["test"] = linspace(1, 10, 10)
txt = tabulate(d3["Transform"], floatfmt=".2f", tablefmt="grid")
d3.text(20, 2, "Transform\n{}".format(txt), fontdict={"size": "x-small"})
d3.title = "Scaling Example"
