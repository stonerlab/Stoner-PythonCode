"""Use curve_fit to fit a plane to some data."""
# pylint: disable=invalid-name
from numpy.random import normal
from numpy import linspace, meshgrid, column_stack, array
import matplotlib.cm as cmap
import matplotlib.pyplot as plt

from Stoner import Data


def plane(coord, a, b, c):
    """Function to define a plane."""
    x, y = coord.T
    return c - (x * a + y * b)


coeefs = [1, -0.5, -1]
col = linspace(-10, 10, 8)
X, Y = meshgrid(col, col)
Z = plane(array([X, Y]).T, *coeefs) + normal(size=X.shape, scale=2.0)
d = Data(
    column_stack((X.ravel(), Y.ravel(), Z.ravel())),
    filename="Fitting a Plane",
    setas="xyz",
)

d.column_headers = ["X", "Y", "Z"]
d.figure(projection="3d")
d.plot_xyz(plotter="scatter")

popt, pcov = d.curve_fit(plane, [0, 1], 2, result=True)
d.setas = "xy.z"

d.plot_xyz(linewidth=0, cmap=cmap.jet)

txt = "$z=c-ax+by$\n"
txt += "\n".join(
    [d.format("plane:{}".format(k), latex=True) for k in ["a", "b", "c"]]
)

ax = plt.gca(projection="3d")
ax.text(15, 5, -50, txt)
d.draw()
