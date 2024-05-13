"""Use curve_fit to fit a plane to some data."""

# pylint: disable=invalid-name
from numpy.random import normal, seed
from numpy import linspace, meshgrid, column_stack, array
import matplotlib.cm as cmap

from Stoner import Data

seed(12345)  # Ensire consistent random numbers!


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
d.plot_xyz(plotter="scatter", title=None, griddata=False, color="k")

popt, pcov = d.curve_fit(plane, [0, 1], 2, result=True)
col = linspace(-10, 10, 128)
X, Y = meshgrid(col, col)
Z = plane(array([X, Y]).T, *popt)
e = Data(
    column_stack((X.ravel(), Y.ravel(), Z.ravel())),
    filename="Fitting a Plane",
    setas="xyz",
)
e.column_headers = d.column_headers

e.plot_xyz(linewidth=0, cmap=cmap.jet, alpha=0.5, figure=d.fig)

txt = "$z=c-ax+by$\n"
txt += "\n".join(
    [d.format("plane:{}".format(k), latex=True) for k in ["a", "b", "c"]]
)

ax = d.axes[0]
ax.text(-30, -10, 10, txt)
