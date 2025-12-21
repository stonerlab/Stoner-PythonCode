"""USe curve_fit to fit a straight line."""

# pylint: disable=invalid-name
from Stoner import Data


def linear(x, m, c):
    """Straight line function."""
    return m * x + c


d = Data("curve_fit_data.dat", setas="xye")
d.plot(fmt="ro")
fit = d.curve_fit(
    linear, result=True, replace=False, header="Fit", output="report"
)
d.setas = "x..y"
d.plot(fmt="b-")
d.annotate_fit(linear)
d.draw()
print(fit.fit_report())
