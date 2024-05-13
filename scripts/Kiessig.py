"""A very quick and dirty Kiessig Fringe Analysis code.

Gavin Burnell 28/12/2010

TODO: Implement an error bar on the uncertainity by understanding the significance of the covariance terms
"""

# pylint: disable=invalid-name
import sys
from copy import copy

import numpy as np
import matplotlib.pyplot as pyplot
from lmfit.models import ExponentialModel

from Stoner import Data
from Stoner.analysis.fitting.models.generic import Linear
from Stoner.Util import format_error

filename = False
sensitivity = 50

critical_edge = 0.8
fringe_offset = 1

d = Data(filename, setas="xy")  # Load the low angle scan

# Now get the section of the data file that has the peak positions
# This is really doing the hard work
# We differentiate the data using a Savitsky-Golay filter with a 5 point window fitting quartics.
# This has proved most succesful for me looking at some MdV data.
# We then threshold for zero crossing of the derivative
# And check the second derivative to see whether we like the peak as signficant. This is the significance parameter
# and seems to be largely empirical
# Finally we interpolate back to the complete data set to make sure we get the angle as well as the counts.
d.lmfit(ExponentialModel, result=True, replace=False, header="Envelope")
d.subtract("Counts", "Envelope", replace=False, header="peaks")
d.setas = "xy"
sys.exit()
t = Data(d.interpolate(d.peaks(significance=sensitivity, width=8, poly=4)))

t.column_headers = copy(d.column_headers)
d %= "peaks"
t %= "peaks"
d.setas = "xy"
d.labels[d.find_col("Angle")] = r"Reflection Angle $\theta$"
t.del_rows(0, lambda x, y: x < critical_edge)
t.setas = "xy"
t.template.fig_width = 7.0
t.template.fig_height = 5.0
t.plot(fmt="go", plotter=pyplot.semilogy)
main_fig = d.plot(figure=t.fig, plotter=pyplot.semilogy)
d.show()
# Now convert the angle to sin^2
t.apply(
    lambda x: np.sin(np.radians(x[0] / 2.0)) ** 2, 0, header=r"$sin^2\theta$"
)
# Now create the m^2 order
m = np.arange(len(t)) + fringe_offset
m = m**2
# And add it to t
t.add_column(m, column_header="$m^2$")
# Now we can it a straight line
t.setas = "x..y"
fit = t.lmfit(Linear, result=True, replace=False, header="Fit")
g = t["LinearModel:slope"]
gerr = t["LinearModel:slope err"] / g
g = np.sqrt(1.0 / g)
gerr /= 2.0
l = float(d["Lambda"])
th = l / (2 * g)
therr = th * (gerr)

t.inset(loc="top right", width=0.5, height=0.4)
t.plot_xy(r"Fit", r"$sin^2\theta$", "b-", label="Fit")
t.plot_xy(r"$m^2$", r"$sin^2\theta$", "ro", label="Peak Position")
t.xlabel = "Fringe $m^2$"
t.ylabel = r"$sin^2\theta$"
t.title = ""
t.legend(loc="upper left")
t.draw()
pyplot.sca(t.axes[0])
# Get the wavelength from the metadata
# Calculate thickness and report
pyplot.text(
    0.05,
    0.05,
    "Thickness is: {} $\AA$".format(format_error(th, therr, latex=True)),
    transform=main_fig.axes[0].transAxes,
)
