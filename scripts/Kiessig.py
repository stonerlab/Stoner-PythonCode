"""Estimate film thickness from Kiessig fringes in a low-angle reflection scan.

The empirical peak sensitivity and critical edge should be chosen for the scan.
The reported uncertainty propagates the fitted slope uncertainty only.
"""

import numpy as np
from lmfit.models import ExponentialModel

from Stoner import Data
from Stoner.analysis.fitting.models.generic import Linear
from Stoner.analysis.utils import format_error


def analyse(filename=False, sensitivity=50, critical_edge=0.8, fringe_offset=1):
    """Fit the fringe order and return the scan, peak table and thickness in angstroms."""
    data = Data(filename, setas="xy")
    data.lmfit(ExponentialModel, result=True, replace=False, header="Envelope")
    data.subtract("Counts", "Envelope", replace=False, header="peaks")
    data.setas = "xy"
    positions = data.peaks(ycol="peaks", xcol=0, significance=sensitivity, width=8, poly=4, full_data=False)
    peaks = Data(data.interpolate(positions, xcol=0),
                 column_headers=list(data.column_headers))
    data %= "peaks"
    peaks %= "peaks"
    peaks.del_rows(0, lambda x, row: x < critical_edge)
    if len(peaks) < 3:
        raise ValueError("At least three fringes above the critical edge are required")
    data.setas = peaks.setas = "xy"
    figure = data.plot.new_figure(figsize=(7, 5))
    ax = data.plot.line(ax=figure.add_subplot(), title=None)
    peaks.plot.line(ax=ax, fmt="go", label="Peak positions", title=None)
    ax.set(xlabel=r"Reflection angle $2\theta$", yscale="log")

    peaks.apply(lambda row: np.sin(np.radians(row[0] / 2.0)) ** 2, 0, header=r"$\sin^2\theta$")
    peaks.add_column((np.arange(len(peaks)) + fringe_offset) ** 2, header="$m^2$")
    peaks.setas(x="$m^2$", y=r"$\sin^2\theta$")
    fit = peaks.lmfit(Linear, result=True, replace=False, header="Fit", output="report")
    slope = fit.params["slope"].value
    thickness = float(data["Lambda"]) / (2 * np.sqrt(slope))
    uncertainty = thickness * fit.params["slope"].stderr / (2 * slope)
    with peaks.plot.InsetPlot(ax=ax, loc="upper right", width=0.5, height=0.4) as inset:
        peaks.plot.line(x="$m^2$", y="Fit", ax=inset, fmt="b-", label="Fit", title=None)
        peaks.plot.line(x="$m^2$", y=r"$\sin^2\theta$", ax=inset, fmt="ro", label="Peak positions", title=None)
        inset.set(xlabel="Fringe $m^2$", ylabel=r"$\sin^2\theta$")
    ax.text(0.05, 0.05, rf"Thickness: {format_error(thickness, uncertainty, latex=True)} $\AA$",
            transform=ax.transAxes)
    return data, peaks, thickness, uncertainty


if __name__ == "__main__":
    analyse()[0].plot.show()
