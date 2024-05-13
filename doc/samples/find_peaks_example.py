"""Detect peaks in a dataset."""

# pylint: disable=invalid-name
from Stoner import Data
from Stoner.analysis.fitting.models.generic import Linear

from matplotlib.cm import jet
from numpy import linspace, log10

d = Data("../../sample-data/New-XRay-Data.dql")
d.y = log10(d.y)
d.smooth(
    windows="hanning", size=4, replace=True, result=True, padtype="constant"
)
d.setas = "x.y"
d.lmfit(Linear, residuals=True, result=True)
d.setas = "x...y"
d.data = d.data[4:-4]
d.plot()
d.find_peaks(
    modify=True, prominence=(0.05, 20), width=(0.02, 0.1), distance=0.01
)
d.labels[1] = "Find_Peaks"
d.plot(figure=d.fig, fmt="bx")
d.title = "Finding peaks with peaks() and find_peaks()"

# Use the metadata that find_peaks produces
colors = jet(linspace(0, 1, len(d)), alpha=0.5)
for l, r, c in zip(d["left_ips"], d["right_ips"], colors):
    ax = d.axes[d.ax]
    ax.axvspan(l, r, facecolor=c)
