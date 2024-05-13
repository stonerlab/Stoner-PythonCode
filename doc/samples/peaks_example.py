"""Detect peaks in a dataset."""

# pylint: disable=invalid-name,unsubscriptable-object
from matplotlib.cm import jet
from numpy import linspace

from Stoner import Data

d = Data("../../sample-data/New-XRay-Data.dql")
e = d.clone
d.plot()
d.peaks(width=0.08, poly=4, significance=100, modify=True)
d.labels[1] = "Peaks"
d.plot(fmt="ro")
e.find_peaks(modify=True, prominence=80, width=0.01)
e.labels[1] = "Find_Peaks"
e.plot(figure=d.fig, fmt="bx")
d.title = "Finding peaks with peaks() and find_peaks()"

# Use the metadata that find_peaks produces
colors = jet(linspace(0, 1, len(e)), alpha=0.5)
for l, r, c in zip(e["left_ips"], e["right_ips"], colors):
    ax = d.axes[d.ax]
    ax.axvspan(l, r, facecolor=c)
