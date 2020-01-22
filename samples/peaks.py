"""Detect peaks in a dataset."""

from Stoner import Data

d = Data("../../sample-data/New-XRay-Data.dql")
d.plot()
d.peaks(width=0.08, poly=4, significance=100, modify=True)
d.labels[1] = "Peaks"
d.plot(fmt="ro")
