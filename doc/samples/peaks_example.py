"""Detect peaks in a dataset."""

from Stoner import Data

d=Data("../../sample-data/New-XRay-Data.dql")
d.plot()
d.peaks(width=8,poly=4,significance=40,modify=True)
d.labels[1]="Peaks"
d.plot(fmt="ro")