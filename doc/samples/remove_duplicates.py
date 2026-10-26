# -*- coding: utf-8 -*-
"""Demonstrate remove duplicates."""

# pylint: disable=no-member
from Stoner import Data, __homepath__

# Load data
data = Data(
    __homepath__ / ".." / "sample-data" / "TDI_Format_RT.txt", setas="xye"
)
data.e = data.e / 10000
# Plot the original data
data.plot(fmt="k-", label="Original Data", capsize=3)
# De-dupe the data
data.remove_duplicates(strategy="average")
# Update the plot
data.plot(fmt="r.", label="De-duped data", capsize=3)
data.plot.xlim = (8.6, 10.0)
data.plot.ylim = (7.4, 9.3)
