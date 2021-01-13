# -*- coding: utf-8 -*-
"""Demonstrate remove duplicates """
from Stoner import Data, __homepath__

# Load data
data = Data(
    __homepath__ / ".." / "sample-data" / "TDI_Format_RT.txt", setas="xy"
)
# Plot the original data
data.plot(fmt="k-", label="Original Data")
# De-dupe the data
data.remove_duplicates(strategy="average")
# Update the plot
data.plot(fmt="r.", label="De-duped data")
data.xlim(8.6, 10.0)
data.ylim(7.4, 9.3)
