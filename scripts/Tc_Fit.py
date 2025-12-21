# -*- coding: utf-8 -*-
"""Simple code to fit critical temperatures in R(T) data."""
# pylint: disable=invalid-name
import numpy as np

from Stoner import Data
from Stoner.analysis.fitting.models.generic import linear


filename = (
    "../sample-data/6221-Lockin-DAQ Temperature Control -30.0Deg 0.004T.txt"
)

t_col = ": T2"  # Temperature column label
r_cols = ("Nb::R", "YIG-Nb-Co::R")  # Resistance Column Labels
iterator = "iterator"  # Temperature ramp iteration column label
threshold = 0.85  # Fraction of transition to fit to

data = Data(
    filename
)  # Use FALSE to get a dialog box to the file containing Tc data

# Define my working x and y axes
# Split one file into a folder of two files by the iterator column
fldr = data.split(iterator)

result = Data()
for data in fldr:  # For each iteration ramp in the Tc data
    if len(data) == 0:
        continue
    row = [data.mean(iterator)]
    data.figure(figsize=(8, 4))
    for i, r_col in enumerate(r_cols):
        data.setas(x=t_col, y=r_col)
        data.del_rows(~np.isnan(data.y))

        # Normalise data on y axis between +/- 1
        data.normalise(base=(-1.0, 1.0), replace=True)

        # Swap x and y axes around so that R is x and T is y
        data = ~data

        # Curve fit a straight line, using only the central 90% of the resistance transition
        data.curve_fit(
            linear,
            bounds=lambda x, r: -threshold < x < threshold,
            result=True,
            p0=[7.0, 0.0],
        )  # result=True to record fit into metadata

        # Plot the results
        data.setas[-1] = "y"
        data.subplot(1, len(r_cols), i + 1)
        data.plot(fmt=["k.", "r-"])
        data.annotate_fit(linear, x=0.1, y=0.1, fontsize="small")
        data.title = f"Ramp {row[0]}"
        row.extend([data["linear:intercept"], data["linear:intercept err"]])
    data.tight_layout()
    result += np.array(row)

result.column_headers = ["Ramp", "Sample 4 R", "dR", "Sample 7 R", "dR"]
result.setas = "xyeye"
result.plot(fmt=["k.", "r."])
