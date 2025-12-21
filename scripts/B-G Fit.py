# -*- coding: utf-8 -*-
"""Sample script for fitting BlochGrueneisen function to a data file.

@author: phygbu
"""
# pylint: disable=invalid-name
import re

import numpy as np

from Stoner import Data
from Stoner.analysis.fitting.models.e_transport import BlochGrueneisen


def select_col(data, message):
    """Select the column with the data."""
    print("Unable to guess column")
    for i, col in enumerate(data.column_headers):
        print("{} : {}".format(i, col))
    while True:
        try:
            return int(input(message))
        except ValueError:
            print("Please select a column number")
            continue


# Load a datafile
d = Data(False)

# These regular expressions are intended to match a tempeerature and resistance coluimns
t_pat = [re.compile(r"^[Tt][\s\(]"), re.compile(r"[Tt]emp")]
r_pat = [re.compile(r"[Rr]ho"), re.compile(r"[Rr]es")]

# Search for each column
for pat in t_pat:
    t_col = d.find_col(pat, force_list=True)
    if len(t_col) == 1:
        t_col = t_col[0]
        break
else:
    t_col = select_col(d, "Select column for temperature data :")

for pat in r_pat:
    r_col = d.find_col(pat, force_list=True)
    if len(r_col) == 1:
        r_col = r_col[0]
        break
else:
    r_col = select_col(d, "Select column for resistance data :")

# Configure the columns in the setas attribute
d.mask = np.isnan(d.data)
d.del_rows()
d.setas(x=t_col, y=r_col)
# Initialise the model, set the n parmeter to be fixed and show our guesses
model = BlochGrueneisen()
model.param_hints["n"] = {"vary": False, "value": 5}
print("Initial guesses: {}".format(model.guess(d.y, x=d.x)))

# Do the fit
popt, pcov = d.lmfit(model, absolute_sigma=False, result=True, header="Bloch")

# Add the Bloch column to the setas
d.setas(Bloch="y", reset=False)
# Make a plot
d.plot(
    fmt=["r.", "b-"], label=["Data", r"$Bloch-Gr\"ueisen Fit$"], markersize=1
)
d.xlabel = "Temperature (K)"
d.ylabel = "Resistance ($\Omega$)"
d.annotate_fit(model, x=0.05, y=0.35, mode="eng")
