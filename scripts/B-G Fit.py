# -*- coding: utf-8 -*-
"""
Sample script for fitting BlochGrueneisen function to a data file

@author: phygbu
"""
from __future__ import print_function

from Stoner.Core import Data, format_error
from Stoner.compat import python_v3
from Stoner.Fit import blochGrueneisen
import numpy as np
from matplotlib.pyplot import text
import re

if python_v3:
    raw_input = input


def bg_wrapper(T, tD, r0, A):
    """Fit wrapper."""
    return blochGrueneisen(T, tD, r0, A, 5)


def select_col(data, message):
    """Select the column with the data."""
    print("Unable to guess column")
    for i, col in enumerate(data.column_headers):
        print("{} : {}".format(i, col))
    while True:
        try:
            return int(raw_input(message))
        except ValueError:
            print("Please select a column number")
            continue


# Load a datafile
d = Data(False)

t_pat = [re.compile(r"^[Tt][\s\(]"), re.compile(r"[Tt]emp")]
r_pat = [re.compile(r"[Rr]ho"), re.compile(r"[Rr]es")]

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

rho0 = d.min(r_col)[0]
A = rho0 * 40
thetaD = 300.0
p0 = [thetaD, rho0, A]
print("Initial guesses: {}".format(p0))

d.del_rows(0, lambda x, r: np.any(np.isnan(r)))

popt, pcov = d.curve_fit(
    bg_wrapper, xcol=t_col, ycol=r_col, p0=p0, absolute_sigma=False
)
perr = np.sqrt(np.diag(pcov))

labels = [r"\theta_D", r"\rho_0", r"A"]
units = ["K", r"\Omega m", r"\Omega m"]

annotation = [
    "${}$: {}\n".format(l, format_error(v, e, latex=True, mode="eng", units=u))
    for l, v, e, u in zip(labels, popt, perr, units)
]
annotation = "\n".join(annotation)
popt = np.append(popt, 5)
T = d.column(t_col)
d.add_column(blochGrueneisen(T, *popt), header=r"Bloch")

d.plot_xy(
    t_col, [r_col, "Bloch"], ["ro", "b-"], label=["Data", r"$Bloch-Gr\"ueisen Fit$"]
)
d.xlabel = "Temperature (K)"
d.ylabel = "Resistance ($\Omega$)"
text(0.05, 0.65, annotation, transform=d.axes[0].transAxes)
