# -*- coding: utf-8 -*-
"""Script to produce plots from GenX output."""
# pylint: disable=invalid-name
import numpy as np
import matplotlib.pyplot as plt
from Stoner import DataFolder, Data


f = DataFolder(directory=False, pattern="*.dat", type=Data)
f.sort("dataset")
up = f[2]
up &= f[1]
asym = f[0]

up.column_headers = [
    "q_up",
    "Isim_up",
    "I_up",
    "dI_up",
    "q_down",
    "Isim_down",
    "I_down",
    "dI_down",
]

up.plot_xy("q_up", "Isim_up", "r-", plotter=plt.semilogy, label=None)
up.plot_xy("q_down", "Isim_down", "b-", label=None)
up.plot_xy("q_up", "I_up", "ro", yerr="dI_up", label="Up", ms=2.0)
up.plot_xy("q_down", "Isim_down", "bx", yerr="dI_down", label="Down", ms=2.0)

plt.legend()
plt.xlabel("Momentum Transfer Q($\AA^{-1}$)")
plt.ylabel("Normalised Intensity")
plt.title("Spin Up and Down Signals")

up.diffsum("I_up", "I_down", header="SpinAsym_data")
up.diffsum("Isim_up", "Isim_down", header="SpinAsym_sim")
up_c = up.find_col("I_up")
down_c = up.find_col("I_down")
up_ec = up.find_col("dI_up")
down_ec = up.find_col("dI_down")
func = lambda r: np.sqrt(
    (
        1.0 / (r[up_c] + r[down_c])
        - (r[up_c] - r[down_c]) / (r[up_c] + r[down_c]) ** 2
    )
    ** 2
    * r[up_ec] ** 2
    + (
        -1.0 / (r[up_c] + r[down_c])
        - (r[up_c] - r[down_c]) / (r[up_c] + r[down_c]) ** 2
    )
    ** 2
    * r[down_ec] ** 2
)
up.apply(func, up.find_col("SpinAsym_data"), replace=False, header="dSpinAsym")
up.fig = plt.figure()

up.plot_xy(
    "q_up",
    "SpinAsym_data",
    fmt="ro",
    yerr="dSpinAsym",
    label="Spin Asymmetry Data",
    ms=2.0,
)
up.plot_xy("q_up", "SpinAsym_sim", "r-", label="Calculated Spin Asymmetry")
plt.legend()
plt.xlabel("Momentum Transfer Q($\AA^{-1}$)")
plt.ylabel("Spin Asymmetry")
plt.title("Spin Asymmetry Data and Calculated")

asym.add("real sld +", "real sld -", header="Structure")
asym.subtract("real sld +", "real sld -", header="Magnetic")
asym.plot_xy(0, "Structure")
asym.plot_xy(0, "Magnetic")
plt.legend()
plt.xlabel("Distance z\ (nm)")
plt.ylabel("Scattering Potential")
plt.title("Magnetic and Structural Scattering Potentials")
