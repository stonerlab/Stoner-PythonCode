"""Simple use of lmfit to fit data."""

# pylint: disable=invalid-name
from numpy import linspace, random

from Stoner import Data
from Stoner.analysis.fitting.models.generic import PowerLaw
from Stoner.analysis.fitting import odr_Model

# Make some data
x = linspace(0, 10.0, 101)
y = PowerLaw().func(x, 1e-1, 1.64) * random.normal(
    loc=1.0, scale=0.05, size=101
)
d = Data(x, y, column_headers=["Time", "Signal"], setas="xy")

# Do the fitting and plot the result
fit = d.curve_fit(
    PowerLaw,
    result=True,
    header="LM-Model Fit",
    residuals=True,
    output="report",
)

ODRModel = odr_Model(PowerLaw, p0=(1, 1))
fit = d.curve_fit(
    ODRModel,
    result=True,
    header="ODR-Fit",
    residuals=True,
    output="report",
    prefix="ODRModel",
)
# Reset labels
d.labels = []
# Make nice two panel plot layout
d.figure(figsize=(7, 5), no_axes=True)
ax = d.subplot2grid((3, 1), (2, 0))
d.setas = "x..y"
d.plot(fmt="g+", label="Fit residuals")
d.setas = "x....y"
d.plot(fmt="b+", label="ODRModel Residuals")
d.title = ""

ax = d.subplot2grid((3, 1), (0, 0), rowspan=2)
d.setas = "xyy.y"
d.plot(fmt=["ro", "g-", "b-"])
d.xticklabels = [[]]
d.ax_xlabel = ""

# Annotate plot with fitting parameters
d.annotate_fit(PowerLaw, x=0.1, y=0.25, fontdict={"size": "x-small"})
d.annotate_fit(
    ODRModel, x=0.65, y=0.15, fontdict={"size": "x-small"}, prefix="ODRModel"
)
d.title = "curve_fit with models"
