"""Simple use of lmfit to fit data."""

# pylint: disable=invalid-name
from numpy import linspace, random

from Stoner import Data
from Stoner.analysis.fitting import ODR_Model
from Stoner.analysis.fitting.models.generic import PowerLaw

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

ODRModel = ODR_Model(PowerLaw, p0=(1, 1))
fit = d.curve_fit(
    ODRModel,
    result=True,
    header="ODR-Fit",
    residuals=True,
    output="report",
    prefix="ODRModel",
)
with d.plot.style("stoner"):
    figure = d.plot.new_figure(figsize=(7, 5))
    with d.plot.StackVertical(2, figure=figure, height_ratios=[2, 1], adjust_figsize=False) as panels:

        d.plot.labels = []
        # Make nice two panel plot layout
        ax = panels[1]
        d.setas = "x..y"
        d.plot(fmt="g+", label="Fit residuals")
        d.setas = "x....y"
        d.plot(fmt="b+", label="ODRModel Residuals")
        ax.set_title("")

        ax = panels[0]
        d.setas = "xyy.y"
        d.plot(fmt=["ro", "g-", "b-"])
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")

        # Annotate plot with fitting parameters
        d.annotate_fit(PowerLaw, x=0.1, y=0.25, fontdict={"size": "x-small"})
        d.annotate_fit(
            ODRModel, x=0.65, y=0.15, fontdict={"size": "x-small"}, prefix="ODRModel"
        )
        ax.set_title("curve_fit with models")
