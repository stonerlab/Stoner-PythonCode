"""Simple use of lmfit to fit data."""

# pylint: disable=invalid-name
from numpy import exp, linspace, random
from scipy.odr import Model as odrModel

from Stoner import Data

random.seed(12345)  # Ensure consistent random numbers!

# Make some data
x = linspace(0, 10.0, 101)
y = 2 + 4 * exp(-x / 1.7) + random.normal(scale=0.2, size=101)
x += +random.normal(scale=0.1, size=101)

d = Data(x, y, column_headers=["Time", "Signal"], setas="xy")


model = odrModel(
    lambda beta, x: beta[0] + beta[1] * exp(-x / beta[2]),
    meta={"param_names": ["A", "B", "C"]},
    estimate=[1, 1, 1],
)
# Do the fitting and plot the result
fit = d.odr(
    model,
    result=True,
    header="Fit",
    A=1,
    B=1,
    C=1,
    prefix="Model",
    residuals=True,
)

with d.plot.style("stoner"):
    figure = d.plot.new_figure(figsize=(7, 5))
    with d.plot.StackVertical(2, figure=figure, height_ratios=[2, 1], adjust_figsize=False) as panels:

        d.plot.labels = []

        # Make nice two panel plot layout
        ax = panels[1]
        d.setas = "x..y"
        d.plot(fmt="g+")
        ax.set_title("")

        # Plot up the data
        ax = panels[0]
        d.setas = "xy"
        d.plot(fmt="ro")

        d.setas = "x.y"
        d.plot.errorfill(yerr=0.2, color="orange")
        d.plot.errorfill(xerr=0.1, color="orange", label=None)
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")

        # Annotate plot with fitting parameters
        d.annotate_fit(
            model, prefix="Model", x=0.7, y=0.3, fontdict={"size": "x-small"}
        )
        text = r"$y=A+Be^{-x/C}$" + "\n\n"
        ax.text(7.2, 3.9, text, fontdict={"size": "x-small"})
        ax.set_title("Orthogonal Distance Regression  Fit")
