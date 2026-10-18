"""Simple use of lmfit to fit data."""

# pylint: disable=invalid-name
from numpy import exp, linspace, random

from Stoner import Data

random.seed(12345)  # Ensure consistent random numbers!
# Make some data
x = linspace(0, 10.0, 101)
y = 2 + 4 * exp(-x / 1.7) + random.normal(scale=0.2, size=101)

d = Data(x, y, column_headers=["Time", "Signal"], setas="xy")


def func(x, A, B, C):
    """Quick fitting function."""
    return A + B * exp(-x / C)


# Do the fitting and plot the result
fit = d.differential_evolution(
    func,
    result=True,
    header="Fit",
    A=1,
    B=1,
    C=1,
    prefix="Model",
    residuals=True,
)

# Use Matplotlib's bundled font for portable mathematical annotations.
with d.plot.style(["stoner", {"font.sans-serif": ["DejaVu Sans"]}]):
    figure = d.plot.new_figure(figsize=(7, 5))
    with d.plot.StackVertical(2, figure=figure, height_ratios=[2, 1], adjust_figsize=False) as panels:

        d.plot.labels = []

        # Make nice two panel plot layout
        ax = panels[1]
        d.setas = "x..y"
        d.plot(fmt="g+")
        ax.set_title("")

        ax = panels[0]
        d.setas = "xyy"
        d.plot(fmt=["r.", "b-"])
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")

        # Annotate plot with fitting parameters
        d.annotate_fit(func, prefix="Model", x=0.7, y=0.3, fontdict={"size": "x-small"})
        text = r"$y=A+Be^{-x/C}$" + "\n\n"
        ax.text(7.2, 3.9, text, fontdict={"size": "x-small"})
        ax.set_title("Differential Evolution  Fit")
