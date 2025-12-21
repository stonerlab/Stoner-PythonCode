"""Example of using lmfit to do a bounded fit."""

# pylint: disable=invalid-name
from Stoner import Data
from Stoner.analysis.fitting.models.generic import StretchedExp

# Load dat and plot
d = Data("lmfit_data.txt", setas="xy")
d.y = d.y

# Do the fit
d.lmfit(StretchedExp, result=True, header="Fit", residuals=True)
# plot
d.setas = "xyy"

d.plot(fmt=["+", "-"])
# Make apretty label using Stoner.Util methods
text = r"$y=A e^{-\left(\frac{x}{x_0}\right)^\beta}$" + "\n"
text += d.annotate_fit(StretchedExp, text_only=True)
d.text(4, 6e4, text, fontsize="x-small")
