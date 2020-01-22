"""Example of using lmfit to do a bounded fit."""
from Stoner import Data
from Stoner.Fit import StretchedExp

# Load dat and plot
d = Data("lmfit_data.txt", setas="xy")

# Do the fit
d.lmfit(StretchedExp, result=True, header="Fit", prefix="")
# plot
d.setas = "xyy"

d.plot(fmt=["+", "-"])
# Make apretty label using Stoner.Util methods
text = r"$y=A e^{-\left(\frac{x}{x_0}\right)^\beta}$" + "\n"
text += d.annotate_fit(StretchedExp, text_only=True, prefix="")
d.text(6, 4e4, text)
