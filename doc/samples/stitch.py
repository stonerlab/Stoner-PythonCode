"""Scale data to stitch it together."""

# pylint: disable=invalid-name
import matplotlib.pyplot as plt

from Stoner import Data
from Stoner.Util import format_error

# Load and plot two sets of data
s1 = Data("Stitch-scan1.txt", setas="xy")
s2 = Data("Stitch-scan2.txt", setas="xy")
s1.plot(label="Set 1")
s2.fig = s1.fig
s2.plot(label="Set 2")
# Stitch scan 2 onto scan 1
s2.stitch(s1)

s2.plot(label="Stictched")
s2.title = "Stictching Example"

# Tidy up the plot by adding annotation fo the stirching co-efficients
labels = ["A", "B", "C"]
txt = []
lead = r"$x'\rightarrow x+A$" + "\n" + r"$y'=\rightarrow By+C$" + "\n"
for l, v, e in zip(
    labels, s2["Stitching Coefficients"], s2["Stitching Coefficient Errors"]
):
    txt.append(format_error(v, e, latex=True, prefix=l + "="))
plt.text(0.7, 0.65, lead + "\n".join(txt), fontdict={"size": "x-small"})
plt.draw()
