"""Scale data to stitch it together.

This example demonstrates specifying the overlap as an integer and scaling both curves to match the other.
"""

# pylint: disable=invalid-name
import matplotlib.pyplot as plt

from Stoner import Data
from Stoner.Util import format_error

# Load and plot two sets of data
s1 = Data("Stitch-scan1.txt", setas="xy")
s2 = Data("Stitch-scan2.txt", setas="xy")
s3 = s2.clone
s1.plot(fmt="b-", label="Set 1", linewidth=1)
s2.fig = s1.fig
s2.plot(fmt="r-", label="Set 2", linewidth=1)

# Stitch scan 2 onto scan 1
s2.stitch(s1, overlap=100)

s2.plot(fmt="b:", label="Set 2 Stictched on Set 1", linewidth=1)
s2.title = "Stictching Example"

# Tidy up the plot by adding annotation fo the stirching co-efficients
labels = ["A", "B", "C"]

txt = []
lead = r"$x'\rightarrow x+A$" + "\n" + r"$y'=\rightarrow By+C$" + "\n"
for l, v, e in zip(
    labels, s2["Stitching Coefficients"], s2["Stitching Coefficient Errors"]
):
    txt.append(format_error(v, e, latex=True, prefix=l + "="))
plt.text(1.5, 1.25, lead + "\n".join(txt), fontdict={"size": "x-small"})

# Now stich s1 onto the clone of s2 and plot and label
s1.stitch(s3, overlap=100)
s1.plot(fmt="r:", label="Set 1 sticked on Set 2", linewidth=1)

txt = []
lead = r"$x'\rightarrow x+A$" + "\n" + r"$y'=\rightarrow By+C$" + "\n"
for l, v, e in zip(
    labels, s1["Stitching Coefficients"], s1["Stitching Coefficient Errors"]
):
    txt.append(format_error(v, e, latex=True, prefix=l + "="))
plt.text(5.0, 1.25, lead + "\n".join(txt), fontdict={"size": "x-small"})
