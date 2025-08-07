"""Re-binning data example."""

# pylint: disable=invalid-name
from Stoner import Data
from Stoner.plot.utils import errorfill

d = Data("Noisy_Data.txt", setas="xy")

d.template.fig_height = 6
d.template.fig_width = 8
d.figure(figsize=(6, 8), no_axes=True)
d.subplot(411)

e = d.bin(bins=0.05, mode="lin")
f = d.bin(bins=0.25, mode="lin")
d.setas = "xye"
g = d.bin(bins=0.05, mode="log")
h = d.bin(bins=50, mode="log")

for i, (binned, label) in enumerate(
    zip([e, f, g, h], ["0.05 Linear", "0.25 Linear", "0.05 Log", "50 log"])
):
    binned.subplot(411 + i)
    d.plot(fmt="k,", capsize=2.0)
    binned.fig = d.fig
    binned.data = binned.data[binned.data[:, 2] != 0]
    binned.plot(plotter=errorfill, label=label, color="red")

    d.xlim = (1, 6)
    d.ylim(-100.0, 400)
    d.title = "Bin demo" if i == 0 else ""
