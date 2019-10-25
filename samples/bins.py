"""Re-binning data example."""
from Stoner import Data
from Stoner.plot.utils import errorfill

d = Data("Noisy_Data.txt", setas="xy")

d.template.fig_height = 6
d.template.fig_width = 8
d.figure(figsize=(6, 8))
d.subplot(411)

e = d.bin(bins=0.05, mode="lin")
f = d.bin(bins=0.25, mode="lin")
g = d.bin(bins=0.05, mode="log")
h = d.bin(bins=50, mode="log")

for i, (binned, label) in enumerate(
    zip([e, f, g, h], ["0.05 Linear", "0.25 Linear", "0.05 Log", "50 log"])
):
    binned.subplot(411 + i)
    d.plot()
    binned.fig = d.fig
    binned.plot(plotter=errorfill, label=label)

    d.xlim = (1, 6)
    d.ylim(-0.1, 0.4)
    d.title = "Bin demo" if i == 0 else ""
d.tight_layout()
