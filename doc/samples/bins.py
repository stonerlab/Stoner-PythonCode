"""Re-binning data example."""

# pylint: disable=invalid-name
from Stoner import Data

d = Data("Noisy_Data.txt", setas="xy")

figure = d.plot.new_figure(figsize=(6, 8), layout="constrained")
axes = figure.subplots(4, 1, sharex=True)

e = d.bin(bins=0.05, mode="lin")
f = d.bin(bins=0.25, mode="lin")
d.setas = "xye"
g = d.bin(bins=0.05, mode="log")
h = d.bin(bins=50, mode="log")

for i, (binned, label) in enumerate(
    zip([e, f, g, h], ["0.05 Linear", "0.25 Linear", "0.05 Log", "50 log"])
):
    binned.data = binned.to_numpy()[binned.to_numpy()[:, 2] != 0]
    ax = axes[i]
    d.plot.line(fmt="k,", ax=ax)
    binned.plot.errorfill(label=label, color="red", ax=ax)
    ax.set(xlim=(1, 6), ylim=(-100, 400), title="Bin demo" if i == 0 else "")
