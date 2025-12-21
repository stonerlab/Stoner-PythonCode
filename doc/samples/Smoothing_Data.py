"""Smoothing Data methods example."""

# pylint: disable=invalid-name, no-member, not-callable
import matplotlib.pyplot as plt

from Stoner import Data

fig = plt.figure(figsize=(9, 6))

d = Data("Noisy_Data.txt", setas="xy")

d.fig = fig
d.plot(color="grey")
# Filter with Savitsky-Golay filter, linear over 7 ppoints
d.SG_Filter(result=True, points=11, header="S-G Filtered")
# Filter with cubic splines
d.spline(result=True, order=3, smoothing=4, header="Spline")
# Rebin data
d.smooth("hamming", size=0.2, result=True, header="Smoothed")

d.plot(0, "S-G Filtered", lw=2, label="SG Filter")
d.plot(0, "Spline", lw=2, label="Spline")
d.plot(0, "Smooth", lw=2, label="Smoooth", color="green")

d2 = d.bin(bins=100, mode="lin")
d2.fig = d.fig
d2.plot(lw=2, label="Re-binned", color="blue")
d2.xlim(3.5, 6.5)
d2.ylim(-200.0, 400.0)
