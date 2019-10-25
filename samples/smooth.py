"""Smoothing Data methods example."""
from Stoner import Data
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9, 6))

d = Data("Noisy_Data.txt", setas="xy")

d.fig = fig
d.plot(color="grey")
# Filter with Savitsky-Golay filter, linear over 7 ppoints
d.SG_Filter(result=True, points=11, header="S-G Filtered")
d.setas = "x.y"
d.plot(lw=2, label="SG Filter")
d.setas = "xy"
# Filter with cubic splines
d.spline(replace=2, order=3, smoothing=4, header="Spline")
d.setas = "x.y"
d.plot(lw=2, label="Spline")
d.setas = "xy"
# Rebin data
d.smooth("hamming", size=0.2, result=True, replace=False, header="Smoothed")
d.setas = "x...y"
d.plot(lw=2, label="Smoooth", color="green")
d.setas = "xy"
d2 = d.bin(bins=100, mode="lin")
d2.fig = d.fig
d2.plot(lw=2, label="Re-binned", color="blue")
d2.xlim(3.5, 6.5)
d2.ylim(-0.2, 0.4)
