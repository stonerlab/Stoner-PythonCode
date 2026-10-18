"""Paginate independent measurements and overlay named bulk plotting calls."""

import numpy as np

from Stoner import Data
from Stoner.folders import PlotFolder

folder = PlotFolder(readlist=False)
x = np.linspace(0, 2*np.pi, 60)
for index in range(5):
    data = Data(x, np.sin(x+index/3), setas="xy", filename=f"Measurement {index+1}",
                column_headers=["Phase / rad", "Signal"])
    data["Run"] = index+1
    folder += data

with folder[0].plot.style("stoner"):
    axes = folder.plot(plots_per_page=3, figsize=(8, 5))
    figure = folder[0].plot.new_figure(figsize=(6, 4))
    overlay = figure.subplots()
    folder.each.plot.line(ax=overlay, label="Run {Run}", title="Overlay of the five measurements")
