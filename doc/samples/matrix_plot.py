"""Render every matrix sample without interpolating new values."""

# pylint: disable=invalid-name
import numpy as np

from Stoner import Data

x, y = np.meshgrid(
    np.linspace(-2, 2, 101), np.linspace(-2, 2, 101), indexing="ij"
)
z = np.cos(4 * np.pi * np.sqrt(x**2 + y**2)) * np.exp(-np.sqrt(x**2 + y**2))

p = Data()
p = p & np.linspace(-2, 2, 101) & z
p.column_headers[0] = "X"
for i, v in enumerate(np.linspace(-2, 2, 101)):
    p.column_headers[i + 1] = str(v)

# plot_surface may otherwise downsample a large matrix. The renderer's rows
# follow Y and its columns follow X, opposite to the stored matrix's X/Y order.
# Antialiasing smooths polygon edges; it does not interpolate matrix values.
ax = p.plot.matrix(
    rcount=z.shape[1],
    ccount=z.shape[0],
    linewidth=0,
    antialiased=True,
    xlabel="x",
    ylabel="y",
    title="Matrix at full sample resolution",
)
