"""Plot data defined on a matrix."""

# pylint: disable=invalid-name
import numpy as np

from Stoner import Data

x, y = np.meshgrid(np.linspace(-2, 2, 101), np.linspace(-2, 2, 101))
z = np.cos(4 * np.pi * np.sqrt(x**2 + y**2)) * np.exp(-np.sqrt(x**2 + y**2))

p = Data()
p = p & np.linspace(-2, 2, 101) & z
p.column_headers = ["X"]
for i, v in enumerate(np.linspace(-2, 2, 101)):
    p.column_headers[i + 1] = str(v)

p.plot_matrix(xlabel="x", ylabel="y", title="Data as Matrix")
