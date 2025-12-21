"""Detect outlying points from a lione."""

# pylint: disable=invalid-name
import numpy as np

from Stoner import Data
from Stoner.analysis.utils import poly_outlier

np.random.seed(12345)
x = np.linspace(0, 100, 201)
y = 0.01 * x**2 + 5 * np.sin(x / 10.0)

i = np.random.randint(len(x) - 20, size=20) + 10
y[i] += np.random.normal(size=len(i), scale=20)

d = Data(np.column_stack((x, y)), column_headers=["x", "y"], setas="xy")
d.plot(fmt="b.", label="raw data")
e = d.clone
e.outlier_detection(window=5, action="delete")
e.plot(fmt="r-", label="Default Outliers removed")
h = d.clone
h.outlier_detection(window=5, action="delete", shape="hamming")
h.plot(color="orange", label="Default Outliers removed with Hamming window")
f = d.clone
f.outlier_detection(
    window=21, order=3, certainty=2, width=3, action="delete", func=poly_outlier
)
f.plot(fmt="g-", label="Poly Outliers removed")
g = d.clone
g = g.outlier_detection(
    window=21, order=3, certainty=3, width=3, action="delete", func=poly_outlier
)
g.plot(color="purple", label="Masked outliers")
g = d.clone
e.title = "Outlier detection test"
