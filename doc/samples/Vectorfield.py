"""Create a 2D vector field plot."""

# pylint: disable=invalid-name
from os import path

from Stoner import Data, __home__

d = Data(path.join(__home__, "..", "sample-data", "OVF1.ovf"))
e = (
    d.select(Z__between=(10, 11))
    .select(X__between=(10, 18))
    .select(Y__between=(5, 13))
)
e.figure(figsize=(8, 4), no_axes=True)

# 2D vectors on a 2D Field
e.setas = "xy.uv."
e.subplot(121)
e.plot()
e.title = "3D Vector, 2D Field"

# 3D Vector on a 2D Field
e.subplot(122)
e.setas = "xy.uvw"
e.plot()
e.title = "3D Vector, 3D Field"
