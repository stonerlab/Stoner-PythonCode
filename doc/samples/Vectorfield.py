"""Create a 2D vector field plot."""
from Stoner import Data,__home__
from os import path
d=Data(path.join(__home__,"..","sample-data","OVF1.ovf"))
e=d.select(Z__between=(10,11)).select(X__between=(10,18)).select(Y__between=(5,13))
e.figure(figsize=(12,4))

#2D vectors on a 2D Field
e.setas="xy.uv."
e.subplot(131)
e.plot()
e.title="2D Vector, 2D Field"

fig=e.fig

#3D Vectors ona 3D Field
e.fig.add_subplot(133,projection="3d")
e.setas="xyzuvw"
e.plot(length=0.4,linewidth=1.5,arrow_length_ratio=.5,pivot="middle")
e.title="3D Vector, 3D Field"
e.tight_layout()

#3D Vector on a 2D Field
e.subplot(132)
e.setas="xy.uvw"
e.plot()
e.title="3D Vector, 2D Field"
