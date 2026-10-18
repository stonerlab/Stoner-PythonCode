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
with e.plot.style("stoner"):
    figure = e.plot.new_figure(figsize=(8, 4))
    with e.plot.MultiPanel((1, 2), figure=figure, adjust_figsize=False) as panels:
        e.plot.vector(x=0, y=1, u=3, v=4, w=False, ax=panels[0], title="U/V on a 2D slice")
        e.plot.vector(x=0, y=1, u=3, v=4, w=5, ax=panels[1], title="U/V/W colour on a 2D slice")
