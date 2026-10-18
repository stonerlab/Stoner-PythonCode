"""Use the stonerplots base style."""

from pathlib import Path

from Stoner import Data, __home__

d = Data(Path(__home__).parent / "doc" / "samples" / "sample.txt", setas="xy")

with d.plot.style("stoner"):
    d.plot.line()
