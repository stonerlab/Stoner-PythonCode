"""Apply a sketch effect through a temporary rcParams style."""

from pathlib import Path

from Stoner import Data, __home__

d = Data(Path(__home__).parent / "doc" / "samples" / "sample.txt", setas="xy")

with d.plot.style("stoner", extra={"path.sketch": (1, 100, 2)}):
    d.plot.line()
