"""Replace the experimental GB template with explicit centred axes."""

from pathlib import Path

from Stoner import Data, __home__

d = Data(Path(__home__).parent / "doc" / "samples" / "sample.txt", setas="xy")

d.y = d.y - max(d.y) / 2
with d.plot.style("stoner"), d.plot.CentredAxes():
    d.plot.line()
