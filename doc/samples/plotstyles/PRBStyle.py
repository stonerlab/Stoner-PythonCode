"""Use the stonerplots APS journal style in place of the old PRB template."""

from pathlib import Path

from Stoner import Data, __home__

d = Data(Path(__home__).parent / "doc" / "samples" / "sample.txt", setas="xy")

# APS is a replacement style, not a pixel-equivalent alias for the PRB template.
with d.plot.style(["stoner", "aps"]):
    d.plot.line()
