"""Replace the JTB template with style and tick configuration."""

from pathlib import Path

from Stoner import Data, __home__

d = Data(Path(__home__).parent / "doc" / "samples" / "sample.txt", setas="xy")

with d.plot.style(["stoner", {"axes.linewidth": 1.5, "lines.linewidth": 2.0}]):
    ax = d.plot.line()
    ax.tick_params(direction="in", top=True, right=True)
