"""Combine Seaborn's native contexts with Stoner plotting and double axes."""

from pathlib import Path

import seaborn as sns

from Stoner import Data, __home__

d = Data(Path(__home__).parent / "doc" / "samples" / "sample.txt", setas="xyy")
with sns.axes_style("dark"), sns.plotting_context("talk"), sns.color_palette("muted"):
    ax = d.plot.line(y=1, legend=False)
    with d.plot.DoubleYAxis(ax=ax) as right:
        d.plot.line(y=2, ax=right, color="red", legend=False)
