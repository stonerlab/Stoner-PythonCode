"""Customise the colour cycle with a temporary style context."""

# pylint: disable=invalid-name, no-member
from cycler import cycler

from Stoner import Data

p = Data("sample.txt", setas="xy")
with p.plot.style("stoner", extra={"axes.prop_cycle": cycler("color", ["r", "g", "b"])}):
    ax = p.plot.line(label="Original")
    p.y = p.y + 1.0
    p.plot.line(ax=ax, label="Offset by 1")
    p.y = p.y + 1.0
    p.plot.line(ax=ax, label="Offset by 2")
