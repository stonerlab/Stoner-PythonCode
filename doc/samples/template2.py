"""Apply a temporary sketch effect without a template class."""

# pylint: disable=invalid-name
from Stoner import Data

p = Data("sample.txt", setas="xy")
with p.plot.style("stoner", extra={"path.sketch": (1, 100, 2)}):
    p.plot.line()
