"""Example using select method to pick out data."""

# pylint: disable=invalid-name
from Stoner import Data

d = Data("sample.txt", setas="xy")
d.plot(fmt="b-")
d.select(Temp__gt=75).select(Res__between=(5.3, 6.3)).plot(
    fmt="ro", label="portion 1"
)
d.select(Temp__lt=30).plot(fmt="g<", label="portion 2")
