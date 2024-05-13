"""Test Weak-localisation fitting."""

# pylint: disable=invalid-name
from copy import deepcopy

from numpy import linspace, ones_like
from numpy.random import normal

from Stoner import Data
from Stoner.analysis.fitting.models.e_transport import (
    blochGrueneisen,
    BlochGrueneisen,
)

T = linspace(4.2, 300, 101)
params = [265, 65, 1.0, 5]
params2 = deepcopy(params)
G = blochGrueneisen(T, *params) + normal(size=len(T), scale=5e-5)
dG = ones_like(T) * 5e-5
d = Data(
    T,
    G,
    dG,
    setas="xye",
    column_headers=["Temperature (K)", "Resistivity", "dR"],
)

d.curve_fit(blochGrueneisen, p0=params, result=True, header="curve_fit")

d.setas = "xy"
d.lmfit(BlochGrueneisen, result=True, header="lmfit")

d.setas = "xyeyy"
d.plot(fmt=["r.", "b-", "g-"])

d.annotate_fit(blochGrueneisen, x=0.65, y=0.35, fontdict={"size": "x-small"})
d.annotate_fit(
    BlochGrueneisen,
    x=0.65,
    y=0.05,
    fontdict={"size": "x-small"},
    prefix="BlochGrueneisen",
)
d.title = "Bloch-Grueneisen Fit"
