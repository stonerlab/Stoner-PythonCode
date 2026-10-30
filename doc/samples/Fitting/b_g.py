"""Demonstrate Bloch-Grueneisen resistivity fitting with a registered shortcut."""

# pylint: disable=invalid-name
from copy import deepcopy

from numpy import linspace, ones_like
from numpy.random import normal

from Stoner import Data
from Stoner.analysis.fitting.models.e_transport import (
    BlochGrueneisen,
    blochGrueneisen,
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

d.fit.curve_fit(blochGrueneisen, p0=params, result=True, header="curve_fit")

d.setas = "xy"
d.fit.bloch_grueneisen(result=True, header="lmfit")

d.setas = "xyeyy"
d.plot(fmt=["r.", "b-", "g-"])

d.plot.annotate_fit(blochGrueneisen, x=0.65, y=0.35, fontdict={"size": "x-small"})
d.plot.annotate_fit(
    BlochGrueneisen,
    x=0.65,
    y=0.05,
    fontdict={"size": "x-small"},
    prefix="BlochGrueneisen",
)
d.plot.title = "Bloch-Grueneisen Fit"
