"""Demonstrate the Data.fit accessor."""

# pylint: disable=invalid-name
from os.path import join

from Stoner import __home__
from Stoner.analysis.fitting.models import cfg_data_from_ini, cfg_model_from_ini
from Stoner.analysis.fitting.models.superconductivity import (
    Woods_Diffusive,
    Woods_Ballistic,
)

config = join(__home__, "..", "scripts", "PCAR-New.ini")
datafile = join(__home__, "..", "sample-data", "PCAR Co Data.csv")

d = cfg_data_from_ini(config, datafile)
model, p0 = cfg_model_from_ini(config, data=d)

fit = d.fit.lmfit(model, result=True, header="Strijkers", output="report")

d.plot.line(0, [1, 2], fmt=["bo", "r-"])  # plot the data
d.plot.annotate_fit(model, x=0.05, y=0.75, fontdict={"size": "x-small"})

print(fit.fit_report())

model = Woods_Diffusive()
fit = d.fit.lmfit(model, result=True, header="Woods Diffusive", output="report")

d.plot.line(0, [3], fmt=["m-"])  # plot the data
d.plot.annotate_fit(
    model, x=0.05, y=0.5, fontdict={"size": "x-small"}, prefix="Woods_Diffusive"
)

print(fit.fit_report())

model = Woods_Ballistic()
fit = d.fit.lmfit(model, result=True, header="Woods Ballistic", output="report")

d.plot.line(0, [4], fmt=["g-"])  # plot the data
d.plot.annotate_fit(
    model,
    x=0.05,
    y=0.25,
    fontdict={"size": "x-small"},
    prefix="Woods_Ballistic",
)

print(fit.fit_report())
