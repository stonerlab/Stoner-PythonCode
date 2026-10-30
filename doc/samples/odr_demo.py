"""Demonstrate the Data.fit accessor."""

# pylint: disable=invalid-name
from os.path import join

from Stoner import __home__
from Stoner.analysis.fitting.models import cfg_data_from_ini, cfg_model_from_ini

config = join(__home__, "..", "scripts", "PCAR-New.ini")
datafile = join(__home__, "..", "sample-data", "PCAR Co Data.csv")

d = cfg_data_from_ini(config, datafile)
model, p0 = cfg_model_from_ini(config, data=d)

d.x = d.x + 0.25
d.setas = "xy"
d.plot(fmt="r.")  # plot the data
fit = d.fit.lmfit(model, result=True, header="lmfit")
fit = d.fit.odr(model, result=True, header="odr", prefix="odr")
fit = d.fit.differential_evolution(model, result=True, header="odr", prefix="diff")
d.setas = "x.yyy"
d.plot(fmt=["b-", "g-", "m-"], label=["lmfit", "odr", "diff.ev."])

d.plot.annotate_fit(
    model, x=0.05, y=0.75, fontdict={"size": "x-small", "color": "blue"}
)
d.plot.annotate_fit(
    model,
    x=0.05,
    y=0.5,
    fontdict={"size": "x-small", "color": "green"},
    prefix="odr",
)
d.plot.annotate_fit(
    model,
    x=0.05,
    y=0.25,
    fontdict={"size": "x-small", "color": "magenta"},
    prefix="diff",
)
