"""Demonstrate the Data.fit accessor."""

# pylint: disable=invalid-name
from os.path import join

from Stoner import __home__
from Stoner.analysis.fitting.models import cfg_data_from_ini, cfg_model_from_ini

config = join(__home__, "..", "scripts", "PCAR-chi^2.ini")
datafile = join(__home__, "..", "sample-data", "PCAR Co Data.csv")

d = cfg_data_from_ini(config, datafile)
model, p0 = cfg_model_from_ini(config, data=d)

fit = d.fit.lmfit(model, p0=p0, result=True, header="Fit", output="data")
with fit.plot.style("stoner"):
    columns = fit.setas._get_cols().ycol
    with fit.plot.StackVertical(len(columns)) as panels:
        for column, ax in zip(columns, panels):
            fit.plot.errorbar(y=column, ax=ax, capsize=3, title=None)
        fit.plot.yscale = "log"  # The most recently drawn panel is chi-squared.
