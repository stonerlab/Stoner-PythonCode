"""Demo of new Stoner.Analysis.AnalyseFile.lmfit."""

# pylint: disable=invalid-name
from os.path import join

from Stoner.analysis.fitting.models import cfg_data_from_ini, cfg_model_from_ini
from Stoner import __home__

config = join(__home__, "..", "scripts", "PCAR-New.ini")
datafile = join(__home__, "..", "sample-data", "PCAR Co Data.csv")

d = cfg_data_from_ini(config, datafile)
model, p0 = cfg_model_from_ini(config, data=d)

fit = d.lmfit(model, result=True, header="lmfit", output="report")

d.plot_xy(0, [2, 1], ["r-", "bo"])  # plot the data
d.annotate_fit(model, x=0.05, y=0.75, fontdict={"size": "x-small"})

print(fit.fit_report())
