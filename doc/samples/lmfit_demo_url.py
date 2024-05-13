"""Demo of new Stoner.Analysis.AnalyseFile.lmfit."""

# pylint: disable=invalid-name
import urllib
import io

from Stoner.analysis.fitting.models import cfg_data_from_ini, cfg_model_from_ini

config = io.StringIO(
    urllib.request.urlopen(  # pylint: disable=urllib_urlopen
        "https://raw.githubusercontent.com/stonerlab/Stoner-PythonCode/master/scripts/PCAR-chi%5E2.ini"
    )
    .read()
    .decode()
)
datafile = io.StringIO(  # pylint: disable=urllib_urlopen
    urllib.request.urlopen(
        "https://github.com/stonerlab/Stoner-PythonCode/raw/master/sample-data/PCAR%20Co%20Data.csv"
    )
    .read()
    .decode()
)

d = cfg_data_from_ini(config, datafile)
model, p0 = cfg_model_from_ini(config, data=d)

fit = d.lmfit(model, result=True, header="lmfit", output="report")

d.plot_xy(0, [2, 1], ["r-", "bo"], title="PCAR Co Data")  # plot the data
d.annotate_fit(model, x=0.05, y=0.75, fontdict={"size": "x-small"})

print(fit.fit_report())
