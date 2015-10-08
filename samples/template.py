from Stoner import Data
from Stoner.PlotFormats import DefaultPlotStyle
p=Data("sample.txt",setas="xy",template=DefaultPlotStyle())
p.template(axes_color__cycle=["r","g","b"])
p.plot()
