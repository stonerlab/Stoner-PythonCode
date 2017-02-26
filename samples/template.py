from Stoner import Data
from Stoner.plot.formats import DefaultPlotStyle
p=Data("sample.txt",setas="xy",template=DefaultPlotStyle())
p.template(axes_color__cycle=["r","g","b"])
p.plot()
