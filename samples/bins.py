from Stoner import Data
from Stoner.plotutils import errorfill

d=Data("Noisy_Data.txt",setas="xy")

d.template.fig_height=6
d.template.fig_width=8
d.plot()

e=d.bin(bins=0.05,mode="lin")
f=d.bin(bins=0.25,mode="lin")

e.fig=d.fig
f.fig=d.fig

e.plot(plotter=errorfill,label="0.05 bins")
f.plot(plotter=errorfill,label="0.25 bins")

d.xlim=(1,6)
d.ylim(-0.1,0.4)
d.title="Bin demo"