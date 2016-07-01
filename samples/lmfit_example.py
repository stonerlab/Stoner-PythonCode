from Stoner import Data
from Stoner.Fit import StretchedExp

#Load dat and plot
d=Data("lmfit_data.txt",setas="xy")

# Do the fit
d.lmfit(StretchedExp,result=True,header="Fit",prefix="")
# plot
d.setas="xyy"

d.plot(fmt=["+","-"])
# Make apretty label using Stoner.Util methods
text="$y=A\\exp\\left[\\left(-\\frac{x}{x_0}\\right)^\\beta\\right]$\n"
text+=d.annotate_fit(StretchedExp,text_only=True)
d.text(6,4E4,text)
#Adjust layout NB pass-through method to pyplot used here
d.tight_layout()