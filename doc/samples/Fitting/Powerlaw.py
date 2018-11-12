"""Example of PowerLaw Fit."""
from Stoner import Data
import Stoner.Fit as SF
from  numpy import linspace
from numpy.random import normal

#Make some data
T=linspace(50,500,101)
R=SF.powerLaw(T,1E-2,0.6666666)*normal(size=len(T),scale=0.1,loc=1.0)
d=Data(T,R,setas="xy",column_headers=["T","Rate"])

#Curve_fit on its own
d.curve_fit(SF.powerLaw,p0=[1,0.5],result=True,header="curve_fit")
d.setas="xyy"
d.plot()
d.annotate_fit(SF.powerLaw,x=50,y=4.5E-1)

# lmfit using lmfit guesses
fit=SF.PowerLaw()
p0=fit.guess(R,x=T)
d.lmfit(fit,p0=p0,result=True,header="lmfit")
d.setas="x..y"
d.plot()
d.annotate_fit(SF.PowerLaw,x=150,y=1.5E-1,prefix="PowerLaw")

d.title="Powerlaw Test Fit"
d.ylabel="Rate"
d.xlabel="Temperature (K)"