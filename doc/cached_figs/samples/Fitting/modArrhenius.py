"""Example of nDimArrhenius Fit."""
from Stoner import Data
import Stoner.Fit as SF
from  numpy import linspace
from numpy.random import normal

#Make some data
T=linspace(200,350,101)
R=SF.modArrhenius(T,1E6,0.5,1.5)*normal(scale=0.00005,loc=1.0,size=len(T))
d=Data(T,R,setas="xy",column_headers=["T","Rate"])

#Curve_fit on its own
d.curve_fit(SF.modArrhenius,p0=[1E6,0.5,1.5],result=True,header="curve_fit")
d.setas="xyy"
d.plot()
d.annotate_fit(SF.modArrhenius,x=200,y=250)

# lmfit using lmfit guesses
fit=SF.ModArrhenius()
p0=[1E6,0.5,1.5]
d.lmfit(fit,p0=p0,result=True,header="lmfit")
d.setas="x..y"
d.plot()
d.annotate_fit(SF.ModArrhenius,x=200,y=150,prefix="ModArrhenius")

d.title="Modified Arrhenius Test Fit"
d.ylabel="Rate"
d.xlabel="Temperature (K)"