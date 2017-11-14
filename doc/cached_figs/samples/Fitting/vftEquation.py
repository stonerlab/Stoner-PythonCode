"""Example of Arrhenius Fit."""
from Stoner import Data
import Stoner.Fit as SF
from  numpy import logspace,log10
from numpy.random import normal

#Make some data
T=logspace(log10(200),log10(350),51)
params=(1E6,0.5,150)
noise=0.5
R=SF.vftEquation(T,*params)*normal(size=len(T),scale=noise,loc=1.0)
dR=SF.vftEquation(T,*params)*noise

d=Data(T,R,dR,setas="xye",column_headers=["T","Rate"])

#Curve_fit on its own
d.curve_fit(SF.vftEquation,p0=params,result=True,header="curve_fit")

# lmfit using lmfit guesses
fit=SF.VFTEquation()
p0=params
d.lmfit(fit,p0=p0,result=True,header="lmfit")

d.setas="xyeyyy"
d.plot(fmt=["k+","r-","b-"])
d.yscale="log"
d.ylim=(1E-43,1)
d.annotate_fit(SF.vftEquation,x=270,y=1E-27,fontdict={"size":"x-small"},mode="eng")

d.annotate_fit(SF.VFTEquation,x=240,y=1E-40,prefix="VFTEquation",fontdict={"size":"x-small"},mode="eng")

d.title="VFT Equation Test Fit"
d.ylabel="Rate"
d.xlabel="Temperature (K)"