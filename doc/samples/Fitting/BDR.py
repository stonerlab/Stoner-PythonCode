"""Example of nDimArrhenius Fit."""
from Stoner import Data
import Stoner.Fit as SF
from  numpy import linspace,ones_like
from numpy.random import normal

#Make some data
V=linspace(-4,4,1000)
I=SF.bdr(V,2500,5.2,0.3,15.0,1.0)+normal(size=len(V),scale=1.0)
dI=ones_like(V)*1.0

#Curve fit
d=Data(V,I,dI,setas="xye",column_headers=["Bias","Current","Noise"])

d.curve_fit(SF.bdr,p0=[2500,5.2,0.3,15.0,1.0],result=True,header="curve_fit")
d.setas="xyey"
d.plot(fmt=["r.","b-"])
d.annotate_fit(SF.bdr,x=-3.95,y=5E-5,prefix="bdr",fontdict={"size":"x-small"})

#lmfit
d.setas="xy"
fit=SF.BDR(missing="drop")
p0=fit.guess(I,x=V)
for p,v,mi,mx in zip(["A","phi","dphi","d","mass"],[2500,5.2,0.3,15.0,1.0],[100,1,0.01,5,0.5],[1E4,20.0,2.0,30.0,5.0]):
    p0[p].value=v
    p0[p].bounds=[mi,mx]
d.lmfit(SF.BDR,p0=p0,result=True,header="lmfit")
d.setas="x...y"
d.plot()
d.annotate_fit(fit,x=0,y=-500,prefix="BDR",fontdict={"size":"x-small"})

d.ylabel="Current"
d.title="BDR Model test"
d.tight_layout()