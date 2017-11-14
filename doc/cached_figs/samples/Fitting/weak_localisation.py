"""Test Weak-localisation fitting."""
from Stoner import Data
import Stoner.Fit as SF
from numpy import linspace,ones_like
from numpy.random import normal
from copy import copy

B=linspace(-8,8,201)
params=[1E-3,2.0,0.25,1.4]
G=SF.wlfit(B,*params)+normal(size=len(B),scale=5E-7)
dG=ones_like(B)*5E-7
d=Data(B,G,dG,setas="xye",column_headers=["Field $\\mu_0H (T)$","Conductance","dConductance"])

d.curve_fit(SF.wlfit,p0=copy(params),result=True,header="curve_fit")

d.setas="xye"
d.lmfit(SF.WLfit,p0=copy(params),result=True,header="lmfit")

d.setas="xyeyy"
d.plot(fmt=["r.","b-","g-"])

d.annotate_fit(SF.wlfit,x=-8,y=9.9E-4,fontdict={"size":"x-small"})
d.annotate_fit(SF.WLfit,x=-4,y=9.7E-4,fontdict={"size":"x-small"},prefix="WLfit")
d.title="Weak Localisation Fit"
d.tight_layout()