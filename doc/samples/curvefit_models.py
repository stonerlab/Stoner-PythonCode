"""Simple use of lmfit to fit data."""
from Stoner import Data
from Stoner.Fit import PowerLaw
from Stoner.Analysis import _odr_Model
from numpy import linspace,exp,random

#Make some data
x=linspace(0,10.0,101)
y=PowerLaw().func(x,1E-1,1.64)*random.normal(loc=1.0,scale=0.05,size=101)
d=Data(x,y,column_headers=["Time","Signal"],setas="xy")

#Do the fitting and plot the result
fit = d.curve_fit(PowerLaw,result=True,header="LM-Model Fit",residuals=True,output="report")

ODRModel=_odr_Model(PowerLaw,p0=(1,1))
fit = d.curve_fit(ODRModel,result=True,header="ODR-Fit",residuals=True,output="report",prefix="ODRModel")
#Reset labels
d.labels=[]

# Make nice two panel plot layout
ax=d.subplot2grid((3,1),(2,0))
d.setas="x..y"
d.plot(fmt="g+",label="Fit residuals")
d.setas="x....y"
d.plot(fmt="b+",label="ODRModel Residuals")
d.title=""

ax=d.subplot2grid((3,1),(0,0),rowspan=2)
d.setas="xyy.y"
d.plot(fmt=["ro","g-","b-"])
d.xticklabels=[[]]
d.ax_xlabel=""

# Annotate plot with fitting parameters
d.annotate_fit(PowerLaw,x=2.2,y=3,fontdict={"size":"x-small"})
d.annotate_fit(PowerLaw,prefix="ODRModel",x=2.2,y=1.5,fontdict={"size":"x-small"})
d.title=u"curve_fit with models"
