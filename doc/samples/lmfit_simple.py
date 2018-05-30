"""Simple use of lmfit to fit data."""
from Stoner import Data
from numpy import linspace,exp,random

#Make some data
x=linspace(0,10.0,101)
y=2+4*exp(-x/1.7)+random.normal(scale=0.2,size=101)

d=Data(x,y,column_headers=["Time","Signal"],setas="xy")

#Do the fitting and plot the result
func=lambda x,A,B,C:A+B*exp(-x/C)
fit = d.lmfit(func,result=True,header="Fit",A=1,B=1,C=1,residuals=True)

# Make nice label of the parameters
ax=d.subplot2grid((3,1),(2,0))
d.setas="x..y"
d.plot(fmt="g+")
d.title=""

ax=d.subplot2grid((3,1),(0,0),rowspan=2)
d.setas="xyy"
d.labels=[]
d.plot(fmt=["ro","b-"])
d.xticklabels=[[]]
d.xlabel=""
d.annotate_fit(func,prefix="Model",x=5,y=3.75,fontdict={"size":"x-small"})

text=r"$y=A+Be^{-x/C}$"+"\n\n"
d.text(5,4,text,fontdict={"size":"x-small"})
d.title=u"Levenberg-Marquardt Fit"
