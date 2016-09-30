from Stoner import Data
from numpy import linspace,exp,random

#Make some data
x=linspace(0,10.0,101)
y=2+4*exp(-x/1.7)+random.normal(scale=0.2,size=101)

d=Data(x,y,column_headers=["Time","Signal"],setas="xy")

d.plot(fmt="ro") # plot our data

#Do the fitting and plot the result
fit = d.lmfit(lambda x,A,B,C:A+B*exp(-x/C),result=True,header="Fit",A=1,B=1,C=1)
d.setas="x.y"
d.labels=[]
d.plot(fmt="b-")

# Make nice label of the parameters
text=r"$y=A+Be^{-x/C}$"+"\n\n"
text+="\n".join([d.format(k,latex=True) for k in ["Model:A","Model:B","Model:C"]])
d.text(5,4,text)


