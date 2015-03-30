from Stoner import Data
from numpy import exp,sin,linspace

x=linspace(0.0,5.0,500)
x2=linspace(4.0,9.0,500)

y=exp(-x)+0.1*sin(x*10)

y2=(exp(-x2)+0.1*sin(x2*10))*3.43-0.0081
x2=x2+3.21E-3

s1=Data()
s1=s1&x&y
s1.column_headers=["Angle","Signal"]
s1.setas="xy"
s1.plot(label="Set 1")

s2=Data()
s2=s2&x2&y2
s2.column_headers=["Angle","Signal"]
s2.setas="xy"
s2.fig=s1.fig
s2.plot(label="Set 2")

s2.stitch(s1)
s2.plot(label="Stictched")