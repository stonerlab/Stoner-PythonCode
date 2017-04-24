"""Extrapolate data example."""
from Stoner import Data
from numpy import linspace
from numpy.random import normal
import matplotlib.pyplot as plt

x=linspace(1,10,91)
d=Data(x,(2*x**2-x+2)+normal(size=91,scale=2.0),setas="xy",column_headers=["x","y"])

extra_x=linspace(8,12,17)

d.plot()
d.title="Extrapolation Demo"

y3=d.extrapolate(extra_x,overlap=2.0,kind="cubic")
y2=d.extrapolate(extra_x,overlap=2.0,kind="quadratic")
y1=d.extrapolate(extra_x,overlap=10,kind="linear")

plt.plot(extra_x,y1,'bo',label="Linear Extrapolation")
plt.plot(extra_x,y2,'g o',label="Quadratic Extrapolation")
plt.plot(extra_x,y3,'ro',label="Cubic Extrapolation")
plt.legend(loc=2)