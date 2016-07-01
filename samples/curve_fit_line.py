from Stoner import Data
import numpy as np
from numpy.random import normal

def linear(x,m,c):
    return m*x+c

d=Data("curve_fit_data.dat",setas="xye")
d.plot(fmt="ro")
popt,pcov=d.curve_fit(linear,result=True,replace=False,header="Fit")
d.setas="x..y"
d.plot(fmt="b-")
d.annotate_fit(linear)
d.draw()