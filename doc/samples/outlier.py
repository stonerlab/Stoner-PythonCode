"""Detect outlying points from a lione."""
from Stoner import Data
import numpy as np

x=np.linspace(0,100,201)
y=0.01*x**2+5*np.sin(x/10.0)

i=np.random.randint(len(x)-20,size=20)+10
y[i]+=np.random.normal(size=len(i),scale=20)

d=Data(np.column_stack((x,y)),column_headers=["x","y"],setas="xy")
d.plot(fmt="b.",label="raw data")
e=d.clone
e.outlier_detection(window=5,action="delete")
e.plot(fmt="r-",label="Default Outliers removed")
f=d.clone
f.outlier_detection(window=21,order=2,certainty=2,width=3,action="delete",func=f._poly_outlier)
f.plot(fmt="g-",label="Poly Outliers removed")
e.title="Outlier detection test"