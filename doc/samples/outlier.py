from Stoner import Data
import numpy as np

x=np.linspace(0,100,201)
y=0.01*x**2+5*np.sin(x/10.0)

i=np.random.randint(len(x),size=10)
y[i]+=np.random.normal(size=len(i),scale=20)

d=Data(np.column_stack((x,y)),column_headers=["x","y"],setas="xy")
d.plot(fmt="b.",label="raw data")
d.outlier_detection(window=5,action="delete")
d.plot(fmt="r-",label="Outliers removed")
d.title="Outlier detection test"
