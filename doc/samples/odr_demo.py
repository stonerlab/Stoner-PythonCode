"""Demo of new Stoner.Analysis.AnalyseFile.lmfit
"""
from __future__ import print_function
from Stoner import __home__
from os.path import join
from Stoner.Fit import cfg_data_from_ini,cfg_model_from_ini

config=join(__home__,"..","scripts","PCAR-New.ini")
datafile=join(__home__,"..","sample-data","PCAR Co Data.csv")

d=cfg_data_from_ini(config,datafile)
model,p0=cfg_model_from_ini(config,data=d)

d.x+=0.25

fit=d.odr(model,result=True,header="odr",residuals=True)

d.plot_xy(0,[2,1],["r-","bo"]) # plot the data
d.annotate_fit(model,x=0.7,y=0.5,fontdict={"size":"x-small"})

fit=d.lmfit(model,result=True,header="lmfit",residuals=True)
d.plot_xy(0, "lmfit","g-")
d.annotate_fit(model,x=0.1,y=0.5,fontdict={"size":"x-small"})
