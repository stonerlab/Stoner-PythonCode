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

fit=d.odr(model,result=True,header="odr")

d.plot_xy(0,[2,1],["r-","bo"]) # plot the data
d.annotate_fit(model,x=-25,y=1.02,fontdict={"size":"x-small"})

fit.pprint()