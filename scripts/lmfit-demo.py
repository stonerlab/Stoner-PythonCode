# -*- coding: utf-8 -*-
"""
Demo of new Stoner.Analysis.AnalyseFile.lmfit
Created on Thu Oct 02 21:17:41 2014

@author: phygbu
"""

from Stoner.Fit import cfg_data_from_ini,cfg_model_from_ini

d=cfg_data_from_ini("PCAR-New.ini","../sample-data/PCAR Co Data.csv")

model,p0=cfg_model_from_ini("PCAR-New.ini",data=d)

fit=d.lmfit(model,result=True,header="lmfit")
text= "\n".join([d.format("Strijkers:{}".format(k),latex=True) for k in model.param_names])

d.plot_xy(0,[1,2],["r-","bo"]) # plot the data
d.xlim=(-30,35)
d.text(6.5,1.04,text)

print fit.fit_report()
