# -*- coding: utf-8 -*-
"""Demo of new Stoner.Analysis.AnalyseFile.lmfit

Created on Thu Oct 02 21:17:41 2014

@author: phygbu
"""
from __future__ import print_function

from Stoner.Fit import cfg_data_from_ini,cfg_model_from_ini

d=cfg_data_from_ini("PCAR-New.ini","../sample-data/PCAR Co Data.csv")

model,p0=cfg_model_from_ini("PCAR-New.ini",data=d)

fit=d.lmfit(model,result=True,header="lmfit")

d.plot_xy(0,[2,1],["r-","bo"]) # plot the data
d.annotate_fit(model,x=0.05,y=0.65,xycoords="axes fraction")

print(fit.fit_report())