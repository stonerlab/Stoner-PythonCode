# -*- coding: utf-8 -*-
"""
Demo of new Stoner.Analysis.AnalyseFile.lmfit
Created on Thu Oct 02 21:17:41 2014

@author: phygbu
"""
import os

import Stoner.Analysis as SA
import Stoner.Plot as SP
from Stoner.Fit import Strijkers # Stoner class builtin PCAR fitting code

class working(SA.AnalyseFile,SP.PlotFile):
    """Just so we can fit and plot in one class"""
    pass

os.chdir("../sample-data")#edit if not running from scripts folder

d=working("PCAR TDI.txt") # file with mpfit'ed PCAR data
d.setas="xy"

#Set some parameter hints for good measure
m=Strijkers()
m.set_param_hint("Z",value=0.5,min=0.05)
m.set_param_hint("omega",value=0.40,min=0.36)
m.set_param_hint("P",value=0.42,min=0.0,max=1.0)
m.set_param_hint("delta",value=1.5,min=0.25,max=1.75)

fit=d.lmfit(m,result=True,header="lmfit")

d.plot_xy(0,[2,3,1],["bo","k-","r-"]) # plot the data

print fit.fit_report()
