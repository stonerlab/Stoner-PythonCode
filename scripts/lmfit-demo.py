# -*- coding: utf-8 -*-
"""
Demo of new Stoner.Analysis.AnalyseFile.lmfit
Created on Thu Oct 02 21:17:41 2014

@author: phygbu
"""
import os

from Stoner import Data
from Stoner.Fit import Strijkers # Stoner class builtin PCAR fitting code

os.chdir("../sample-data")#edit if not running from scripts folder

d=Data("PCAR TDI.txt",setas="xy") # file with mpfit'ed PCAR data

#Set some parameter hints for good measure
m=Strijkers()
m.set_param_hint("Z",value=0.25,min=0.05)
m.set_param_hint("omega",value=0.40,min=0.36)
m.set_param_hint("P",value=0.5,min=0.0,max=1.0)
m.set_param_hint("delta",value=1.5,min=0.25,max=1.75)

fit=d.lmfit(m,result=True,header="lmfit")
text= "\n".join([d.format("Strijkers:{}".format(k),latex=True,prefix="{} = ".format(k)) for k in m.param_names])

d.plot_xy(0,[2,3,1],["bo","k-","r-"]) # plot the data
d.xlim=(-30,35)
d.text(6.5,1.04,text)

print fit.fit_report()
