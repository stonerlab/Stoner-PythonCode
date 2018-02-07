# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:53:40 2018

@author: phygbu
"""

from Stoner import Data,DataFolder
from Stoner.Fit import linear
from Stoner.Util import format_error

#Adjust these to match expected columns - a unique partial match is sufficient
t_col="T2"
r_col=":R"

data=Data(False)  #Get a dialog box to the file containing Tc data

#Define my working x and y axes
data.setas(x=t_col,y=r_col)

#Normalise data on y axis between +/- 1
data.normalise(base=(-1.,1.),replace=True)

#Swap x and y axes around so that R is x and T is y
data=~data

#Curve fit a straight line, using only the central 90% of the resistance transition
data.curve_fit(linear,bounds=lambda x,r:-0.9<x<0.9,result=True) #result=True to record fit into metadata

# Just print and format nicely
print("Tc={}K".format(format_error(data["linear:intercept"],data["linear:intercept err"])))
