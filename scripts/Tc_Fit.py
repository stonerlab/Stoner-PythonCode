# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 16:53:40 2018

@author: phygbu
"""

from Stoner import Data,DataFolder
from Stoner.Util import format_error

t_col=": T2" # Temperature column label
r_col="Nb::R" #Resistance Column Label
iterator="iterator" #Temperature ramp iteration coolumn label
threshold=0.95 #Fraction of transition to fit to

def cubic(x,d,c,a):
    return a*x**3+c*x+d

data=Data(False)  #Get a dialog box to the file containing Tc data

#Define my working x and y axes
data.setas(x=t_col,y=r_col)

#Split one file into a folder of two files by the iterator column
fldr=data.split(iterator)

for data in fldr: #For each iteration ramp in the Tc data

    #Normalise data on y axis between +/- 1
    data.normalise(base=(-1.,1.),replace=True)
    
    #Swap x and y axes around so that R is x and T is y
    data=~data
    
    #Curve fit a straight line, using only the central 90% of the resistance transition
    data.curve_fit(cubic,bounds=lambda x,r:-threshold<x<threshold,result=True) #result=True to record fit into metadata

    #Plot the results
    data.setas[-1]="y"
    data.plot(fmt=["k.","r-"])
    data.annotate_fit(cubic)   
    # Just print and format nicely
