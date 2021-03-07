#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test fitting mixin classes. Much of the code is also tested by the doc/samples scripts - so this is just
filling in some gaps.

Created on Sat Aug 24 22:56:59 2019

@author: phygbu
"""
import pytest
import sys
import os.path as path
import numpy as np


pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../"))
sys.path.insert(0,pth)
from Stoner import Data

def fit(x,a,b,c):
    """Fitting function"""
    return a*x**2+b*x+c


datadir=path.join(pth,"sample-data")


def test_cuve_fit():
    x_data=np.linspace(-10,10,101)
    y_data=0.01*x_data**2+0.3*x_data-2

    y_data*=np.random.normal(size=101,loc=1.0,scale=0.01)
    x_data+=np.random.normal(size=101,scale=0.02)

    sdata=Data(x_data,y_data,column_headers=["X","Y"])
    sdata.setas="xy"
    for output,fmt in zip(["fit","row","full","dict","data"],[tuple,np.ndarray,tuple,dict,Data]):
        res=sdata.curve_fit(fit,p0=[0.02,0.2,2],output=output)
        assert isinstance(res,fmt),"Failed to get expected output from curve_fit for {} (got {})".format(output,type(res))

def test_lmfit():
    x_data=np.linspace(-10,10,101)
    y_data=0.01*x_data**2+0.3*x_data-2

    y_data*=np.random.normal(size=101,loc=1.0,scale=0.01)
    x_data+=np.random.normal(size=101,scale=0.02)

    sdata=Data(x_data,y_data,column_headers=["X","Y"])
    sdata.setas="xy"
    for output,fmt in zip(["fit","row","full","dict","data"],[tuple,np.ndarray,tuple,dict,Data]):
        res=sdata.lmfit(fit,p0=[0.02,0.2,2],output=output)
        assert isinstance(res,fmt),"Failed to get expected output from lmfit for {} (got {})".format(output,type(res))

def test_odr():
    x_data=np.linspace(-10,10,101)
    y_data=0.01*x_data**2+0.3*x_data-2

    y_data*=np.random.normal(size=101,loc=1.0,scale=0.01)
    x_data+=np.random.normal(size=101,scale=0.02)

    sdata=Data(x_data,y_data,column_headers=["X","Y"])
    sdata.setas="xy"
    for output,fmt in zip(["fit","row","full","dict","data"],[tuple,np.ndarray,tuple,dict,Data]):
        res=sdata.odr(fit,p0=[0.02,0.2,2],output=output)
        assert isinstance(res,fmt),"Failed to get expected output from idr for {} (got {})".format(output,type(res))

def test_differential_evolution():
    x_data=np.linspace(-10,10,101)
    y_data=0.01*x_data**2+0.3*x_data-2

    y_data*=np.random.normal(size=101,loc=1.0,scale=0.01)
    x_data+=np.random.normal(size=101,scale=0.02)

    sdata=Data(x_data,y_data,column_headers=["X","Y"])
    sdata.setas="xy"
    for output,fmt in zip(["fit","row","full","dict","data"],[tuple,np.ndarray,tuple,dict,Data]):
        res=sdata.differential_evolution(fit,p0=[0.02,0.2,2],output=output)
        assert isinstance(res,fmt),"Failed to get expected output from differential_evolution for {} (got {})".format(output,type(res))

if __name__=="__main__": # Run some tests manually to allow debugging
    pytest.main(["--pdb",__file__])


