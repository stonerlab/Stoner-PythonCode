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


from Stoner import Data


def test_add():
    x=np.linspace(1,10,10)
    y=2*x-3
    dy=np.abs(y/100)
    z=x+4
    dz=np.abs(z/100)
    data=Data(np.column_stack((x,y,dy,z,dz)),column_headers=["Tine","Signal 1","d1","Signal 2","d2"])
    data.add(1,3,header="Add")
    assert np.all(data//"Add" == (data//"Signal 1" + data//"Signal 2")),"Failed to add column correctly"
    data.add((1,2),(3,4),header="Add",index="Add",replace=True)
    d_man=np.sqrt((data//"d1")**2+(data//"d2")**2)
    assert np.allclose(data//-1,d_man),"Failed to calculate error in add"
    data.add(1,3.0,index="Add",replace=True)
    assert np.all(data//5==data//1+3),"Failed to add with a constant"
    data.add(1,np.ones(10),index=5,replace=True)
    assert np.all(data//5==data//1+1),"Failed to add with a array"


def test_subtract():
    x=np.linspace(1,10,10)
    y=2*x-3
    dy=np.abs(y/100)
    z=x+4
    dz=np.abs(z/100)
    data=Data(np.column_stack((x,y,dy,z,dz)),column_headers=["Tine","Signal 1","d1","Signal 2","d2"])
    data.subtract(1,3,header="Subtract")
    assert np.all(data//"Subtract" == (data//"Signal 1" - data//"Signal 2")),"Failed to add column correctly"
    data.subtract((1,2),(3,4),header="Subtract",index="Subtract",replace=True)
    d_man=np.sqrt((data//"d1")**2+(data//"d2")**2)
    assert np.allclose(data//-1,d_man),"Failed to calculate error in add"
    data.subtract(1,3.0,index="Subtract",replace=True)
    assert np.all(data//5==data//1-3),"Failed to subtract with a constant"
    data.subtract(1,np.ones(10),index=5,replace=True)
    assert np.all(data//5==data//1-1),"Failed to subtract with a array"

def test_multiply():
    x=np.linspace(1,10,10)
    y=2*x-3
    dy=np.abs(y/100)
    z=x+4
    dz=np.abs(z/100)
    data=Data(np.column_stack((x,y,dy,z,dz)),column_headers=["Tine","Signal 1","d1","Signal 2","d2"])
    data.multiply(1,3,header="Multiply")
    assert np.all((data//"Multiply") == ((data//"Signal 1") * (data//"Signal 2"))),"Failed to add column correctly"
    data.multiply((1,2),(3,4),header="Multiply",index="Multiply",replace=True)
    d_man=np.sqrt(2)*0.01*np.abs(data//-2)
    assert np.allclose(data//-1,d_man),"Failed to calculate error in add"
    data.multiply(1,3.0,index="Multiply",replace=True)
    assert np.all(data//5==(data//1)*3),"Failed to multiply with a constant"
    data.multiply(1,2*np.ones(10),index=5,replace=True)
    assert np.all(data//5==(data//1)*2),"Failed to multiply with a array"

def test_divide():
    x=np.linspace(1,10,10)
    y=2*x-3
    dy=np.abs(y/100)
    z=x+4
    dz=np.abs(z/100)
    data=Data(np.column_stack((x,y,dy,z,dz)),column_headers=["Tine","Signal 1","d1","Signal 2","d2"])
    data.divide(1,3,header="Divide")
    assert np.all((data//"Divide") == ((data//"Signal 1") / (data//"Signal 2"))),"Failed to add column correctly"
    data.divide((1,2),(3,4),header="Divide",index="Divide",replace=True)
    d_man=np.sqrt(2)*0.01*np.abs(data//-2)
    assert np.allclose(data//-1,d_man),"Failed to calculate error in diffsum"
    data.divide(1,3.0,index="Divide",replace=True)
    assert np.all(data//5==(data//1)/3),"Failed to add with a constant"
    data.divide(1,2*np.ones(10),index=5,replace=True)
    assert np.all(data//5==(data//1)/2),"Failed to add with a array"

def test_diffsum():
    x=np.linspace(1,10,10)
    y=2*x-3
    dy=np.abs(y/100)
    z=x+4
    dz=np.abs(z/100)
    data=Data(np.column_stack((x,y,dy,z,dz)),column_headers=["Tine","Signal 1","d1","Signal 2","d2"])
    data.diffsum(1,3,header="Diffsum")
    a=data//1
    b=data//3
    man=(a-b)/(a+b)
    assert np.all((data//"Diffsum") == man),"Failed to diffsum column correctly"
    data.diffsum((1,2),(3,4),header="Diffsum",index="Diffsum",replace=True)

def test_limits():
    x=np.linspace(1,10,10)
    y=2*x-3
    dy=np.abs(y/100)
    z=x+4
    dz=np.abs(z/100)
    data=Data(np.column_stack((x,y,dy,z,dz)),column_headers=["Tine","Signal 1","d1","Signal 2","d2"])
    data.setas="x..ye"
    assert data.min(1)==(-1,0),"Minimum method failed"
    assert data.min()==(5.0,0),"Minimum method failed"
    assert data.min(1,bounds=lambda r:r[0]>2)==(3.0,2.0),"Max with bounds failed"
    assert data.max()==(14,9),"Max method failed"
    assert data.max(1)==(17,9),"Max method failed"
    assert data.max(1,bounds=lambda r:r[0]<5)==(5.0,3.0),"Max with bounds failed"
    assert data.span()==(5.0, 14.0),"span method failed"
    assert data.span(1)==(-1.0, 17.0),"span method failed"
    assert data.span(1,bounds=lambda r:2<r.i<8)==(5.0,13.0),"span with bounds failed"

def test_stats():
    x=np.linspace(1,10,10)
    y=2*x-3
    dy=np.abs(y/100)
    z=x+4
    dz=np.abs(z/100)
    data=Data(np.column_stack((x,y,dy,z,dz)),column_headers=["Tine","Signal 1","d1","Signal 2","d2"])
    assert data.mean(1)==8.0,"Simple channel mean failed"
    assert np.allclose(data.mean(1,sigma=2),(0.048990998729652346, 0.031144823004794875)),"Channel mean with sigma failed"
    assert np.isclose(data.std(1), 6.0553007081949835),"Simple Standard Deviation failed"
    assert np.isclose(data.std(1,2), 2.7067331877422456),"Simple Standard Deviation failed"


if __name__=="__main__": # Run some tests manually to allow debugging
    pytest.main(["--pdb",__file__])

