# -*- coding: utf-8 -*-
"""
Test the Image Widgets used for selections

@author: phygbu
"""
import pytest
from matplotlib.backend_bases import Event
import Stoner
import os
import threading
import time
import numpy as np


def _trigger(image):
    time.sleep(2)
    select=image._image._select
    event=Event("fake",select.fig.canvas)
    event.xdata=1
    event.ydata=1
    select.on_click(event)
    for coord in np.linspace(1,100,51):
        event.xdata=coord
        event.ydata=coord
        select.draw_line(event)
    select.on_click(event)

def _trigger2(image):
    time.sleep(2)
    select=image._image._select
    event=Event("fake",select.fig.canvas)
    event.xdata=50
    event.ydata=75
    event.key="x"
    select.keypress(event)
    select.on_click(event)

def _trigger3(image):
    time.sleep(2)
    select=image._image._select
    event=Event("fake",select.fig.canvas)
    event.xdata=50
    event.ydata=75
    event.key="y"
    select.keypress(event)
    select.on_click(event)

def _trigger4(image):
    time.sleep(2)
    select=image._image._select
    event1=Event("fake",select.fig.canvas)
    event1.xdata=25
    event1.ydata=25
    event2=Event("fake",select.fig.canvas)
    event2.xdata=75
    event2.ydata=50
    select.on_select(event1,event2)
    event=Event("fake",select.fig.canvas)
    event.key="enter"
    select.finish(event)


def test_profile_line():
    os.chdir(Stoner.__homepath__/".."/"sample-data")
    img=Stoner.HDF5.STXMImage("Sample_Image_2017-10-15_100.hdf5")
    thread=threading.Thread(target=_trigger,args=(img,))
    thread.start()
    result = img.profile_line()
    assert len(result)==142
    assert result.x.min()==0.0
    assert np.isclose(result.x.max(),140,atol=0.01)
    assert np.isclose(result.y.mean(),26407.86,atol=0.01)
    thread=threading.Thread(target=_trigger2,args=(img,))
    thread.start()
    result = img.profile_line()
    assert len(result)==101
    assert result.x.min()==0.0
    assert result.x.max()==100.0
    assert np.isclose(result.y.mean(),20022.16,atol=0.01)
    thread=threading.Thread(target=_trigger3,args=(img,))
    thread.start()
    result = img.profile_line()
    assert len(result)==101
    assert result.x.min()==0.0
    assert result.x.max()==100.0
    assert np.isclose(result.y.mean(),27029.16,atol=0.01)

def test_regionSelect():
    os.chdir(Stoner.__homepath__/".."/"sample-data")
    img=Stoner.HDF5.STXMImage("Sample_Image_2017-10-15_100.hdf5")
    thread=threading.Thread(target=_trigger4,args=(img,))
    thread.start()
    result = img.crop()
    assert result.shape==(25,50)


if __name__=="__main__": # Run some tests manually to allow debugging
    pytest.main(["--pdb",__file__])

