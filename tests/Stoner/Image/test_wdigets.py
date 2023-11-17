# -*- coding: utf-8 -*-
"""
Test the Image Widgets used for selections

@author: phygbu
"""
import os
import threading
import time

import numpy as np
from matplotlib.backend_bases import Event

import pytest
import Stoner
from Stoner.Image.widgets import send_event as _event


def _trigger(image):
    time.sleep(5)
    _event(image, "on_click", xdata=1, ydata=1, button=1)
    for coord in np.linspace(1, 100, 51):
        _event(image, "draw_line", xdata=coord, ydata=coord)
    _event(image, "on_click", xdata=coord, ydata=coord, button=1)


def _trigger2(image):
    time.sleep(5)
    _event(image, "keypress", xdata=50, ydata=75, key="x")
    _event(image, "on_click", xdata=50, ydata=75, button=1)


def _trigger3(image):
    time.sleep(5)
    _event(image, "keypress", xdata=50, ydata=75, key="y")
    _event(image, "on_click", xdata=50, ydata=75, button=1)


def _trigger4(image):
    time.sleep(5)
    select = image._image._select
    event1 = Event("fake", select.fig.canvas)
    event1.xdata = 25
    event1.ydata = 25
    event2 = Event("fake", select.fig.canvas)
    event2.xdata = 75
    event2.ydata = 50
    select.on_select(event1, event2)
    time.sleep(2)
    _event(image, "finish", key="enter")


def _trigger5(image, mode):
    time.sleep(5)
    _event(image, ["draw", "on_click"], xdata=50, ydata=25, button=1, dblclick=False)
    _event(image, ["draw", "on_click"], xdata=75, ydata=50, button=1, dblclick=False)
    _event(image, "keypress", xdata=50, ydata=75, key=mode.lower()[0])
    if mode == "c":  # add some extra points:
        _event(image, ["draw", "on_click"], xdata=30, ydata=40, button=1, dblclick=False)
        _event(image, ["draw", "on_click"], xdata=30, ydata=30, button=1, dblclick=False)
    _event(image, "keypress", xdata=50, ydata=75, key="i")
    time.sleep(2)
    _event(image, "keypress", xdata=50, ydata=75, key="enter")


def _trigger6(image, mode):
    time.sleep(5)
    _event(image, ["draw", "on_click"], xdata=50, ydata=25, button=1, dblclick=False)
    _event(image, ["draw", "on_click"], xdata=75, ydata=50, button=1, dblclick=False)
    time.sleep(2)
    _event(image, "keypress", xdata=50, ydata=75, key="escape")


def test_profile_line():
    os.chdir(Stoner.__homepath__ / ".." / "sample-data")
    img = Stoner.HDF5.STXMImage("Sample_Image_2017-10-15_100.hdf5")
    thread = threading.Thread(target=_trigger, args=(img,))
    thread.start()
    result = img.profile_line()
    assert len(result) == 142
    assert result.x.min() == 0.0
    assert np.isclose(result.x.max(), 140, atol=0.01)
    assert np.isclose(result.y.mean(), 26548.72, atol=0.01)
    thread = threading.Thread(target=_trigger2, args=(img,))
    thread.start()
    result = img.profile_line()
    assert len(result) == 101
    assert result.x.min() == 0.0
    assert result.x.max() == 100.0
    assert np.isclose(result.y.mean(), 20022.16, atol=0.01)
    thread = threading.Thread(target=_trigger3, args=(img,))
    thread.start()
    result = img.profile_line()
    assert len(result) == 101
    assert result.x.min() == 0.0
    assert result.x.max() == 100.0
    assert np.isclose(result.y.mean(), 27029.16, atol=0.01)


def test_crop_with_ui():
    os.chdir(Stoner.__homepath__ / ".." / "sample-data")
    img = Stoner.HDF5.STXMImage("Sample_Image_2017-10-15_100.hdf5")
    thread = threading.Thread(target=_trigger4, args=(img,))
    thread.start()
    result = img.crop()
    assert result.shape == (25, 50)


def test_mask_select():
    os.chdir(Stoner.__homepath__ / ".." / "sample-data")
    img = Stoner.HDF5.STXMImage("Sample_Image_2017-10-15_100.hdf5")
    thread = threading.Thread(target=_trigger5, args=(img, "p"))
    thread.start()
    img.mask.select()
    result = img.mask.sum()
    assert result == 9324, f"Mask selection by polygon failed result={result}"
    img.mask = False
    thread = threading.Thread(target=_trigger5, args=(img, "c"))
    thread.start()
    img.mask.select()
    result = img.mask.sum()
    assert result in (3664, 7489), f"Mask selection by circle failed result={result}"
    img.mask = False
    thread = threading.Thread(target=_trigger5, args=(img, "r"))
    thread.start()
    img.mask.select()
    result = img.mask.sum()
    assert result == 8699, f"Mask selection by reverse failed result={result}"
    img.mask = False
    thread = threading.Thread(target=_trigger6, args=(img, "c"))
    thread.start()
    img.mask.select()
    result = img.mask.sum()
    assert result == 0, f"Cancelling selkection failed with result={result}"


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
