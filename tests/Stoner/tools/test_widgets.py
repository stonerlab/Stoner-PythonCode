# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:04:58 2020

@author: phygbu
"""

import sys
import pytest
from matplotlib.backend_bases import Event
import threading
import time
import numpy as np
import Stoner

ret_pth = Stoner.__homepath__ / ".." / "sample-data" / "TDI_Format_RT.txt"

# Horrible hack to patch QFileDialog  for testing

import Stoner.tools.widgets as widgets
from Stoner import Data, DataFolder


def test_filedialog():
    def dummy(mode="getOpenFileName"):
        modes = {
            "getOpenFileName": ret_pth,
            "getOpenFileNames": [ret_pth],
            "getSaveFileName": None,
            "getExistingDirectory": ret_pth.parent,
        }
        return lambda *args, **kargs: (modes[mode], None)

    modes = {
        "OpenFile": {
            "method": dummy("getOpenFileName"),
            "caption": "Select file to open...",
            "arg": ["parent", "caption", "directory", "filter", "options"],
        },
        "OpenFiles": {
            "method": dummy("getOpenFileNames"),
            "caption": "Select file(s_ to open...",
            "arg": ["parent", "caption", "directory", "filter", "options"],
        },
        "SaveFile": {
            "method": dummy("getSaveFileName"),
            "caption": "Save file as...",
            "arg": ["parent", "caption", "directory", "filter", "options"],
        },
        "SelectDirectory": {
            "method": dummy("getExistingDirectory"),
            "caption": "Select folder...",
            "arg": ["parent", "caption", "directory", "options"],
        },
    }

    widgets = sys.modules["Stoner.tools.widgets"]
    app = getattr(widgets, "App")
    setattr(app, "modes", modes)

    assert widgets.fileDialog.openDialog() == ret_pth
    assert widgets.fileDialog.openDialog(title="Test", start=".") == ret_pth
    assert widgets.fileDialog.openDialog(patterns={"*.bad": "Very bad files"}) == ret_pth
    assert widgets.fileDialog.openDialog(mode="OpenFiles") == [ret_pth]
    assert widgets.fileDialog.openDialog(mode="SaveFile") is None
    assert widgets.fileDialog.openDialog(mode="SelectDirectory") == ret_pth.parent
    with pytest.raises(ValueError):
        widgets.fileDialog.openDialog(mode="Whateve")


def test_loader():
    d = Data(False)
    assert d.shape == (1676, 3), "Failed to load data with dialog box"
    with pytest.raises(RuntimeError):
        d.save(False)
    fldr = DataFolder(False)
    del fldr["bad_data"]
    assert fldr.shape == (
        52,
        {
            "attocube_scan": (15, {}),
            "maximus_scan": (2, {"MPI_210127021": (3, {})}),
            "NLIV": (11, {}),
            "recursivefoldertest": (1, {}),
            "working": (4, {}),
        },
    )
    fldr = DataFolder(False, multifile=True)
    assert fldr.shape == (1, {}), "multifile mode failed!"


def _event(data, name, **kargs):
    """Make a fake event."""
    select = data._select
    event = Event("fake", select.data.fig.canvas)
    for k, v in kargs.items():
        setattr(event, k, v)
    try:
        getattr(select, name)(event)
    except Exception:
        breakpoint()
        pass


def _trigger0(data):
    time.sleep(1)
    select = data._select
    select.onselect(50, 100)
    _event(data, "keypress", key="escape")


def _trigger1(data):
    time.sleep(1)
    select = data._select
    select.onselect(50, 100)
    _event(data, "keypress", key="enter")


def _trigger2(data):
    time.sleep(1)
    select = data._select
    select.onselect(50, 100)
    select.onselect(150, 200)
    _event(data, "keypress", key="i")
    _event(data, "keypress", key="backspace")
    _event(data, "keypress", key="enter")


def test_range_select():
    data = Stoner.Data(ret_pth, setas="xy")
    thread = threading.Thread(target=_trigger0, args=(data,))
    thread.start()
    result = data.search()
    xmin, xmax = result.x.min(), result.x.max()
    assert xmin < 4.4 and xmax > 291, "Failed to select and clear"
    thread = threading.Thread(target=_trigger1, args=(data,))
    thread.start()
    result = data.search()
    xmin1, xmax1 = result.x.min(), result.x.max()
    assert np.isclose(xmin1, 50, atol=1) and np.isclose(xmax1, 100, 1), "Single selection failed."
    thread = threading.Thread(target=_trigger2, args=(data,))
    thread.start()
    result = data.search()
    xmin2, xmax2 = result.x.min(), result.x.max()
    assert np.isclose(xmin, xmin2) and np.isclose(xmax, xmax2), "Selection with keypresses failed"


if __name__ == "__main__":
    pytest.main(["--pdb", __file__])
