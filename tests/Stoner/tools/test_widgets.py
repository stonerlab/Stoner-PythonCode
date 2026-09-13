# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:04:58 2020

@author: phygbu
"""

import sys
import threading
import time

import numpy as np
import pytest
from matplotlib.backend_bases import Event

import Stoner

ret_pth = Stoner.__homepath__ / ".." / "sample-data" / "TDI_Format_RT.txt"

# Horrible hack to patch QFileDialog  for testing

import Stoner.tools.widgets as widgets
from Stoner import Data, DataFolder


@pytest.fixture
def mocked_dialog(monkeypatch, no_interactive_file_dialogs):
    """Install predictable dialog responses for one test."""
    def dummy(mode="getOpenFileName"):
        modes = {
            "getOpenFileName": ret_pth,
            "getOpenFileNames": [ret_pth],
            "getSaveFileName": None,
            "getExistingDirectory": ret_pth.parent,
        }
        return lambda *args, **kwargs: (modes[mode], None)

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

    monkeypatch.setattr(widgets.App, "modes", modes)


def test_unexpected_dialog_is_rejected():
    """Unexpected interaction must fail before reaching a native modal dialog."""
    if widgets.QT_VERSION is None:
        pytest.skip("Qt is unavailable; native file dialogs cannot open")
    with pytest.raises(pytest.fail.Exception, match="Unexpected file dialog"):
        widgets.file_dialog.open_dialog()


def test_filedialog(mocked_dialog):
    assert widgets.file_dialog.open_dialog() == ret_pth
    assert widgets.file_dialog.open_dialog(title="Test", start=".") == ret_pth
    assert widgets.file_dialog.open_dialog(patterns={"*.bad": "Very bad files"}) == ret_pth
    assert widgets.file_dialog.open_dialog(mode="OpenFiles") == [ret_pth]
    assert widgets.file_dialog.open_dialog(mode="SaveFile") is None
    assert widgets.file_dialog.open_dialog(mode="SelectDirectory") == ret_pth.parent
    with pytest.raises(ValueError):
        widgets.file_dialog.open_dialog(mode="Whateve")


def test_loader(mocked_dialog):
    d = Data(False)
    assert d.shape == (1676, 3), "Failed to load data with dialog box"
    with pytest.raises(RuntimeError):
        d.save(False)
    fldr = DataFolder(False)
    del fldr["bad_data"]
    assert fldr.shape == (
        55,
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


def _event(data, name, **kwargs):
    """Make a fake event."""
    select = data._select
    event = Event("fake", select.data.fig.canvas)
    for k, v in kwargs.items():
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
    xmin, xmax = result[:, data.setas.cols.xcol].min(), result[:, data.setas.cols.xcol].max()
    assert xmin < 4.4 and xmax > 291, "Failed to select and clear"
    thread = threading.Thread(target=_trigger1, args=(data,))
    thread.start()
    result = data.search()
    xmin1, xmax1 = result[:, data.setas.cols.xcol].min(), result[:, data.setas.cols.xcol].max()
    assert np.isclose(xmin1, 50, atol=1) and np.isclose(xmax1, 100, 1), "Single selection failed."
    thread = threading.Thread(target=_trigger2, args=(data,))
    thread.start()
    result = data.search()
    xmin2, xmax2 = result[:, data.setas.cols.xcol].min(), result[:, data.setas.cols.xcol].max()
    assert np.isclose(xmin, xmin2) and np.isclose(xmax, xmax2), "Selection with keypresses failed"


if __name__ == "__main__":
    pytest.main(["--pdb", __file__])
