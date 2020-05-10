# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:04:58 2020

@author: phygbu
"""

import sys
from warnings import warn
import pytest
from pathlib import Path

import Stoner
ret_pth = Stoner.__homepath__/".."/"sample-data"/"TDI_Format_RT.txt"

#Horrible hack to patch QFileDialog  for testing

import Stoner.tools.widgets as widgets
from Stoner import Data, DataFolder


def test_filedialog():

    def dummy(mode="getOpenFileName"):
        modes={"getOpenFileName":ret_pth,
               "getOpenFileNames":[ret_pth],
               "getSaveFileName":None,
               "getExistingDirectory":ret_pth.parent}
        return lambda *args,**kargs:(modes[mode],None)

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


    widgets=sys.modules["Stoner.tools.widgets"]
    app=getattr(widgets,"App")
    setattr(app,"modes",modes)

    assert widgets.fileDialog.openDialog()== ret_pth
    assert widgets.fileDialog.openDialog(title="Test",start=".")== ret_pth
    assert widgets.fileDialog.openDialog(patterns={"*.bad":"Very bad files"})== ret_pth
    assert widgets.fileDialog.openDialog(mode="OpenFiles")== [ret_pth]
    assert widgets.fileDialog.openDialog(mode="SaveFile")== None
    assert widgets.fileDialog.openDialog(mode="SelectDirectory")== ret_pth.parent
    with pytest.raises(ValueError):
        widgets.fileDialog.openDialog(mode="Whateve")

def test_loader():
    d=Data(False)
    assert d.shape==(1676,3),"Failed to load data with dialog box"
    with pytest.raises(RuntimeError):
        d.save(False)
    fldr=DataFolder(False)
    assert fldr.shape==(48, {'attocube_scan': (15, {}), 'NLIV': (11, {}), 'recursivefoldertest': (1, {}), 'working': (4, {})})
    fldr=DataFolder(False,multifile=True)
    assert fldr.shape==(1, {}), "multifile mode failed!"

if __name__ == "__main__":
    pytest.main(["--pdb",__file__])