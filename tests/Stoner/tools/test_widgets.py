# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:04:58 2020

@author: phygbu
"""


import pytest
from pathlib import Path
from PyQt5.QtCore import QTimer
from Stoner import __home__
ret_pth = Path(__home__)/".."/"sample-data"/"TDI_Format_RT.txt"

#Horrible hack to patch QFileDialog  for testing

def dummy(mode="getOpenFileName"):
    modes={"getOpenFileName":str(ret_pth),
           "getOpenFileNames":[str(ret_pth)],
           "getSaveFileName":str(ret_pth),
           "getExistingDirectory":str(ret_pth.parent)}
    return lambda *args,**kargs:modes[mode]

from PyQt5.QtWidgets import QFileDialog
for mode in ["getOpenFileName","getOpenFileNames","getSaveFileName","getExistingDirectory"]:
    setattr(QFileDialog,mode,dummy(mode))

import Stoner.tools.widgets as widgets
from Stoner import Data, DataFolder


def test_filedialog():

    assert widgets.fileDialog.openDialog()== str(ret_pth)
    assert widgets.fileDialog.openDialog(title="Test",start=".")== str(ret_pth)
    assert widgets.fileDialog.openDialog(patterns={"*.bad":"Very bad files"})== str(ret_pth)
    assert widgets.fileDialog.openDialog(mode="OpenFiles")== [str(ret_pth)]
    assert widgets.fileDialog.openDialog(mode="SaveFile")== str(ret_pth)
    assert widgets.fileDialog.openDialog(mode="SelectDirectory")== str(ret_pth.parent)
    with pytest.raises(ValueError):
        widgets.fileDialog.openDialog(mode="Whateve")

def test_loader():
    d=Data(False)
    assert d.shape==(1676,3) or d.shape==(100,2),"Failed to load data with dialog box"
    fldr=DataFolder(False)
    assert fldr.shape==(47, {'attocube_scan': (15, {}), 'NLIV': (11, {}), 'recursivefoldertest': (1, {}), 'working': (4, {})})
    fldr=DataFolder(False,multifile=True)
    assert fldr.shape==(1, {}), "multifile mode failed!"

if __name__ == "__main__":
    pytest.main(["--pdb",__file__])