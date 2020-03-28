# -*- coding: utf-8 -*-
"""Creat File dialog boxes using the PyQt5 module.

Code based on the PyQt5 Tutorial code,
"""
__all__ = ["fileDialog"]
from PyQt5.QtWidgets import QWidget, QFileDialog


class App(QWidget):

    modes = {
        "OpenFile": {
            "method": QFileDialog.getOpenFileName,
            "caption": "Select file to open...",
            "arg": ["caption", "directory", "filter", "options"],
        },
        "OpenFiles": {
            "method": QFileDialog.getOpenFileNames,
            "caption": "Select file(s_ to open...",
            "arg": ["caption", "directory", "filter", "options"],
        },
        "SaveFile": {
            "method": QFileDialog.getSaveFileName,
            "caption": "Save file as...",
            "arg": ["caption", "directory", "filter", "options"],
        },
        "SelectDirectory": {
            "method": QFileDialog.getExistingDirectory,
            "caption": "Select folder...",
            "arg": ["caption", "directory", "options"],
        },
    }

    def __init__(self):
        super().__init__()
        self.title = "PyQt5 file dialogs - pythonspot.com"
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

    def openDialog(self, title=None, start="", patterns=None, mode="OpenFile"):
        """Create a dialog box for selecting filenames or directories.

        Keyword Arguments:
            title (str, None):
                Label of the dialog box, default None will select something depending on the mode.
            start (str):
                The starting directory for the dialog box.
            patterns (dict):
                Filename patterns - the keys of the dictionary are glob patterns, the values the corresponding
                explanation of the file
                type.
            mode (str):
                Determines the type of filedialog box used. Values are:
                    -   "OpenFile" - get a single existing file
                    -   "OpenFiles" - get one or more existing files
                    -   "SaveFile" - get an existing or new file - warns about overwriting files
                    -   "SelectDirectory" - gets the name of an existing (possibly newly created) directory.

        Returns:
            (str, None):
                Either a string containing the absolute path to the file or directory, or None if the dialog
                was cancelled.
        """

        if mode not in self.modes:
            raise ValueError(f"Unknown dialog mode {mode}")
        method = self.modes[mode]["method"]
        if title is None:
            title = self.modes[mode]["caption"]
        if patterns is None:
            patterns = {"*.*": "All Files", "*.py": "Python Files"}
        patterns = ";;".join([f"{v} ({k})" for k, v in patterns.items()])
        options = QFileDialog.Options()

        kwargs = {"caption": title, "directory": start, "filter": patterns, "options": options}
        kwargs = {k: kwargs[k] for k in (set(kwargs.keys()) & set(self.modes[mode]["arg"]))}

        ret = method(self, **kwargs)

        if isinstance(ret, tuple):
            ret = ret[0]
        if not ret:
            ret = None
        return ret


fileDialog = App()
