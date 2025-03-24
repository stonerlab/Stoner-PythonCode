# -*- coding: utf-8 -*-
"""Creat File dialog boxes using the PyQt5 module.

Code based on the PyQt5 Tutorial code,
"""
__all__ = ["fileDialog"]
import pathlib
from typing import Any, Union, Optional, Dict, Type
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from matplotlib.patches import Rectangle

from ..compat import mpl_version

QT_VERSION = None
try:
    from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication

    QT_VERSION = 5
except ImportError:
    pass
if QT_VERSION is None:
    try:
        from PyQt6.QtWidgets import QWidget, QFileDialog, QApplication

        QT_VERSION = 6
    except ImportError:
        pass
if QT_VERSION is None:

    class App:
        """Mock App that raises an error when you try to call openDialog on it."""

        modes: Dict = dict()

        def openDialog(
            self,
            title: Optional[str] = None,
            start: Union[str, pathlib.Path] = "",
            patterns: Optional[Dict] = None,
            mode: str = "OpenFile",
        ) -> Optional[pathlib.Path]:
            """Raise and error because PyQT5 not present."""
            raise ValueError("Cammpt open a dialog box because PqQt5 library missing!")

else:

    class App(QApplication):
        """Placehold PyQT5 Application for producing filedialog boxes."""

        modes = {
            "OpenFile": {
                "method": QFileDialog.getOpenFileName,
                "caption": "Select file to open...",
                "arg": ["parent", "caption", "directory", "filter", "options"],
            },
            "OpenFiles": {
                "method": QFileDialog.getOpenFileNames,
                "caption": "Select file(s_ to open...",
                "arg": ["parent", "caption", "directory", "filter", "options"],
            },
            "SaveFile": {
                "method": QFileDialog.getSaveFileName,
                "caption": "Save file as...",
                "arg": ["parent", "caption", "directory", "filter", "options"],
            },
            "SelectDirectory": {
                "method": QFileDialog.getExistingDirectory,
                "caption": "Select folder...",
                "arg": ["parent", "caption", "directory", "options"],
            },
        }

        def __init__(self, *args: Any, **kargs: Any) -> None:
            super().__init__([""], *args, **kargs)
            self.title = "PyQt5 file dialogs - pythonspot.com"
            self.left = 10
            self.top = 10
            self.width = 640
            self.height = 480
            self.initUI()

        def initUI(self) -> None:
            self.dialog = QWidget()
            self.dialog.title = "PyQt5 file dialogs - pythonspot.com"
            self.dialog.left = 10
            self.dialog.top = 10
            self.dialog.width = 640
            self.dialog.height = 480

            self.dialog.setWindowTitle(self.title)
            self.dialog.setGeometry(self.left, self.top, self.width, self.height)

        def openDialog(
            self,
            title: Optional[str] = None,
            start: Union[str, pathlib.Path] = "",
            patterns: Optional[Dict] = None,
            mode: str = "OpenFile",
        ) -> Optional[pathlib.Path]:
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
            try:
                options = QFileDialog.Options()
            except AttributeError:
                options = QFileDialog().options()

            kwargs = {"caption": title, "directory": str(start), "filter": patterns, "options": options, "modal": True}
            kwargs = {k: kwargs[k] for k in (set(kwargs.keys()) & set(self.modes[mode]["arg"]))}

            ret = method(self.dialog, **kwargs)

            if isinstance(ret, tuple):
                ret = ret[0]
            if isinstance(ret, (str, pathlib.PurePath)):
                ret = pathlib.Path(ret)
            elif isinstance(ret, list):
                ret = [pathlib.Path(x) for x in ret]
            elif not ret:
                ret = None
            else:
                raise TypeError(f"Something when wrong here - can't handle {ret} as a {type(ret)}")
            return ret


class RangeSelect:
    """A simple class to allow a matplotlib graph to be used to select data."""

    def __init__(self):
        """Initialise the selector state."""
        self.data = None
        self.finished = False
        self.selector = []
        self.invert = False
        self.xcol = None
        self.ycol = None
        self.selection = []
        self.ax = None

    def __call__(self, data, xcol, accuracy, invert=False):
        """Run the selector with the data."""
        self.data = data
        self.data._select = self  # To allow for unit testing
        self.xcol = xcol
        self.invert = invert

        col = "red" if self.invert else "green"

        if len(self.data.setas.y) > 0:
            self.ycol = self.data.setas.y
        else:
            self.ycol = list(range(self.data.shape[1]))
            self.ycol.remove(self.xcol)
        # Preserve figure settings before creating plot
        fig_tmp = getattr(self.data, "fig", None), getattr(self.data, "axes", None)
        fig = plt.figure()
        self.data.plot(self.xcol, self.ycol, figure=fig)
        self.data.title = "Select Data and press Enter to confirm\nEsc to cancel, i to invert selection."
        self.ax = self.data.axes[-1]
        if mpl_version.minor >= 5:
            kargs = {"props": {"edgecolor": col, "facecolor": col, "alpha": 0.5}}
        else:
            kargs = {"rectprops": {"edgecolor": col, "facecolor": col, "alpha": 0.5}}
        self.selector = SpanSelector(
            self.ax,
            self.onselect,
            "horizontal",
            useblit=True,
            **kargs,
        )
        fig.canvas.mpl_connect("key_press_event", self.keypress)
        while not self.finished:
            plt.pause(0.1)
        # Clean up and restore the figure settings
        plt.close(self.data.fig.number)
        if fig_tmp[0] is not None and fig_tmp[0] in plt.get_fignums():
            self.data.fig = fig_tmp[0]
        delattr(self.data, "_select")
        idx = np.ones(len(self.data), dtype=bool)
        for ix, selection in enumerate(self.selection):
            if ix == 0:
                idx = self.data._search_index(xcol, selection, accuracy, invert=self.invert)
            else:
                idx = np.logical_or(idx, self.data._search_index(xcol, selection, accuracy, invert=self.invert))
        return idx

    def onselect(self, xmin, xmax):
        """Add the selection limits to the selections list."""
        self.selection.append((xmin, xmax))
        ylim = self.ax.get_ylim()
        col = "red" if self.invert else "green"
        rect = Rectangle((xmin, ylim[0]), xmax - xmin, ylim[1] - ylim[0], edgecolor=col, facecolor=col, alpha=0.5)
        self.ax.add_patch(rect)

    def keypress(self, event):
        """Habndle key press events.

        Args:
            event (matplotlib event):
                The matplotlib event object

        Returns:
            None.

        Tracks the mode and adjusts the cursor display.
        """
        if event.key.lower() == "enter":  # Finish selection
            self.finished = True
        elif event.key.lower() == "escape":  # Abandon selection
            self.selection = []
            self.finished = True
        elif event.key.lower() == "backspace":  # Delete last selection
            if len(self.selection) > 0:
                del self.selection[-1]
                self.ax.patches[-1].remove()
                self.data.fig.canvas.draw()
        elif event.key.lower() == "i":  # Invert selection
            self.invert = not self.invert
            col = "red" if self.invert else "green"
            for p in self.ax.patches:
                p.update({"color": col, "facecolor": col})


fileDialog: Type[App] = App()
