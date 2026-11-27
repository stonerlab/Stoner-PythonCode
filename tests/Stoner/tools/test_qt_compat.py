"""Check Qt binding selection in isolated interpreters."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("binding", ["PyQt5", "PyQt6", "PySide6", "PySide2"])
def test_shared_qt_binding(binding):
    """Dialogs, clipboard and plotting must use the same requested Qt binding."""
    if importlib.util.find_spec(binding) is None:
        pytest.skip(f"{binding} is not installed")
    env = dict(os.environ, QT_API=binding.lower(), QT_QPA_PLATFORM="offscreen", MPLBACKEND="QtAgg")
    result = subprocess.run(
        [sys.executable, "-c", """
import sys
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends import qt_compat

# Select the plotting binding before Stoner imports its file dialogs.
from matplotlib.backends import backend_qtagg
from Stoner.tools import widgets
from Stoner.Image import imagefuncs

assert qt_compat.QT_API.lower() == sys.argv[1].lower()
assert widgets.QApplication is qt_compat.QtWidgets.QApplication
assert imagefuncs.QApplication is widgets.QApplication
assert imagefuncs.QImage is qt_compat.QtGui.QImage
assert widgets.QT_VERSION == int(qt_compat.QtCore.qVersion().split('.')[0])
with warnings.catch_warnings():
    warnings.simplefilter('error', UserWarning)
    fig = plt.figure()
    fig.canvas.draw()
    plt.close(fig)

# Exercise the real GUI polling loop and a callback delivered by Qt itself.
import threading
import numpy as np
from matplotlib.backend_bases import KeyEvent
from Stoner import Data
data = Data(np.column_stack((np.arange(10), np.arange(10))), setas='xy')
callbacks = []
def select_range():
    assert threading.current_thread() is threading.main_thread()
    selector = data._select
    selector.onselect(2, 5)
    selector.keypress(KeyEvent('key_press_event', selector.ax.figure.canvas, key='enter'))
    callbacks.append(True)
qt_compat.QtCore.QTimer.singleShot(300, select_range)
selected = data.search()
assert callbacks == [True]
np.testing.assert_array_equal(selected.x, [3, 4, 5])
assert not hasattr(data, '_select')
""", binding],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_qt_binding():
    """Non-interactive use must still work when Qt bindings are unavailable."""
    result = subprocess.run(
        [sys.executable, "-c", """
import importlib.abc
import sys

class NoQt(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PyQt5', 'PyQt6', 'PySide6', 'PySide2'}:
            raise ImportError('Qt unavailable for this test')

sys.meta_path.insert(0, NoQt())
from Stoner.tools import widgets
from Stoner.Image import imagefuncs
assert widgets.QT_VERSION is None
assert imagefuncs.QImage is None
try:
    widgets.file_dialog.open_dialog()
except ValueError:
    pass
else:
    raise AssertionError('Missing Qt should prevent file dialogs')
import matplotlib.pyplot as plt
plt.figure().canvas.draw()
plt.close('all')
"""],
        cwd=Path(__file__).resolve().parents[3],
        env=dict(os.environ, MPLBACKEND="Agg"),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
