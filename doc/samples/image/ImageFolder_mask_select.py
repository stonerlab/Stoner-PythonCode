# -*- coding: utf-8 -*-
"""Demonstrate ImageFolder mask_select."""

import threading
import time
from matplotlib.backend_bases import Event
import numpy as np

### Some functions to allow the selection to be scripted
def _event(image, names, **kargs):
    """Make a fake event to simulate user input."""
    select = image._image._select
    event = Event("fake", select.fig.canvas)
    if not isinstance(names, list):
        names = [names]
    for name in names:
        for k, v in kargs.items():
            setattr(event, k, v)
        try:
            getattr(select, name)(event)
        except Exception as err:
            breakpoint()
            pass


def fake_user_action(image):
    """This function sends events to the selection widget to sumulate a user."""
    time.sleep(1)
    _event(
        image,
        ["draw", "on_click"],
        xdata=50,
        ydata=20,
        button=1,
        dblclick=False,
    )
    for y in np.linspace(20, 80, 10):
        _event(image, "keypress", xdata=50, ydata=y, key="c")
        time.sleep(0.1)
    _event(image, "keypress", xdata=50, ydata=80, key="c")
    time.sleep(0.5)
    _event(image, "keypress", xdata=50, ydata=80, key="i")
    time.sleep(0.5)
    _event(
        image,
        ["draw", "keypress"],
        xdata=50,
        ydata=80,
        button=1,
        dblclick=False,
        key="enter",
    )


from Stoner import ImageFolder, __homepath__
from Stoner.HDF5 import STXMImage

fldr = ImageFolder(
    __homepath__ / ".." / "sample-data",
    pattern="Sample*.hdf5",
    type=STXMImage,
    recursive=False,
)

# Start the scripted control
fake_user = threading.Thread(target=fake_user_action, args=(fldr[0],))
fake_user.start()

fldr.mask_select()
fldr.montage()
