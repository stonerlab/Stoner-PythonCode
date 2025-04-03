# -*- coding: utf-8 -*-
"""Demonstrate ImageFolder mask_select."""

import threading
import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

from Stoner import ImageFolder, __datapath__
from Stoner.HDF5 import STXMImage
from Stoner.Image.widgets import send_event as _event


def fake_user_action(image):
    """Send events to the selection widget to simulate a user."""
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
        _event(image, "draw", xdata=50, ydata=y)
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


def extra_ops(i, j, image):
    """Add Extra operation to image."""
    pixel_size = np.diff(image["sample_x"]).mean() * 1e-6
    ax = image["ax"]
    scalebar = ScaleBar(pixel_size)
    ax.add_artist(scalebar)


fldr = ImageFolder(
    __datapath__, pattern="Sample*.hdf5", type=STXMImage, recursive=False
)

# Start the scripted control
fake_user = threading.Thread(target=fake_user_action, args=(fldr[0],))
fake_user.start()

fldr.mask_select()
for i in range(4):
    fldr += fldr[0]
    fldr += fldr[1]
fig = plt.figure(figsize=(8, 4))
fldr.montage(figure=fig, plots_per_page=4, plot_extra=extra_ops)
