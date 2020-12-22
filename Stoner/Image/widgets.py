# -*- coding: utf-8 -*-
"""Line and Box selection Tools for Images."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Cursor, RectangleSelector


class LineSelect:

    """Show an Image and slow the user to draw a line on it using cursors."""

    def __init__(self):
        """Create the LineSelect object, display the image and register the hooks.

        The constructor will wait until the finished co-ordinates are set and then return
        [(x_start,y_start),(x_finish, y_finish)].
        """
        self.started = False
        self.finished = False
        self.ax = None
        self.fig = None
        self.crs = None
        self.mode = "xy"
        self.kargs = {}

    def __call__(self, image, **kargs):
        """Do the actual line selection.

        Args:
            image (ImageArray, ImageFile):
                The image to shopw to the user for the selection
            **kargs (mixed):
                Other keywords to pass to the line drawing.

        Returns:
            list:
                [(x_start,y_start),(x_finish,y_finish)] coordinates for the endpoints of the line

        """
        self.fig = image.imshow()
        self.ax = plt.gca()
        self.kargs = kargs
        image._select = self  # allows us to hook to the selector.
        self.crs = Cursor(self.ax)
        self.crs.connect_event("button_press_event", self.on_click)
        self.crs.connect_event("motion_notify_event", self.draw_line)
        self.fig.canvas.mpl_connect("key_press_event", self.keypress)
        while not self.finished:
            plt.pause(0.01)
        plt.close(self.fig.number)
        delattr(image, "_select")  # cleanup
        return [self.started, self.finished]

    def keypress(self, event):
        """Habndle key press events.

        Args:
            event (matplotlib event):
                The matplotlib event object

        Returns:
            None.

        Tracks the mode and adjusts the cursor display.
        """
        if event.key.lower() == "x":
            if self.mode == "x":
                self.mode = "xy"
            else:
                self.mode = "x"
        elif event.key.lower() == "y":
            if self.mode == "y":
                self.mode = "xy"
            else:
                self.mode = "y"
        self.crs.horizOn = "y" in self.mode
        self.crs.vertOn = "x" in self.mode

    def on_click(self, event):
        """Habndle mouse click events.

        Args:
            event (matplotlib event):
                The matplotlib event object

        Returns:
            None.

        Records with the starting or finishing co-ordinates of the line.
        """
        if self.mode == "x":
            y1, y2 = self.ax.get_ylim()
            x = event.xdata
            self.started = (x, y1)
            self.finished = (x, y2)
            return
        if self.mode == "y":
            x1, x2 = self.ax.get_xlim()
            y = event.ydata
            self.started = (x1, y)
            self.finished = (x2, y)
            return
        if not self.started:
            self.started = event.xdata, event.ydata
        else:
            self.finished = event.xdata, event.ydata

    def draw_line(self, event):
        """Handle the drawing of the selection line.

        Args:
            event (matplotlib event):
                The matplotlib event object

        Returns:
            None.

        Optiuonal line properties can be overriden by passing keyword parameters to the constructor.
        """
        if not self.started:  # Do nothing until we start
            return
        if len(self.ax.lines) > 2:  # Rremove the old line
            del self.ax.lines[2]

        self.kargs.setdefault("linewidth", 2)
        self.kargs.setdefault("linestyle", "dash")
        self.kargs.setdefault("")

        xc, yc = event.xdata, event.ydata
        xs, ys = self.started
        plt.plot([xs, xc], [ys, yc], "r-", linewidth=2)


class RegionSelect:

    """Show an Image and slow the user to select a rectangular section."""

    def __init__(self):
        """Create the LineSelect object, display the image and register the hooks.

        The constructor will wait until the finished co-ordinates are set and then return
        [(x_start,y_start),(x_finish, y_finish)].
        """
        self.p1 = False
        self.p2 = False
        self.finished = False
        self.ax = None
        self.fig = None
        self.select = None
        self.kargs = {}

    def __call__(self, image, **kargs):
        """Actuall do the region selection.

        Args:
            image (ImageArray, ImageFile):
                The image to shopw to the user for the selection
            **kargs (mixed):
                Other keywords to pass to the line drawing.

        Returns:
            list:
                The x and y range coordinates, in te order left-x, right-x, top-y, bottom-y

        """
        self.fig = image.imshow()
        plt.title("Click and drag to select region and press return")
        self.ax = plt.gca()
        self.kargs = kargs
        image._select = self  # allows us to hook to the selector.
        self.select = RectangleSelector(
            self.ax, self.on_select, button=[1], minspanx=5, minspany=5, useblit=True, interactive=True
        )
        self.fig.canvas.mpl_connect("key_press_event", self.finish)
        while not self.finished:
            plt.pause(0.01)
        plt.close(self.fig.number)
        delattr(image, "_select")  # cleanup
        return [
            min(self.p1[0], self.p2[0]),
            max(self.p1[0], self.p2[0]),
            min(self.p1[1], self.p2[1]),
            max(self.p1[1], self.p2[1]),
        ]

    def on_select(self, start, stop):
        """Habndle mouse click events.

        Args:
            start (matplotlib event):
                The matplotlib event object for the start of the selection
            stop (matplotlib event):
                The matplotlib event object for the start of the selection

        Returns:
            None.

        Records with the locations of the corners of the rectangular section.
        """
        self.p1 = (int(np.round(start.xdata)), int(np.round(start.ydata)))
        self.p2 = (int(np.round(stop.xdata)), int(np.round(stop.ydata)))
        self.select.active = True

    def finish(self, key_event):
        """Handle ley press events.

        Args:
            key_event (matplotlib event):
                The matplotlib event object

        Returns:
            None.

        Records with the Finish of the selection on <enter>.
        """
        if self.p1 and self.p2 and key_event.key == "enter":
            self.finished = True
