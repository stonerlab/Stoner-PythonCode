# -*- coding: utf-8 -*-
"""Line and Box selection Tools for Images."""

import time
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.widgets import Cursor, RectangleSelector
from matplotlib.colors import to_rgba
from matplotlib.backend_bases import Event

from skimage import draw


def send_event(image, names, **kargs):
    """Make a fake event."""
    time.sleep(0.05)
    select = image._image._select
    event = Event("fake", select.fig.canvas)
    if not isinstance(names, list):
        names = [names]
    for name in names:
        for k, v in kargs.items():
            setattr(event, k, v)
        getattr(select, name)(event)


def _straight_ellipse(p, data):
    """A non-rotated ellipse."""
    xc, yc, a, b = p
    x, y = data.T
    t1 = (x - xc) ** 2 / a**2
    t2 = (y - yc) ** 2 / b**2
    return np.abs(np.sum(t1 + t2 - 1.0))


def _rotated_ellipse(p, data):
    """A non-rotated ellipse."""
    xc, yc, a, b, phi = p
    x, y = data.T
    t1 = ((x - xc) * np.cos(phi) + (y - yc) * np.sin(phi)) ** 2 / a**2
    t2 = ((x - xc) * np.sin(phi) - (y - yc) * np.cos(phi)) ** 2 / b**2
    return np.abs(np.sum(t1 + t2 - 1.0))


class LineSelect:
    """Show an Image and slow the user to draw a line on it using cursors."""

    def __init__(self):
        """Create the LineSelect object, display the image and register the hooks.

        The constructor will wait until the finished coordinates are set and then return
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

        Records with the starting or finishing coordinates of the line.
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

        Optiuonal line properties can be overridden by passing keyword parameters to the constructor.
        """
        if not self.started:  # Do nothing until we start
            return
        if len(self.ax.lines) > 2:  # Rremove the old line
            self.ax.lines[2].remove()

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

        The constructor will wait until the finished coordinates are set and then return
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


class ShapeSelect:
    """Show an Image and slow the user to draw a line on it using cursors."""

    def __init__(self):
        """Create the LineSelect object, display the image and register the hooks.

        The constructor will wait until the finished coordinates are set and then return
        [(x_start,y_start),(x_finish, y_finish)].
        """
        self.invert = False
        self.ov_layer = None
        self.shape = None
        self.colour = "red"
        self.alpha = 0.5
        self.obj = "poly"
        self.finished = False
        self.vertices = []
        self.ax = None
        self.fig = None
        self.crs = None
        self.mode = "xy"
        self.kargs = {}

    def __call__(self, image, **kargs):
        """Do the work of the polygon selection.

        Args:
            image (ImageArray, ImageFile):
                The image to shopw to the user for the selection
            **kargs (mixed):
                Other keywords to pass to the drawing.

        Returns:
            mask_array:
                A boolean array of te same shape as the image that can be used as a mask.

        """
        self.fig = image.imshow()
        self.shape = image.shape
        self.ax = plt.gca()
        # Sort colours out
        self.colour = kargs.pop("colour", getattr(image.mask, "colour", "red"))
        self.alpha = kargs.pop("alpha", 0.5)
        self.colour = np.array(to_rgba(self.colour))
        self.colour[3] = self.alpha
        # Create overlay plot
        overlay = np.zeros(self.shape + (4,))
        self.ax.imshow(overlay)
        self.ov_layer = self.ax.images[-1]

        self.kargs = kargs
        image._select = self  # allows us to hook to the selector.
        self.crs = Cursor(self.ax)
        plt.title(self.draw_poly.instructions)
        plt.xlabel(
            "LMB: Select, RMB: remove, LMB-Dbl: finish, Esc: cancel\ni: invert,"
            + "c: circle/ellipse, r:rectangle, p:polygon"
        )
        plt.pause(0.1)
        bp_ev = self.crs.connect_event("button_press_event", self.on_click)
        kp_ev = self.crs.connect_event("motion_notify_event", self.draw)
        mm_ev = self.fig.canvas.mpl_connect("key_press_event", self.keypress)
        while not self.finished:
            plt.pause(0.01)
        delattr(image, "_select")  # cleanup
        plt.disconnect(bp_ev)
        plt.disconnect(kp_ev)
        plt.disconnect(mm_ev)
        plt.close(self.fig.number)
        return self.get_mask()

    def keypress(self, event):
        """Habndle key press events.

        Args:
            event (matplotlib event):
                The matplotlib event object

        Returns:
            None.

        Tracks the mode and adjusts the cursor display.
        """
        shapes = {"p": "poly", "c": "circle", "r": "rectangle"}
        if event.key.lower() in shapes:
            self.obj = shapes[event.key.lower()]
            plt.title(getattr(getattr(self, f"draw_{self.obj}", ""), "instructions", ""))
            return self.draw(event)
        if event.key.lower() == "i":
            self.invert = not self.invert
            return self.draw(event)
        if event.key.lower() == "enter":
            self.vertices.append((event.xdata, event.ydata))
            event.button = 1
            if len(self.vertices) >= getattr(getattr(self, f"draw_{self.obj}", ""), "min_vertices", 0):
                event.dblclick = True
                return self.on_click(event)
            return self.draw(event)
        if event.key == "escape":
            self.vertices = []
            self.invert = False
            self.ov_layer.set_array(np.ones(self.shape + (4,)))
            self.finished = True
            return self.draw(event)

    def on_click(self, event):
        """Habndle mouse click events.

        Args:
            event (matplotlib event):
                The matplotlib event object

        Returns:
            None.

        Single left clicking adds new vertices to the list, double clicking adds vertices and finishes. Right
        clicking removes vertices.
        """
        if event.button == 1:  # Left mouse button
            if not event.dblclick:  # discard second click vertex addition.
                self.vertices.append((event.xdata, event.ydata))
            else:
                self.finished = True
        elif event.button == 3 and len(self.vertices) > 0:  # Right click toi remove last vertex
            del self.vertices[-1]
            if len(self.vertices) == 0:
                self.ov_layer.set_array(np.zeros(self.shape + (4,)))
            else:
                self.draw(event)

    def draw(self, event=None):
        """Handle the drawing of the selection shape.

        Keyword Args:
            event (matplotlib event):
                The matplotlib event object

        Returns:
            None.

        Delegates drawing to a draw_{shape} method.
        """
        if event is not None:
            vertices = np.array(self.vertices + [(event.xdata, event.ydata)])
        else:
            vertices = np.array(self.vertices)
        if vertices.size < 4:
            return
        vertices = np.round(vertices).astype(int)
        meth = getattr(self, f"draw_{self.obj}", lambda x: ([], []))
        rr, cc = meth(vertices)
        if self.invert:
            overlay = np.ones(self.shape + (4,))
            overlay[:, :] *= self.colour
            overlay[rr, cc] = [0, 0, 0, 0]
        else:
            overlay = np.zeros(self.shape + (4,))
            overlay[rr, cc] = self.colour

        # Add handles for the vertices.
        vertex_colour = np.copy(self.colour)
        vertex_colour[3] = 1.0  # Make vertices solid
        overlay[self.draw_vertices(vertices)] = vertex_colour

        self.ov_layer.set_array(overlay)

    def draw_vertices(self, vertices):
        """Return coordinates for small circules at each vertex."""
        radius = max(self.shape) / 50
        rr = np.array([], dtype=int)
        cc = np.array([], dtype=int)
        for vertex in vertices:
            vertex = (vertex[1], vertex[0])
            r, c = draw.disk(vertex, radius)
            rr = np.append(rr, r)
            cc = np.append(cc, c)
        return rr, cc

    def draw_poly(self, vertices):
        """Draw a polygon method using the specified vertices.

        Returns rr,cc coordinates.
        """
        if len(vertices) < 2:
            return ([], [])
        if len(vertices) == 2:
            rr, cc = draw.line(vertices[0, 1], vertices[0, 0], vertices[1, 1], vertices[1, 0])
        else:
            rr, cc = draw.polygon(vertices[:, 1], vertices[:, 0], self.shape)
        return rr, cc

    draw_poly.min_vertices = 3
    draw_poly.instructions = "Click to add vertices in order."

    def draw_circle(self, vertices):
        """Draw a circule or elipsoid."""
        if len(vertices) < 2:
            return ([], [])
        if len(vertices) == 2:  # Circle mode
            c0, r0 = vertices[0]
            c1, r1 = vertices[1]
            cc = (c0 + c1) // 2
            rc = (r0 + r1) // 2
            r = np.sqrt((rc - r1) ** 2 + (cc - c1) ** 2)
            return draw.disk((rc, cc), r, shape=self.shape)
        if len(vertices) == 3:
            p0 = vertices[0]
            p1 = vertices[1]
            p2 = vertices[2]
            bc = (np.dot(p0, p0) - np.dot(p1, p1)) / 2
            cd = (np.dot(p1, p1) - np.dot(p2, p2)) / 2
            det = (p0[0] - p1[0]) * (p1[1] - p2[1]) - (p1[0] - p2[0]) * (p0[1] - p1[1])
            if abs(det) < 1.0e-6:
                return ([], [])
            cx = (bc * (p1[1] - p2[1]) - cd * (p0[1] - p1[1])) / det
            cy = ((p0[0] - p1[0]) * cd - (p1[0] - p2[0]) * bc) / det
            r = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)
            return draw.disk((cy, cx), r, shape=self.shape)
        if len(vertices) == 4:
            xc, yc = vertices.mean(axis=0)
            a, b = vertices.max(axis=0) - vertices.min(axis=0)
            x0 = [xc, yc, a, b]
            result = minimize(_straight_ellipse, x0=x0, args=(vertices,))
            xc, yc, a, b = result.x
            return draw.ellipse(yc, xc, b, a, shape=self.shape)
        if len(vertices) > 4:
            xc, yc = vertices.mean(axis=0)
            a, b = vertices.max(axis=0) - vertices.min(axis=0)
            phi = 0
            x0 = [xc, yc, a, b, phi]
            result = minimize(_rotated_ellipse, x0=x0, args=(vertices,))
            xc, yc, a, b, phi = result.x
            return draw.ellipse(yc, xc, b, a, shape=self.shape, rotation=phi)

    draw_circle.min_vertices = 2
    draw_circle.instructions = "2 or 3 perimeter vertices to define a circle\n4 or more to define an ellipse"

    def draw_rectangle(self, vertices):
        """Calculate the coordinates for a rectangle from the vertices."""
        if len(vertices) < 2:
            return ([], [])
        if len(vertices) == 2:
            c0, r0 = vertices[0]
            c1, r1 = vertices[1]
            rr, cc = draw.rectangle((r0, c0), (r1, c1), shape=self.shape)
        else:
            c0, r0 = vertices[-3]
            c1, r1 = vertices[-2]
            c2, r2 = vertices[-1]
            theta = np.arctan2(c1 - c0, r0 - r1)
            xp, yp = np.cos(theta), np.sin(theta)
            w = np.dot([c2 - c1, r2 - r1], [xp, yp])
            c2 = c1 + w * xp
            r2 = r1 + w * yp
            c3 = c0 + w * xp
            r3 = c0 + w * yp
            xvert = np.array([c0, c1, c2, c3], dtype=int)
            yvert = np.array([r0, r1, r2, r3], dtype=int)
            rr, cc = draw.polygon(yvert, xvert, shape=self.shape)
        return rr.astype(int), cc.astype(int)

    draw_rectangle.min_vertices = 2
    draw_rectangle.instructions = "Click to add corner vertices."

    def get_mask(self):
        """Convert a list of vertices to a mask array."""
        mask = np.ones(self.shape, dtype=bool) & self.invert
        vertices = np.array(self.vertices)
        meth = getattr(self, f"draw_{self.obj}", lambda x: ([], []))
        rr, cc = meth(vertices)
        mask[rr, cc] = not self.invert
        return mask
