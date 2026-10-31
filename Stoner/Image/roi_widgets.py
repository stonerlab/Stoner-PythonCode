"""Provide cancellable ROI widget sessions without changing legacy mask selectors."""

import matplotlib.pyplot as plt
from matplotlib._pylab_helpers import Gcf
from matplotlib.widgets import PolygonSelector, RectangleSelector

from ..core.roi import ROI, geometry_signature
from .coordinates import resolve_grid
from .storage_bridge import owner_of


class _ROISelection:
    """Keep pending geometry separate from the owner's accepted selection."""

    def __init__(self, image, kind, ax, coordinates):
        self.image, self.ax, self.coordinates = image, ax, coordinates
        self.pending = None
        self.accepted = False
        self.finished = False
        self.reference = owner_of(image)._state.dataset.intensity
        self.signature = geometry_signature(self.reference)
        if kind == "rectangle":
            self.widget = RectangleSelector(ax, self.rectangle, button=[1], interactive=True, useblit=True)
        else:
            self.widget = PolygonSelector(ax, self.polygon, useblit=True)
        canvas = ax.figure.canvas
        self.connections = [canvas.mpl_connect("key_press_event", self.key),
                            canvas.mpl_connect("close_event", self.close)]

    def rectangle(self, start, stop):
        x0, x1 = sorted((start.xdata, stop.xdata))
        y0, y1 = sorted((start.ydata, stop.ydata))
        self.pending = ROI.box(x0, x1, y0, y1, coordinates=self.coordinates, reference=self.reference)

    def polygon(self, vertices):
        self.pending = ROI.polygon(vertices, coordinates=self.coordinates, reference=self.reference)

    def key(self, event):
        if event.key == "escape":
            self.close(event)
        elif event.key in {"enter", "return"} and self.pending is not None:
            self.accepted = True
            self.close(event)

    def close(self, event):
        self.finished = True
        self.ax.figure.canvas.stop_event_loop()

    def disconnect(self):
        self.widget.disconnect_events()
        for connection in self.connections:
            self.ax.figure.canvas.mpl_disconnect(connection)


def select_roi(image, kind, *, ax, coordinates):
    """Run an explicit interactive session, publishing only accepted valid geometry."""
    resolve_grid(image, coordinates=coordinates)  # Validate before creating any figure.
    if "_roi_selector" in image.__dict__:
        raise RuntimeError("An ROI selection is already running for this image.")
    from matplotlib.axes import Axes

    from ..plot.state import live_figure

    if ax is not None and (not isinstance(ax, Axes) or not live_figure(ax.figure)):
        raise ValueError("ax must be live Matplotlib Axes.")
    previous_manager = Gcf.get_active()
    owned = ax is None
    if owned:
        _, ax = plt.subplots()
    original_children = set(ax.get_children())
    original = (ax.get_xlim(), ax.get_ylim(), ax.get_xlabel(), ax.get_ylabel(), ax.get_title(), ax.get_aspect(),
                ax.get_autoscalex_on(), ax.get_autoscaley_on(), ax.dataLim.frozen())
    plot_state = image.__dict__.pop("_plot_state", None)
    session = None
    try:
        image.plot(ax=ax, coordinates=coordinates)
        ax.set_title("Select region; Enter accepts, Escape cancels")
        session = _ROISelection(image, kind, ax, coordinates)
        image.__dict__["_roi_selector"] = session
        plt.show(block=False)
        if not session.finished:
            ax.figure.canvas.start_event_loop(0)
        if session.accepted:
            if session.signature != geometry_signature(owner_of(image)._state.dataset.intensity):
                raise ValueError("Image geometry changed during ROI selection; previous ROI retained.")
            image.roi.set(session.pending)
    finally:
        if session is not None:
            session.disconnect()
        image.__dict__.pop("_roi_selector", None)
        for artist in set(ax.get_children()) - original_children:
            artist.remove()
        ax.set_xlim(original[0])
        ax.set_ylim(original[1])
        ax.set_xlabel(original[2])
        ax.set_ylabel(original[3])
        ax.set_title(original[4])
        ax.set_aspect(original[5])
        ax.set_autoscalex_on(original[6])
        ax.set_autoscaley_on(original[7])
        ax.dataLim.set(original[8])
        image.__dict__.pop("_plot_state", None)
        if plot_state is not None:
            image.__dict__["_plot_state"] = plot_state
        if owned:
            plt.close(ax.figure)
        if previous_manager is not None and previous_manager in Gcf.get_all_fig_managers():
            Gcf.set_active(previous_manager)
    return image
