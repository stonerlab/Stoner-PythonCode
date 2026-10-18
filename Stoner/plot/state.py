"""Keep transient plotting targets separate from copied measurement state."""

from dataclasses import dataclass

from matplotlib._pylab_helpers import Gcf


def live_figure(figure):
    """Return whether a figure is unmanaged or still belongs to its pyplot manager."""
    if figure is None:
        return False
    # Querying the manager registry does not select/create a figure, unlike gcf.
    return not hasattr(figure, "number") or any(
        manager.canvas.figure is figure for manager in Gcf.get_all_fig_managers()
    )


@dataclass
class PlotState:
    """Record recent rendering targets without copying or serialising live artists."""

    figure: object = None
    ax: object = None

    def __deepcopy__(self, memo):
        """Detach rendering targets when the owning Data is cloned."""
        result = type(self)()
        memo[id(self)] = result
        return result

    def __getstate__(self):
        """Exclude live Matplotlib objects from serialised measurement state."""
        return {}

    def __setstate__(self, state):
        """Restore an empty rendering state."""
        self.figure = None
        self.ax = None
