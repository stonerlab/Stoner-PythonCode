"""Keep the upstream regressions required by the stonerplots dependency floor."""

import matplotlib.pyplot as plt
import stonerplots
from matplotlib.ticker import MultipleLocator


def test_inset_restores_parent():
    fig, parent = plt.subplots()
    try:
        with stonerplots.InsetPlot(ax=parent, loc="upper right") as inset:
            inset.plot([0, 1], [0, 1])
        assert plt.gca() is parent
    finally:
        plt.close(fig)


def test_labeller_allocates_locator_per_axis():
    try:
        with stonerplots.PlotLabeller(y=MultipleLocator):
            _, axes = plt.subplots(1, 2)
            axes[0].set_ylim(0, 2)
            axes[1].set_ylim(10, 12)
        assert axes[0].yaxis.get_major_locator() is not axes[1].yaxis.get_major_locator()
        assert axes[0].yaxis.get_major_locator().axis is axes[0].yaxis
    finally:
        plt.close("all")
