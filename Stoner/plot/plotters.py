# -*- coding: utf-8 -*-
"""Functions to carry out different types of plots."""


def plot_xy(ax, data, **kargs):
    """Make a 2D plot optionally with error bars."""
    if "xerr" in data.dtype.fields:
        kargs["xerr"] = data["xerr"].ravel()
    if "yerr" in data.dtype.fields:
        kargs["yerr"] = data["yerr"].ravel()
    ax.errorbar(data["xcol"].ravel(), data["ycol"].ravel(), **kargs)
