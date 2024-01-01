# -*- coding: utf-8 -*-
"""Class that knows how to plot data."""
from functools import partial
import weakref

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ..tools import isiterable, AttributeStore
from ..Core import DataFile
from .formats import DefaultPlotStyle
from . import plotters
from .utils import add_properties


def _trim_kargs(kargs, columns=False, figure=False, annotations=False):
    """Trim keywrod arguments dictionary by moving groups of args to a secondary dictionary.

    Args:
        kargs (dict):
            Keyword Arguments to filter.

    Keyword Arguments:
        columns, figure, annotations (bool):
            Remove arguments associated with the corresponding class of items.

    Returns:
        (tuple of dict, dict):
            Classified arguments and unclassificed arguments.
    """
    args = {
        "columns": ["xcol", "ycol", "zcol", "xerr", "yerr", "zerr", "ucol", "vcol", "wcol"],
        "figure": [
            "figsize",
            "dpi",
            "facecolor",
            "edgecolor",
            "frameon",
            "subplotpars",
            "tight_layout",
            "constrained_layout",
        ],
        "annotations": ["xlabel", "ylabel", "zlabel", "title", "label_props"],
    }
    classified = {}
    for var, name in zip([columns, figure, annotations], ["columns", "figure", "annotations"]):
        if var:
            trim = set(args[name]) & set(kargs.keys())
            for k in trim:
                classified[k] = kargs.pop(k)

    return classified, kargs


def _setup_multi_axes(figure, ax, data, cols, kargs):
    """Create multiple set of axes on one figure.

    Args:
        figure (Figure):
            The figure instance we are working with
        ax (Axes):
            The first Axes instance on the figure (we can assume this already exists)
        data (list of 2D array):
            The data values we are plotting.
        kargs (dict):
            Other keyword arguments.

    Returns a list of axes that is the same length as data for plotting on.
    """
    axes = [ax]
    mode = kargs.pop("multi", "common")  # default operation - one set of axes
    if mode == "common" or "zcol" in cols:  # common axes or 3D plots bug out early
        return axes * len(data)  # Bugout - reuse the same axes each time
    cols = AttributeStore({k: [col.get(k, None) for col in cols] for k in ["xcol", "ycol"]})
    if mode == "y2":  # Workout whether we need new X and or Y axes.
        for ix, (xc, yc) in enumerate(zip(cols["xcol"], cols["ycol"])):
            if ix == 0:  # Skip first axes
                continue
            xi = cols["xcol"].index(xc)
            yi = cols["ycol"].index(yc)
            if (xi == ix and yi == ix) or (xi != ix and yi != ix and xi != yi):  # New X and Y cols
                ax = figure.add_subplot(111, label=str(ix), frame_on=False)
                newx, newy = True, True
            elif xi == ix:
                ax = ax.twiny()
                newx, newy = True, False
            elif yi == ix:
                ax = ax.twinx()
                newx, newy = False, True
            else:
                ax = axes[xi]
                newx, newy = False, False

            # Setup the labels for second an subsequent axes
            if newx:
                ax.xaxis.tick_top()
                ax.spines["top"].set_position(("axes", 0.8 + 0.2 * ix))
                ax.xaxis.set_label_position("top")
            if newy:
                ax.yaxis.tick_right()
                ax.spines["right"].set_position(("axes", 0.8 + 0.2 * ix))
                ax.yaxis.set_label_position("right")
            axes.append(ax)
        return axes
    if mode == "panels":  # Panels are subplots with common x or y axes
        if np.unique(cols["xcol"]).size == 1:
            axes = figure.subplots(nrows=len(cols.ycol), sharex=True, gridspec_kw={"hspace": 0})
        elif np.unique(cols["ycol"]).size == 1:
            axes = figure.subplots(ncols=len(cols.ycol), sharey=True, gridspec_kw={"vspace": 0})
        else:
            mode = "subplots"  # Can't set common x or y plots
    if mode == "subplots":
        m = int(np.floor(np.sqrt(len(cols.ycol))))
        n = int(np.ceil(len(cols.ycol) / m))
        axes = figure.subplots(nrows=m, ncols=n)
    return axes


def _fix_labels(ax, xlabel=None, ylabel=None, zlabel=None, title=None, props=None):
    """Fix the axes labels and titles for the current Axes.

    Args:
        ax (Axes):
            Current Axes instance to work with

    Keyword Arguments:
        xlabel,ylabel,zlabel,title (str or None):
            If None, no processesing of label is done (default), otherwise use str
        props (dict or None):
            Additional keyword arguments to pass. Can be a dictionary with keys xlabel, ylabel, zlabel title etc
            for arguments specific to one label or arguments common to all.

    Returns:
        (Axes):
            Copy of ax

    Notes:
        if the labels are set, the current value is checked, if it is the same as the new value, nothing is done,
        otherwise append the extra label with a separating ;
    """
    pr = {} if props is None else props  # initialise pr with the extra keywords
    xprops, yprops, zprops, tprops = (
        pr.pop("xlabel", {}),
        pr.pop("xlabel", {}),
        pr.pop("xlabel", {}),
        pr.pop("xlabel", {}),
    )
    for prop, label in zip([xprops, yprops, zprops, tprops], ["xlabel", "ylabel", "zlabel", "title"]):
        val = locals()[label]
        if val is None:  # If we've not set the label carry on.
            continue
        val = val.replace(";", " ")  # Needed to stop labels with ; from messing things up
        new_pr = prop.copy()
        new_pr.update(prop)
        current = [x for x in plt.get(ax, label).split(";") if x != ""]
        if val in current:
            continue
        else:
            current.append(val)
        current = ";".join(current)
        setter = getattr(ax, f"set_{label}")
        setter(current, **new_pr)
    return ax


def _build_dispatcher(data):
    """Workout the name of a plotting function to use given the data array supplied.

    Args:
        data (structured numy array):
            The data to plot as generated from PlotAttr._assemble_data

    Returns:
        (str):
            Name of plot method to call.

    Raises:
        (ValueError):
            If missing either x or y axis labels
    """
    root = "plot_"
    fields = data.dtype.fields
    for name, br in [
        ("xcol", False),
        ("ycol", False),
        ("zcol", False),
        ("ucol", True),
        ("vcol", True),
        ("wcol", True),
    ]:
        if name in fields:
            root += name[0]
        elif br:
            break
    if root.endswith("u"):
        root = root[:-1]
    if "x" not in root or "y" not in root:
        raise ValueError("Unable to determine plot type - must have an x and y axis defined!")
    return root


@add_properties((Axes, "_ax"), (Figure, "figure"))
class PlotAttr:

    """Defines an atribute that holds information relating to plotting data from a :py:class:`Stoner.Data` instance."""

    def __init__(self, data: DataFile):
        """Attach the DataFile instance to this PlotAttr instance."""
        self._data_ref = weakref.ref(data)
        self._template = DefaultPlotStyle()
        self._figure = None

    def __call__(self, *args, **kargs):
        """Determine the plotting function to use and then call it, creating a new figure etc as required."""

        figure = kargs.pop("figure", self._figure)
        ax = kargs.pop("ax", None)

        for ix, missing in enumerate(["xcol", "ycol", "zcol", "ucol", "vcol", "wcol"]):  # YTransfer args to kargs
            if len(args) > ix and missing not in kargs:
                kargs[missing] = args[ix]
            elif len(args) == ix:
                break
            else:
                raise ValueError(f"Unsure how to deal with argument {ix}")

        cols, data = self._assemble_data(**kargs)  # Get a list of data
        plot_funcs = [_build_dispatcher(d) for d in data]  # Get the plot functions
        num_plots = len(data)
        col_kargs, kargs = _trim_kargs(kargs, columns=True)
        fig_kargs, kargs = _trim_kargs(kargs, figure=True)
        annotations, kargs = _trim_kargs(kargs, annotations=True)

        if (figure is None or (isinstance(figure, bool) and not figure)) and not isinstance(
            ax, Axes
        ):  # new figure needed
            figure, ax = self.template.new_figure(None, **fig_kargs)
        elif isinstance(ax, Axes):  # I we provide a an Axes instance then use those
            figure = ax.get_figure()
            ax = figure.axes.index(ax)
        self.figure = figure

        if isinstance(ax, int) and len(self.axes) > ax:
            ax = self.axes[ax]
        if ax is None:
            ax = figure.gca()

        labels = self._setup_labels(cols, **annotations)
        if num_plots > 1:
            axes = _setup_multi_axes(figure, ax, data, cols, kargs)
        else:
            axes = [ax]
        for ix, (d, func, ax, plot_cols, plot_labels) in enumerate(zip(data, plot_funcs, axes, cols, labels)):
            func = getattr(plotters, func, partial(self._no_plotter, func))
            func(ax, d, **kargs)
            plot_lab = annotations.copy()
            plot_lab.update(plot_labels)
            _fix_labels(ax, **plot_lab)
            self.template.customise_axes(ax, self._data)

    def _no_plotter(self, name, *args, **kargs):
        """Catcher of illformed plotters."""
        raise RuntimeError(f"No plotting function defined for f{name} in Stoner.plot.plotters!")

    def _fix_cols(self, **kargs):
        """Sorts out axis specs, replacing with contents from setas as necessary.

        Returns:
            (list of AttributeStore):
                An AttributeStore for each plot."""
        kargs = kargs.copy()  # Avoid side effects !
        startx = kargs.pop("startx", 0)
        c = self._data.setas._get_cols(startx=startx)

        for k in ["xcol", "ycol", "zcol"]:  # Deal with tuples being passed as columns for col+err
            if k in kargs and isinstance(kargs[k], tuple) and len(kargs[k]) == 2:
                kargs[k[0] + "err"] = kargs[k][1]
                kargs[k] = kargs[k][0]

        for k in "xcol", "xerr":  # Sort out xcol and xerror
            if k not in kargs or kargs[k] is None:
                kargs[k] = c[k]
            elif not isinstance(kargs[k], np.ndarray):  # Deal with direct passing of values to plot
                kargs[k] = self._data.find_col(kargs[k], force_list=True)

        for k in ["ycol", "zcol", "ucol", "vcol", "wcol", "yerr", "zerr"]:  # these columns may be lists
            if k not in kargs or kargs[k] is None and isiterable(c[k]) and len(c[k]) > 0:
                kargs[k] = c[k]
            elif not isinstance(kargs[k], np.ndarray):  # Deal with direct passing of values to plot
                kargs[k] = self._data.find_col(kargs[k], force_list=True)

        # Clenup kargs of unwanted keys now
        for k in set(kargs.keys()) - {"xcol", "ycol", "zcol", "ucol", "vcol", "wcol", "xerr", "yerr", "zerr"}:
            del kargs[k]

        for k, v in kargs.items():  # Force all kargs to be iterable
            if not isiterable(v) or isinstance(v, np.ndarray):  # Also wrap arrays
                kargs[k] = [v]
        # Number of plots is defined by the biggest number of spatial axes
        num_plots = np.max([len(kargs.get(col, [])) for col in ["xcol", "ycol", "zcol"]])
        for col, err in [("xcol", "xerr"), ("ycol", "yerr"), ("zcol", "zerr")]:
            if col not in kargs or len(kargs[col]) == 0:
                kargs.pop(col, None)
                kargs.pop(err, None)  # Remove rror if spatial colun not defined.
                continue
            if len(kargs[col]) == 1 and len(kargs.get(err, [])) == 1:  # Single xd or zf specified.
                kargs[col] = kargs[col] * num_plots
                kargs[err] = kargs[err] * num_plots
            elif len(kargs[col]) != num_plots:  # Now fix the length of the spatial column
                kargs[col] = (kargs[col] * num_plots)[:num_plots]
            if err in kargs and len(kargs[err]) != num_plots:  # Pad up the error column with None
                kargs[err].extend([None] * num_plots)
                kargs[err] = kargs[err][:num_plots]

        for k in ["ucol", "vcol", "wcol"]:  # Fix length of other columns
            if k in kargs and len(kargs[k]) > 0 and len(kargs[k]) != num_plots:
                kargs[k] = (kargs[k] * num_plots)[:num_plots]
            else:
                kargs.pop(k, None)
        output = [AttributeStore({k: v[i] for k, v in kargs.items()}) for i in range(num_plots)]
        return output

    def _setup_labels(self, cols, **kargs):
        """Lookin the keyword arguments for xlabel/ylabel/zlabel/title and build into lists."""
        output = {"xlabel": [], "ylabel": [], "zlabel": [], "title": []}
        num_plots = len(cols)
        for arg, col in zip(["xlabel", "ylabel", "zlabel", "title"], ["xcol", "ycol", "zcol", ""]):
            # Get the label from the kargs and ensure it is a list of length num_plots
            label = kargs.get(arg, None)
            if not isinstance(label, list):
                label = [label]
            label = (label * num_plots)[:num_plots]

            if col == "":  # Title
                output[arg].extend(kargs.get(arg, [self._data.basename] * num_plots))
            elif col != "" or col not in cols[0]:  # If this isn't the title and the column is defined.
                for ix, col_index in enumerate(cols):  # For this column for each plot
                    if col_index.get(col, None) is not None and label[ix] is None:  # Havew column but not label
                        output[arg].append(self._data.column_headers[col_index[col]])  # Use the column_header
                    else:
                        output[arg].append(label[ix])  # Use the label
            else:
                output[arg].extend([None] * num_plots)  # Not needed.
        output = [
            {k: v[i] for k, v in output.items()} for i in range(num_plots)
        ]  # convert dict of lists to list of dicts
        return output

    def _assemble_data(self, **kargs):
        """Assemble the data to be plotted as a list of structured 2D arrays.

        Keyword Arguments:
            xcol, ycol, zcol, ucol, vcol, wcol, xerr, yerr, zerr (column index types):
                Names, numbers or lists that can be used to identify columns.

        Returns:
            (AttributeStore of colums, list of 2D structured arrays):
                Each element in the list represents a separate plot, the structured column names represent
                the columns to include in the plot.
        """
        cols = self._fix_cols(scalar=False, **kargs)
        output = []
        for i, plot_cols in enumerate(cols):
            names = ["xcol"]
            data = self._data[:, plot_cols.xcol]
            for k in ["xerr", "ycol", "yerr", "zcol", "zerr", "ucol", "vcol", "wcol"]:
                if plot_cols.get(k, None) is not None:
                    names.append(k)
                    data = np.column_stack([data, self._data[:, plot_cols[k]]])
            data = np.ma.compress_rows(data)  # Rempove masked data points.
            data.dtype = [(n, data.dtype) for n in names]
            output.append(data)
        return cols, output

    @property
    def _data(self):
        """Unpack the weakref to the data structure."""
        return self._data_ref()

    @property
    def _ax(self):
        """Return the current axis number."""
        if self._figure is None:
            return None
        return self._figure.gca()

    @property
    def ax(self):
        """Return the current axis number."""
        return self.axes.index(self.figure.gca())

    @ax.setter
    def ax(self, value):
        """Change the current axis number or instance."""
        if isinstance(value, int) and 0 <= value < len(self.axes):
            self.figure.sca(self.axes[value])
        elif isinstance(value, Axes) and value in self.axes:
            self.figure.sca(value)
        else:
            raise IndexError("Figure doesn't have enough axes or didn't have those particular axes")

    @property
    def axes(self):
        """Get the axes ovject from the current figure."""
        return self.figure.axes

    @property
    def figure(self):
        """Return the current matplotlib.ffigure."""
        if not isinstance(self._figure, Figure):
            self._figure, _ = self.template.new_figure()
        return self._figure

    @figure.setter
    def figure(self, val):
        """Set figure either with a figure or by calling plt.figure()."""
        if not isinstance(val, Figure):
            val, _ = self.template.new_figure(val)  # Try to treat this as a constructor to a new figure
        self._figure = val

    @property
    def template(self):
        """Return the current plot template."""
        return self._template

    @template.setter
    def template(self, value):
        """Set the current template."""
        if isinstance(value, type) and issubclass(value, DefaultPlotStyle):
            value = value()
        if isinstance(value, DefaultPlotStyle):
            self._template = value
        else:
            raise ValueError(f"Template is not of the right class:{type(value)}")
        self._template.apply()
