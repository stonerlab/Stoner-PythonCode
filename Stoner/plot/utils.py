# -*- coding: utf-8 -*-
"""A copy of mpltools.special.errorfill with a few hacks to make it place nice.

This software is licensed under the Modified BSD License.

Copyright (c) 2012, Tony S. Yu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name "mpltools" nor the names of its contributors may be used
      to endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
__all__ = ["errorfill", "extrema_from_error_input", "fill_between", "fill_between_x", "hsl2rgb", "joy_division"]
import warnings
from distutils.version import LooseVersion
from colorsys import hls_to_rgb

import numpy as np
import matplotlib.pyplot as plt

from Stoner.compat import mpl_version

__all__ = ["errorfill"]


def errorfill(
    x,
    y,
    yerr=None,
    xerr=None,
    color=None,
    ls=None,
    lw=None,
    alpha=1,
    alpha_fill=0.5,
    label="",
    label_fill="",
    ax=None,
    **kargs
):
    """Plot data with errors marked by a filled region.

    Args:
        x, y (arrays):
            Coordinates of data.
        yerr, xerr ([scalar | N, (N, 1), or (2, N) array]):
            Error for the input data:
                -   If scalar, then filled region spans `y +/- yerr` or `x +/- xerr`.
        color (Matplotlib color):
            Color of line and fill region.
        ls (Matplotlib line style):
            Style of the line
        lw (Matplotlib line width, float):
            Width of the line in points
        alpha (float):
            Opacity used for plotting.
        alpha_fill (float):
            Opacity of filled region. Note: the actual opacity of the fill is
            `alpha * alpha_fill`.
        label (str):
            Label for line.
        label_fill (str):
            Label for filled region.
        ax (Axis instance):
            The plot is drawn on axis `ax`. If `None` the current axis is used
    """
    ax = ax if ax is not None else plt.gca()

    alpha_fill *= alpha

    if color is None:
        if LooseVersion(mpl_version) < LooseVersion("1.5.0"):
            color = next(ax._get_lines.color_cycle)
        else:
            color = next(ax._get_lines.prop_cycler)["color"]
    if ls is None:
        ls = plt.rcParams["lines.linestyle"]
    if lw is None:
        lw = plt.rcParams["lines.linewidth"]
    ax.plot(x, y, color, linestyle=ls, linewidth=lw, alpha=alpha, label=label, **kargs)

    if yerr is not None and xerr is not None:
        msg = "Setting both `yerr` and `xerr` is not supported. Ignore `xerr`."
        warnings.warn(msg)

    kwargs_fill = dict(color=color, alpha=alpha_fill, label=label_fill)
    if yerr is not None:
        ymin, ymax = extrema_from_error_input(y, yerr)
        if x.size > 1:
            fill_between(x, ymax, ymin, ax=ax, **kwargs_fill)
    elif xerr is not None:
        xmin, xmax = extrema_from_error_input(x, xerr)
        fill_between_x(y, xmax, xmin, ax=ax, **kwargs_fill)


def extrema_from_error_input(z, zerr):
    """Work out where to draw limits."""
    if np.isscalar(zerr) or len(zerr) == len(z):
        zmin = z - zerr
        zmax = z + zerr
    elif len(zerr) == 2:
        zmin, zmax = z - zerr[0], z + zerr[1]
    return zmin, zmax


# Wrappers around `fill_between` and `fill_between_x` that create proxy artists
# so that filled regions show up correctly legends.


def fill_between(x, y1, y2=0, ax=None, **kwargs):
    """Draw shading around line."""
    ax = ax if ax is not None else plt.gca()
    ym = (y1 + y2) / 2.0
    yd = (y1 - y2) / 3.0
    # Remove any bad data points
    keep = np.logical_not(np.isnan(ym))
    x = x[keep]
    ym = ym[keep]
    yd = yd[keep]
    alpha = kwargs["alpha"]
    a = np.linspace(0.1, 0.9, 15)
    z = lambda x, s, y: s * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * s * y))
    for h in a:
        y = h / (np.sqrt(2 * np.pi) * yd)
        y1 = ym - (z(y, yd, y))
        y2 = ym + (z(y, yd, y))
        kwargs["alpha"] = alpha * h
        if x.size > 1:
            ax.fill_between(x, y1, y2, **kwargs)
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, **kwargs))


def fill_between_x(x, y1, y2=0, ax=None, **kwargs):
    """Draw shading around line."""
    ax = ax if ax is not None else plt.gca()
    ym = (y1 + y2) / 2.0
    yd = (y1 - y2) / 3.0
    alpha = kwargs["alpha"]
    a = np.linspace(0.1, 0.9, 15)
    z = lambda x, s, y: s * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * s * y))
    for h in a:
        y = h / (np.sqrt(2 * np.pi) * yd)
        y1 = ym - (z(y, yd, y))
        y2 = ym + (z(y, yd, y))
        kwargs["alpha"] = alpha * h
        ax.fill_betweenx(x, y1, y2, **kwargs)
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, **kwargs))


def hsl2rgb(hue, sat, lum):
    """Convert from hsl colourspace to rgb colour space with numpy arrays for speed.

    Args:
        h (array):
            Hue value
        s (array):
            Saturation value
        l (array):
            Luminence value

    Returns:
        2D array (Mx3) of unsigned 8bit integers
    """
    hue = np.atleast_1d(hue)
    sat = np.atleast_1d(sat)
    lum = np.atleast_1d(lum)

    if hue.shape != lum.shape or hue.shape != sat.shape:
        raise RuntimeError("Must have equal shaped arrays for h, s and l")

    rgb = np.zeros((hue.size, 3))
    hls = np.column_stack([hue, lum, sat])
    for i in range(hue.size):
        rgb[i, :] = np.array(hls_to_rgb(*hls[i]))
    return (255 * rgb).astype("u1")


def joy_division(x, y, z, **kargs):
    """Produce a classic black and white water fall plot.

    Parameters:
        x,y,z (1D arrays):
            x y and z co-ordinates. data should be arranged so that z(x,y=constant)

    Keyword Parameters:
        ax (matplotlib.Axes):
            Axes to use (defaults to current axes)
        y_shift (float):
            Shift in data for successive y values. Defaults to z-span / number of values in y
        bg_colour (matplotlib colour):
            Colour to use for background of the plot (default is "k" for black)
        colour (matplotlib colour):
            Colour of the lines on the plot (default 'white')
        axes_colour (matplotlib colour):
            Coulour of the frame, ticks, labels on the plot (default, same as *colour*)
        lw (float):
            Width of lines to use on the plot (default 2)
        legend_fmt (str):
            String to use to format the elgend text. Should include one place holder {}. Default "{}"

    Returns:
        None

    Constructurs a mono-chromatic waterfall plot in the style of the Joy Division album cover of Pulsar signals.
    """
    ax = kargs.pop("ax", plt.gca())
    y_shift = kargs.pop("y_shift", (z.max() - z.min()) / np.unique(y).size)
    bg_colour = kargs.pop("bg_color", "k")
    color = kargs.pop("color", kargs.pop("colour", "w"))
    axes_colour = kargs.pop("axes_color", color)
    lw = kargs.pop("lw", 2)
    legend_fmt = kargs.pop("legend_fmt", "{}")

    ax.figure.set_facecolor(bg_colour)
    ax.set_facecolor(bg_colour)

    yvals = np.unique(y)
    yvals = np.sort(yvals)

    data = np.column_stack((x, z, y))

    for ix, yval in enumerate(yvals):
        offset = y_shift * (len(yvals) - ix - 1)
        this_data = data[y == yval, :]

        ax.plot(
            this_data[:, 0], this_data[:, 1] + offset, color, lw=lw, zorder=(ix + 1) * 2, label=legend_fmt.format(yval)
        )
        ax.fill_between(
            this_data[:, 0], this_data[:, 1] + offset, offset, facecolor=bg_colour, lw=0, zorder=(ix + 1) * 2 - 1
        )

    ax.tick_params(color=axes_colour, labelcolor=axes_colour)
    for spine in ax.spines.values():
        spine.set_edgecolor(axes_colour)
    ax.set_title(ax.get_title(), color=axes_colour)
    ax.set_xlabel(ax.get_xlabel(), color=axes_colour)
    ax.set_ylabel(ax.get_ylabel(), color=axes_colour)

    # plt.legend(ncol=int(np.floor(np.sqrt(yvals.size))),fontsize="x-small")
    # plt.draw()

    # for t in ax.get_legend().get_texts():
    #     t.set_color(axes_colour)
    # plt.draw()
