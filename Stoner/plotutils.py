# -*- coding: utf-8 -*-
"""
Created on Mon Jun 02 22:37:42 2014

A copy of mpltools.special.errorfill with a few hacks to make it place nice

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

import warnings

import numpy as np
import matplotlib.pyplot as plt

__all__ = ['errorfill']


def errorfill(x, y,
              yerr=None,
              xerr=None,
              color=None,
              ls=None,
              lw=None,
              alpha=1,
              alpha_fill=0.7,
              label='',
              label_fill='',
              ax=None, **kargs):
    """Plot data with errors marked by a filled region.

    Parameters
    ----------
    x, y : arrays
        Coordinates of data.
    yerr, xerr: [scalar | N, (N, 1), or (2, N) array]
        Error for the input data.
        - If scalar, then filled region spans `y +/- yerr` or `x +/- xerr`.
    color : Matplotlib color
        Color of line and fill region.
    ls : Matplotlib line style
        Style of the line
    lw : Matplotlib line width, float value in points
        Width of the line
    alpha : float
        Opacity used for plotting.
    alpha_fill : float
        Opacity of filled region. Note: the actual opacity of the fill is
        `alpha * alpha_fill`.
    label : str
        Label for line.
    label_fill : str
        Label for filled region.
    ax : Axis instance
        The plot is drawn on axis `ax`. If `None` the current axis is used
    """
    ax = ax if ax is not None else plt.gca()

    alpha_fill *= alpha

    if color is None:
        color = next(ax._get_lines.color_cycle)
    if ls is None:
        ls = plt.rcParams['lines.linestyle']
    if lw is None:
        lw = plt.rcParams['lines.linewidth']
    ax.plot(x, y, color, linestyle=ls, linewidth=lw, alpha=alpha, label=label, **kargs)

    if yerr is not None and xerr is not None:
        msg = "Setting both `yerr` and `xerr` is not supported. Ignore `xerr`."
        warnings.warn(msg)

    kwargs_fill = dict(color=color, alpha=alpha_fill, label=label_fill)
    if yerr is not None:
        ymin, ymax = extrema_from_error_input(y, yerr)
        fill_between(x, ymax, ymin, ax=ax, **kwargs_fill)
    elif xerr is not None:
        xmin, xmax = extrema_from_error_input(x, xerr)
        fill_between_x(y, xmax, xmin, ax=ax, **kwargs_fill)


def extrema_from_error_input(z, zerr):
    if np.isscalar(zerr) or len(zerr) == len(z):
        zmin = z - zerr
        zmax = z + zerr
    elif len(zerr) == 2:
        zmin, zmax = z - zerr[0], z + zerr[1]
    return zmin, zmax

# Wrappers around `fill_between` and `fill_between_x` that create proxy artists
# so that filled regions show up correctly legends.


def fill_between(x, y1, y2=0, ax=None, **kwargs):
    ax = ax if ax is not None else plt.gca()
    ym = (y1 + y2) / 2.0
    yd = (y1 - y2) / 3.0
    alpha = kwargs["alpha"]
    a = np.linspace(0.1, 0.9, 15)
    for h in a:
        y = h / (np.sqrt(2 * np.pi) * yd)
        z = lambda x, s: s * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * s * y))
        y1 = ym - (z(y, yd))
        y2 = ym + (z(y, yd))
        kwargs["alpha"] = alpha * h
        ax.fill_between(x, y1, y2, **kwargs)
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, **kwargs))


def fill_between_x(x, y1, y2=0, ax=None, **kwargs):
    ax = ax if ax is not None else plt.gca()
    ym = (y1 + y2) / 2.0
    yd = (y1 - y2) / 3.0
    alpha = kwargs["alpha"]
    a = np.linspace(0.1, 0.9, 15)
    a = np.linspace(0.1, 0.9, 15)
    for h in a:
        y = h / (np.sqrt(2 * np.pi) * yd)
        z = lambda x, s: s * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * s * y))
        y1 = ym - (z(y, yd))
        y2 = ym + (z(y, yd))
        kwargs["alpha"] = alpha * h
        ax.fill_betweenx(x, y1, y2, **kwargs)
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, **kwargs))


if __name__ == '__main__':
    x = np.linspace(0, 2 * np.pi)
    y_sin = np.sin(x)
    y_cos = np.cos(x)

    errorfill(x, y_sin, 0.2)
    errorfill(x, y_cos, 0.2)

    plt.show()
