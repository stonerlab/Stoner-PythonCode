#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Defines the magic attriobutes that :py:class:`Stoner.Image.ImageFolder` uses."""

__all__ = ["DrawProxy", "MaskProxy"]

from functools import wraps
from io import BytesIO as StreamIO

import numpy as np
from skimage import draw
import matplotlib.pyplot as plt

from Stoner.tools import fix_signature


class DrawProxy:

    """Provides a wrapper around scikit-image.draw to allow easy drawing of objects onto images."""

    def __init__(self, *args, **kargs):  # pylint: disable=unused-argument
        """Grab the parent image from the constructor."""
        self.img = args[0]

    def __getattr__(self, name):
        """Retiurn a callable function that will carry out the draw operation requested."""
        func = getattr(draw, name)

        @wraps(func)
        def _proxy(*args, **kargs):
            value = kargs.pop("value", np.ones(1, dtype=self.img.dtype)[0])
            coords = func(*args, **kargs)
            self.img[coords] = value
            return self.img

        return fix_signature(_proxy, func)

    def __dir__(self):
        """Pass through to the dir of skimage.draw."""
        own = set(dir(super()))
        d = set(dir(draw))
        return list(own | d)

    def annulus(self, r, c, radius1, radius2, shape=None, value=1.0):
        """Use a combination of two circles to draw and annulus.

        Args:
            r,c (float): Centre co-ordinates
            radius1,radius2 (float): Inner and outer radius.

        Keyword Arguments:
            shape (2-tuple, None): Confine the co-ordinates to staywith shape
            value (float): value to draw with
        Returns:
            A copy of the image with the annulus drawn on it.

        Notes:
            If radius2<radius1 then the sense of the whole shape is inverted
            so that the annulus is left clear and the filed is filled.
        """
        if shape is None:
            shape = self.img.shape
        invert = radius2 < radius1
        if invert:
            buf = np.ones(shape)
            fill = 0.0
            bg = 1.0
        else:
            buf = np.zeros(shape)
            fill = 1.0
            bg = 0.0
        radius1, radius2 = min(radius1, radius2), max(radius1, radius2)
        rr, cc = draw.circle(r, c, radius2, shape=shape)
        buf[rr, cc] = fill
        rr, cc = draw.circle(r, c, radius1, shape=shape)
        buf[rr, cc] = bg
        self.img[:, :] = (self.img * buf + value * (1.0 - buf)).astype(self.img.dtype)
        return self.img

    def rectangle(self, r, c, w, h, angle=0.0, shape=None, value=1.0):
        """Draw a rectangle on an image.

        Args:
            r,c (float): Centre co-ordinates
            w,h (float): Lengths of the two sides of the rectangle

        Keyword Arguments:
            angle (float): Angle to rotate the rectangle about
            shape (2-tuple or None): Confine the co-ordinates to this shape.
            value (float): The value to draw with.

        Returns:
            A copy of the current image with a rectangle drawn on.
        """
        if shape is None:
            shape = self.img.shape

        x1 = r - h / 2
        x2 = r + h / 2
        y1 = c - w / 2
        y2 = c + w / 2
        co_ords = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        if angle != 0:
            centre = np.array([r, c])
            cos, sin, m = np.cos, np.sin, np.matmul
            r = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
            co_ords = np.array([centre + m(r, xy - centre) for xy in co_ords])
        rr, cc = draw.polygon(co_ords[:, 0], co_ords[:, 1], shape=shape)
        self.img[rr, cc] = value
        return self.img

    def square(self, r, c, w, angle=0.0, shape=None, value=1.0):
        """Draw a square on an image.

        Args:
            r,c (float): Centre co-ordinates
            w (float): Length of the side of the square

        Keyword Arguments:
            angle (float): Angle to rotate the rectangle about
            shape (2-tuple or None): Confine the co-ordinates to this shape.
            value (float): The value to draw with.

        Returns:
            A copy of the current image with a rectangle drawn on.
        """
        return self.rectangle(r, c, w, w, angle=angle, shape=shape, value=value)


class MaskProxy:

    """Provides a wrapper to support manipulating the image mask easily."""

    @property
    def _IA(self):
        """Get the underliying image data."""
        return self._IF.image

    @property
    def _mask(self):
        """Get the mask for the underlying image."""
        self._IA.mask = np.ma.getmaskarray(self._IA)
        return self._IA.mask

    @property
    def colour(self):
        """Get the colour of the mask."""
        return self._IA._mask_color

    @colour.setter
    def colour(self, value):
        """Set the colour of the mask."""
        self._IA._mask_color = value

    @property
    def data(self):
        """Get the underlying data as an array - compatibility accessor."""
        return self[:]

    @property
    def image(self):
        """Get the underlying data as an array - compatibility accessor."""
        return self[:]

    @property
    def draw(self):
        """Access the draw proxy opbject."""
        return DrawProxy(self._mask)

    def __init__(self, *args, **kargs):
        """Keep track of the underlying objects."""
        self._IF = args[0]

    def __getitem__(self, index):
        """Proxy through to mask index."""
        return self._mask.__getitem__(index)

    def __setitem__(self, index, value):
        """Proxy through to underlying mask."""
        self._IA.mask.__setitem__(index, value)

    def __delitem__(self, index):
        """Proxy through to underyling mask."""
        self._IA.mask.__delitem__(index)

    def __getattr__(self, name):
        """Check name against self._IA._funcs and constructs a method to edit the mask as an image."""
        if hasattr(self._IA.mask, name):
            return getattr(self._IA.mask, name)
        if f".*__{name}$" not in self._IA._funcs:
            raise AttributeError(f"{name} not a callable mask method.")
        func = self._IA._funcs[f".*__{name}$"]

        @wraps(func)
        def _proxy_call(*args, **kargs):
            retval = func(self._mask.astype(float) * 1000, *args, **kargs)
            if isinstance(retval, np.ndarray) and retval.shape == self._IA.shape:
                retval = (retval + retval.min()) / (retval.max() - retval.min())
                self._IA.mask = retval > 0.5
            return retval

        _proxy_call.__doc__ = func.__doc__
        _proxy_call.__name__ = func.__name__
        return fix_signature(_proxy_call, func)

    def __repr__(self):
        """Make a textual representation of the image."""
        output = ""
        f = np.array(["."] * self._mask.shape[1])
        t = np.array(["X"] * self._mask.shape[1])
        for ix in self._mask:
            row = np.where(ix, t, f)
            output += "".join(row) + "\n"
        return output

    def __str__(self):
        """Make a textual representation of the image."""
        return repr(self._mask)

    def __invert__(self):
        """Invert the mask."""
        return np.logical_not(self._mask)

    def __neg__(self):
        """Invert the mask."""
        return np.logical_not(self._mask)

    def _repr_png_(self):
        """Provide a display function for iPython/Jupyter."""
        fig = self._IA._funcs[".*imshow"](self._mask.astype(int))
        data = StreamIO()
        fig.savefig(data, format="png")
        plt.close(fig)
        data.seek(0)
        ret = data.read()
        data.close()
        return ret

    def clear(self):
        """Clear a mask."""
        self._IA.mask = np.zeros_like(self._IA)

    def invert(self):
        """Invert the mask."""
        self._IA.mask = ~self._IA.mask

    def threshold(self, thresh=None):
        """Mask based on a threshold.

        Keyword Arguments:
            thresh (float):
                Threshold to apply to the current image - default is to calculate using threshold_otsu
        """
        if thresh is None:
            thresh = self._IA.threshold_otsu()
        self._IA.mask = self._IA > thresh
