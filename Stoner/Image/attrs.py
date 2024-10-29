#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Defines the magic attriobutes that :py:class:`Stoner.Image.ImageFolder` uses."""

__all__ = ["DrawProxy", "MaskProxy"]

from functools import wraps
from io import BytesIO as StreamIO

import numpy as np
from skimage import draw
import matplotlib.pyplot as plt

from ..tools import fix_signature
from ..tools.decorators import class_modifier
from .widgets import ShapeSelect
from .imagefuncs import imshow


def _draw_apaptor(func):
    """Adapt methods for DrawProxy class to bind :py:mod:`skimage.draw` functions."""

    @wraps(func)
    def _proxy(self, *args, **kargs):
        value = kargs.pop("value", np.ones(1, dtype=self._img.dtype)[0])
        coords = func(*args, **kargs)
        if len(coords) == 3:
            rr, cc, vv = coords
            if len(rr) == len(cc):
                coords = rr, cc
                value = value * vv
        if len(coords) == 2 and isinstance(coords[0], np.ndarray) and coords[0].ndim == 3:
            im = type(self._img)(np.zeros(self._img.shape, dtype="uint32"))
            im += coords[0][:, :, 0]
            im += coords[0][:, :, 1].astype("uint32") * 256
            im += coords[0][:, :, 2].astype("uint32") * 256**2
            im[im == 16777215] = 0
            im.convert(self._img.dtype)
            self._img[im != 0] = im[im != 0]
            return self._parent

        self._img[coords] = value
        return self._parent

    return fix_signature(_proxy, func)


@class_modifier(draw, adaptor=_draw_apaptor, RTD_restrictions=False, no_long_names=True)
class DrawProxy:
    """Provides a wrapper around :py:mod:`skimage.draw` to allow easy drawing of objects onto images.

    This class allows access the user to draw simply shapes on an image (or its mask) by specifying the desired shape
    and geometry (centre, length/width etc). Mostly this implemented by pass throughs to the :py:mod:`skimage.draw`
    module, but methods are provided for an annulus, rectangle (and square) and rectangle-perimeter meothdds- the
    latter offering rotation about the centre point in contrast to the :py:mod:`skimage.draw` equivalents.

    No state data is stored with this class so the attribute does not need to be serialised when the parent ImageFile
    is saved.
    """

    def __init__(self, *args, **kargs):  # pylint: disable=unused-argument
        """Grab the parent image from the constructor."""
        self._img = args[0]
        self._parent = args[1]

    def annulus(self, r, c, radius1, radius2, shape=None, value=1.0):
        """Use a combination of two circles to draw and annulus.

        Args:
            r,c (float): Centre coordinates
            radius1,radius2 (float): Inner and outer radius.

        Keyword Arguments:
            shape (2-tuple, None): Confine the coordinates to staywith shape
            value (float): value to draw with
        Returns:
            A copy of the image with the annulus drawn on it.

        Notes:
            If radius2<radius1 then the sense of the whole shape is inverted
            so that the annulus is left clear and the filed is filled.
        """
        if shape is None:
            shape = self._img.shape
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
        rr, cc = draw.disk((r, c), radius2, shape=shape)
        buf[rr, cc] = fill
        rr, cc = draw.disk((r, c), radius1, shape=shape)
        buf[rr, cc] = bg
        self._img[:, :] = (self._img * buf + value * (1.0 - buf)).astype(self._img.dtype)
        return self._parent

    if "circle" not in dir(draw):

        def circle(self, r, c, radius, shape=None, value=1.0):
            """ "Generate coordinates of pixels within circle.

            Args:
                r,c (int): coordinates of the centre of the circle to be drawn.
                radius (float): Radius of the circle

            Keyword arguments:
                shape (tuple): Image shape as a tuple of size 2. Determines the maximum extent of output
                    pixel coordinates. This is useful for disks that exceed the image size. If None, the full
                    extent of the disk is used. The shape might result in negative coordinates and wraparound
                    behaviour.
                value (float): pixel value to write with.


            Notes:
                This is actually just a proxy for disk
            """
            return self.disk((r, c), radius, shape=shape, value=value)  # pylint: disable=no-member

    def rectangle(self, r, c, w, h, angle=0.0, shape=None, value=1.0):
        """Draw a rectangle on an image.

        Args:
            r,c (float): Centre coordinates
            w,h (float): Lengths of the two sides of the rectangle

        Keyword Arguments:
            angle (float): Angle to rotate the rectangle about
            shape (2-tuple or None): Confine the coordinates to this shape.
            value (float): The value to draw with.

        Returns:
            A copy of the current image with a rectangle drawn on.
        """
        if shape is None:
            shape = self._img.shape

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
        self._img[rr, cc] = value
        return self._parent

    def rectangle_perimeter(self, r, c, w, h, angle=0.0, shape=None, value=1.0):
        """Draw the perimter of a rectangle on an image.

        Args:
            r,c (float): Centre coordinates
            w,h (float): Lengths of the two sides of the rectangle

        Keyword Arguments:
            angle (float): Angle to rotate the rectangle about
            shape (2-tuple or None): Confine the coordinates to this shape.
            value (float): The value to draw with.

        Returns:
            A copy of the current image with a rectangle drawn on.
        """
        if shape is None:
            shape = self._img.shape

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
        rr, cc = draw.polygon_perimeter(co_ords[:, 0], co_ords[:, 1], shape=shape)
        self._img[rr, cc] = value
        return self._parent

    def square(self, r, c, w, angle=0.0, shape=None, value=1.0):
        """Draw a square on an image.

        Args:
            r,c (float): Centre coordinates
            w (float): Length of the side of the square

        Keyword Arguments:
            angle (float): Angle to rotate the rectangle about
            shape (2-tuple or None): Confine the coordinates to this shape.
            value (float): The value to draw with.

        Returns:
            A copy of the current image with a rectangle drawn on.
        """
        return self.rectangle(r, c, w, w, angle=angle, shape=shape, value=value)


class MaskProxy:
    """Provides a wrapper to support manipulating the image mask easily.

    The actual mask of a :py:class:`Stonmer.ImageFile` is held by the mask attribute of the underlying
    :py:class:`numpy.ma.MaskedArray`, but this class implements an attribute for an ImageFile that not only
    provides a means to index into the mask data, but supported other operations too.

    Attributes:
        color (matplotlib colour):
            This defines the colour of the mask that is used when showing a masked image in a window. It can be a
            named colour - such as *red*, *blue* etc. or a tuple of 3 or 4 numbers between 0 and 1 - which define an
            RGB(Alpha) colour. Note that the Alpha channel is actually an opacity channel = so 1.0 is solid and 0 is
            transparent.
        data (numpy array of bool):
            This accesses the actual boolean masked array if it is necessary to get at the full array. It is equivalent
            to .mask[:]
        image (numpoy array of bool):
            This is a synonym for :py:attr:`MaskProxy.data`.
        draw (:py:class:`DrawProxy`):
            This allows the mask to drawn on like the image. This is particularly useful as it provides a convenient
            programmatic way to define regions of interest in the mask with simply geometric shapes.

    Indexing of the MaskProxy simply passes through to the underlying mask data - thus getting, setting and deleting
    element directly changes the mask data.

    The string representation of the mask is an ascii art version of the mask where . is unmasked and X is masked.

    Conversion to a boolean is equaivalent to testing whether **any** elements of the mask are True.

    The mask also supports the invert and negate operators which both return the inverse of the mask (but do not
    change the mask itself - unlike :py:meth:`MaskProxy.invert`).

    For rich displays, the class also supports a png representation which is simply a black and white version of the
    mask with black pixels being masked elements and white unmasked elements.
    """

    @property
    def _imagearray(self):
        """Get the underliying image data."""
        return self._imagefolder.image

    @property
    def _mask(self):
        """Get the mask for the underlying image."""
        self._imagearray.mask = np.ma.getmaskarray(self._imagearray)
        return self._imagearray.mask

    @property
    def colour(self):
        """Get the colour of the mask."""
        return getattr(self._imagearray, "_mask_color", None)

    @colour.setter
    def colour(self, value):
        """Set the colour of the mask."""
        self._imagearray._mask_color = value

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
        """Access the draw proxy object."""
        return DrawProxy(self._mask, self._imagefolder)

    def __init__(self, *args):
        """Keep track of the underlying objects."""
        self._imagefolder = args[0]

    def __getitem__(self, index):
        """Proxy through to mask index."""
        return self._mask.__getitem__(index)

    def __setitem__(self, index, value):
        """Proxy through to underlying mask."""
        self._imagearray.mask.__setitem__(index, value)

    def __delitem__(self, index):
        """Proxy through to underlying mask."""
        self._imagearray.mask.__delitem__(index)

    def __getattr__(self, name):
        """Check name against self._imagearray._funcs and constructs a method to edit the mask as an image."""
        if hasattr(self._imagearray.mask, name):
            return getattr(self._imagearray.mask, name)
        func = getattr(type(self._imagearray), name, None)
        if func is None:
            raise AttributeError(f"{name} not a callable mask method.")

        @wraps(func)
        def _proxy_call(*args, **kargs):
            retval = func(self._mask.astype(float).view(type(self._imagearray)) * 1000, *args, **kargs)
            if isinstance(retval, np.ndarray) and retval.shape == self._imagearray.shape:
                retval.normalise()
                self._imagearray.mask = retval > 0
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

    def __bool__(self):
        """Check whether any of the mask elements are set."""
        return np.any(self._mask)

    def __invert__(self):
        """Invert the mask."""
        return np.logical_not(self._mask)

    def __neg__(self):
        """Invert the mask."""
        return np.logical_not(self._mask)

    def _repr_png_(self):
        """Provide a display function for iPython/Jupyter."""
        fig = imshow(self._mask.astype(int))
        data = StreamIO()
        fig.savefig(data, format="png")
        plt.close(fig)
        data.seek(0)
        ret = data.read()
        data.close()
        return ret

    def clear(self):
        """Clear a mask."""
        self._imagearray.mask = np.zeros_like(self._imagearray)

    def invert(self):
        """Invert the mask."""
        self._imagearray.mask = ~self._imagearray.mask

    def select(self, **kargs):
        """Interactive selection mode.

        This method allows the user to interactively choose a mask region on the image. It will require the
        Matplotlib backen to be set to Qt or other non-inline backend that supports a user vent loop.

        The image is displayed in the window and athe user can interact with it with the mouse and keyboard.

            -   left-clicking the mouse sets a new vertex
            -   right-clicking the mouse removes the last set vertex
            -   pressing "i" inverts the mask (i.e. controls whether the shape the user is drawing is masked or clear)
            -   pressing "p" sets polygon mode (the default) - each vertex is then the corener of a polygon. The
                polygon
                vertices are defined in order going around the shape.
            -   pressing "r" sets rectangular mode. The first vertex defined is one corner. With only two vertices the
                rectangle is not-rotated and the two vertices define opposite corners. If three vertices are defined
                then the first two form one side and then third vertex controls the extent of the rectangle in the
                direction perpendicular to the side defined.
            -   pressing "c" sets circle/ellipse mode. The first vertex defines one point on the circumference of the
                circle, the next point will define a point on the opposite side of the circumference. If three
                vertices are defined then a circle that passes through all three of them is used. Defining 4
                vertices causes the mode to attempt to find the non-rotated ellipse through the points and further
                vertices allows the ellipse to be rotated.

        This method directly sets the mask and then returns a copy of the parent :py:class:`Stoner.ImageFile`.
        """
        selection = kargs.get("_selection", [])
        if len(selection) == 0:
            selector = ShapeSelect()
            self._imagearray.mask = selector(self._imagearray)
            selection.append(self._imagearray.mask)
        elif len(selection) == 1 and isinstance(selection[0], np.ndarray) and selection[0].dtype.kind == "b":
            self._imagearray.mask = selection[0]
        else:
            raise ValueError("Unknown value for private keyword _selection")
        return self._imagefolder

    def threshold(self, thresh=None):
        """Mask based on a threshold.

        Keyword Arguments:
            thresh (float):
                Threshold to apply to the current image - default is to calculate using threshold_otsu
        """
        if thresh is None:
            thresh = self._imagearray.threshold_otsu()
        self._imagearray.mask = self._imagearray > thresh
