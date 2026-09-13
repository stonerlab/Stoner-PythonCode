# -*- coding: utf-8 -*-
"""Provide xarray-owned image stacks and the retained specialised array backend."""

__all__ = ["ImageStack"]
import warnings

import numpy as np

from ..folders.core import BaseFolder
from ..folders.mixins import DiskBasedFolderMixin
from .numerical import numerical_image
from .folders import ImageFolderMixin
from .stack_bridge import OwnedStackMixin

IM_SIZE = (512, 672)  # Standard Kerr image size
AN_IM_SIZE = (554, 672)  # Kerr image with annotation not cropped


def _load_numerical_image(f, **kwargs):
    """Create and image array."""
    kwargs.pop("Img_num", None)  # REemove img_num if it exists
    return numerical_image(f, **kwargs)




class StackAnalysisMixin:
    def asfloat(self, normalise=True, clip=False, clip_negative=False, **kwargs):
        """Convert stack to floating point type.

        Keyword Arguments:
            normalise(bool):
                normalise the image to the max value of current int type
            clip(bool):
                clip resulting range to values between -1 and 1
            clip_negative(bool):
                clip range further to 0,1

        Notes:
            Analogous behaviour to ImageFile.asfloat()

            If currently an int type and normalise then floats will be normalised
            to the maximum allowed value of the int type.
            If currently a float type then no change occurs.
            If clip_negative then clip values outside the range 0,1

        """
        if self.imarray.dtype.kind == "f":
            pass
        else:
            self.convert(dtype=np.float64, normalise=normalise)
        if "clip_neg" in kwargs:
            warnings.warn(
                "clip_neg argument renamed to clip_negative in ImageStack. This will cause an error in future"
                + "versions of the Stoner Package."
            )
            clip_negative = kwargs.pop("clip_neg")
        if clip or clip_negative:
            self.each.clip_intensity(clip_negative=clip_negative)
        return self
    def dtype_limits(self, clip_negative=True):
        """Return intensity limits, i.e. (min, max) tuple, of imarray dtype.

        Keyword Arguments:
            clip_negative(bool):
                If True, clip the negative range (i.e. return 0 for min intensity)
                even if the image dtype allows negative values.

        Returns:
            (imin,imax) (tuple):
                Lower and upper intensity limits.
        """
        ret = self[0].dtype_limits
        if clip_negative:
            ret = [max(0, x) for x in ret]
        return ret
    def correct_drifts(self, refindex, threshold=0.005, upsample_factor=50, box=None):
        """Dispatch legacy drift correction across the stack.

        Args:
            refindex (int or str):
                Index or name of the image used as the zero-drift reference.

        Keyword Arguments:
            threshold (float):
                Legacy feature-detection threshold forwarded to correct_drift. Defaults to 0.005.
            upsample_factor (int):
                Legacy registration upsampling factor forwarded to correct_drift. Defaults to 50.
            box (tuple or None):
                Region forwarded to correct_drift as (xmin, xmax, ymin, ymax).
                Defaults to None. Interpretation is delegated to the alignment implementation.

        Returns:
            None:
                The result of applying the correction is not returned for chaining.

        Notes:
            This compatibility method emits a deprecation warning and calls the deprecated
            apply_all dispatcher with the selected reference image and the supplied keywords.
            It retains legacy argument forwarding; prefer the stack's align method for new code.

        """
        warnings.warn("correct_drift is a deprecated method for an image stack - consider using align.")
        ref = self[refindex]
        self.apply_all("correct_drift", ref, threshold=threshold, upsample_factor=upsample_factor, box=box)
    def crop_stack(self, box):
        """Crop the imagestack to a box.

        Args:
            box(array or list of type int):
                [xmin,xmax,ymin,ymax]

        Returns:
            (ImageStack):
                cropped images
        """
        warnings.warn("crop_stack is deprecated - sam effect can be achieved with crop(box)")
        self.each.crop(box)
    def show(self):
        """Pass through to :py:meth:`Stoner.Image.ImageFolder.view`."""
        warnings.warn("show() is deprecated in favour of ImageFolder.view()")
        return self.view()

    """Add some analysis capability to ImageStack.

    These functions may override :py:class:`Stoner,Image.ImageFile` functions but do them efficiently for a numpy
    stack of images.
    """



class ImageStack(OwnedStackMixin, StackAnalysisMixin, ImageFolderMixin, DiskBasedFolderMixin, BaseFolder):
    """An alternative implementation of an image stack based on BaseFolder."""
