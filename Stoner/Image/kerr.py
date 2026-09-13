# -*- coding: utf-8 -*-
"""Kerr Image Processing Module.

Created on Fri Apr 21 17:29:08 2017
Derivatives of ImageFile and ImageStack specific to processing Kerr images.

@author: phyrct
"""

__all__ = ["KerrImageFile", "KerrStack", "MaskStack"]

import os
from shutil import which
from typing import Optional, Self, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..tools import make_Data
from ..tools.decorators import class_modifier, image_file_adaptor
from ..tools.typing import Args, Data, Kwargs
from . import kerrfuncs
from .core import ImageFile
from .numerical import numerical_image
from .stack import ImageStack

try:
    import pytesseract  # pylint: disable=unused-import

    _tesseractable = True
except ImportError:
    pytesseract = None
    _tesseractable = False

GRAY_RANGE = (0, 65535)  # 2^16
IM_SIZE = (512, 672)  # Standard Kerr image size
AN_IM_SIZE = (554, 672)  # Kerr image with annotation not cropped
pattern_file = os.path.join(os.path.dirname(__file__), "kerr_patterns.txt")




@class_modifier(kerrfuncs, adaptor=image_file_adaptor)
class KerrImageFile(ImageFile):
    """Own Kerr image storage and dispatch specialised numerical functions."""

    tesseractable = property(lambda self: _tesseractable and which("tesseract") is not None)
    priority = 16
    mime_type = ["image/png"]
    pattern = ["*.png"]


    def __init__(self: Self, *args: Args, **kwargs: Kwargs) -> None:
        """Construct a Kerr image with optional preparation steps.

        Keyword Arguments:
            reduce_metadata (bool):
                Keep recognised Kerr metadata fields when True. Default False.
            ocr_metadata (bool):
                Recognise the annotation strip before cropping. Default False.
            field_only (bool):
                Limit OCR to the applied field. Default False.
            asfloat (bool):
                Convert intensities to normalised floating point. Default False.
            crop_text (bool):
                Remove the standard annotation strip. Default False.
        """
        options = {name: kwargs.pop(name, False) for name in
                   ("reduce_metadata", "ocr_metadata", "field_only", "asfloat", "crop_text")}
        super().__init__(*args, **kwargs)
        if self.size:
            if options["reduce_metadata"]:
                self.reduce_metadata()
            if options["ocr_metadata"]:
                self.ocr_metadata(field_only=options["field_only"])
            if options["asfloat"]:
                self.asfloat()
            if options["crop_text"]:
                self.crop_text()


class KerrStackMixin:
    """A mixin for :py:class:`ImageStack` that adds some functionality particular to Kerr images.

    Attributes:
        fields(list):
            list of applied fields in stack. This is the most important metadata
            for things like hysteresis.
    """

    _defaults = {"type": KerrImageFile}

    @property
    def fields(self: Self) -> NDArray:
        """Produce an array of field values from the metadata."""
        if hasattr(self, "_field"):
            return self._field
        if "field" not in self.metadata:
            return np.arange(len(self))
        return np.array(self.metadata["field"])

    def crop_text(self: Self) -> Self:
        """Crop the bottom text area from a standard Kermit image across the complete stack.

        Returns:
            ImageStack:
                This stack, with each image cropped from (554, 672) to (512, 672)
                by removing the bottom 42 rows. An already cropped stack is returned unchanged.

        Raises:
            ValueError:
                If the image dimensions match neither the annotated nor the cropped shape.

        Notes:
            Cropping updates the stack storage and recorded image sizes in place.
            Pixel masks are retained over the cropped region, and the number of images is unchanged.
        """
        if self.shape[1:3] == IM_SIZE:
            return self
        if self.shape[1:3] != AN_IM_SIZE:
            raise ValueError(
                f"Need a full sized Kerr image to crop. Current size is {self.shape}"
            )  # check it's a normal image
        package = self.export_storage()
        package.dataset = package.dataset.isel(y=slice(0, IM_SIZE[0]))
        package.dataset.valid_height.data = np.minimum(package.dataset.valid_height.values, IM_SIZE[0])
        self._stack_owner.replace(package)
        return self

    def hysteresis(self: Self, mask=None) -> Data:
        """Make a hysteresis loop of the average intensity in the given images.

        Keyword Arguments:
            mask(ndarray or list):
                boolean array of same size as an image or imarray or list of
                masks for each image. If True then don't include that area in
                the intensity averaging.

        Returns:
            hyst(Data):
                'Field', 'Intensity', 2 column array
        """
        hyst = np.column_stack((self.fields, np.zeros(len(self))))
        for i, im in enumerate(self):
            if isinstance(im, ImageFile):
                im = im.image
            if isinstance(mask, np.ndarray) and mask.ndim == 2:
                hyst[i, 1] = np.average(im[np.invert(mask.astype(bool))])
            elif isinstance(mask, np.ndarray) and mask.ndim == 3:
                hyst[i, 1] = np.average(im[np.invert(mask[i, :, :].astype(bool))])
            elif isinstance(mask, (tuple, list)):
                hyst[i, 1] = np.average(im[np.invert(mask[i])])
            else:
                hyst[i, 1] = np.average(im)
        d = make_Data(hyst, setas="xy")
        d.column_headers = ["Field", "Intensity"]
        return d

    def index_to_field(self: Self, index_map: NDArray) -> np.ma.MaskedArray:
        """Convert an image of index values into an image of field values."""
        fieldvals = np.take(self.fields, index_map)
        return numerical_image(fieldvals)

    def denoise_thresh(
        self: Self, denoise_weight: float = 0.1, thresh: float = 0.5, invert: bool = False
    ) -> "MaskStack":
        """Apply denoise then threshold images.

        Returns:
            (ndarray) MaskStack:
                True for values greater than thresh, False otherwise
                else return True for values between thresh and 1
        """
        masks = self.clone
        masks.each.denoise(weight=denoise_weight)
        masks.each.threshold_minmax(threshmin=thresh, threshmax=np.max(masks.imarray))
        masks = MaskStack(masks)
        if invert:
            with masks.edit_numpy() as draft:
                for index, image in enumerate(masks):
                    height, width = image.shape
                    draft.data[index, :height, :width] = ~draft.data[index, :height, :width]
        return masks

    def find_threshold(self: Self, testim: Optional[Union[NDArray, int, str]] = None, mask: Optional[NDArray] = None):
        """Try to find the threshold value at which the image switches.

        Takes it as the median value of the testim. Masks values
        where the difference is less than tolerance in case part of the image is
        irrelevant.
        """
        if testim is None:
            testim = self[len(self) // 2]
        elif isinstance(testim, (int, str)):
            testim = self[testim]
        elif isinstance(testim, np.ndarray) and testim.shape == self[len(self) // 2].shape:
            pass
        else:
            raise ValueError("Cannot find testimage for thresholding.")
        if mask is None:
            med = testim.median()
        else:
            med = testim[~testim.mask]
        return med

    def stable_mask(self: Self, tolerance: float = 1e-2, comparison: Tuple[int, int] = (0, -1)) -> NDArray:
        """Produce a mask of areas of the image that are changing little over the stack.

        comparison is an optional tuple that gives the index of two images
        to compare, otherwise first and last used. tolerance is the difference
        tolerance
        """
        first, last = comparison
        mask = np.zeros(self[0].shape, dtype=bool)
        mask[abs(self[last] - self[first]) < tolerance] = True
        return mask

    def HcMap(  # pylint: disable=invalid-name
        self: Self,
        threshold: float = 0.5,
        correct_drift: bool = False,
        baseimage: int = 0,
        quiet: bool = True,
        saturation_end: bool = True,
        saturation_white: bool = True,
        extra_info: bool = False,
    ) -> NDArray:
        """Produce a map of the switching field at every pixel in the stack.

        It needs the stack to start saturated one way and end saturated the other way.

        Keyword Arguments:
            threshold(float):
                the threshold value for the intensity switching. This will need to
                be tuned for each stack
            correct_drift(bol):
                whether to correct drift on the image stack before proceeding
            baseimage(int or ImageFile):
                we use drift correction from the baseimage.
            saturation_end(bool):
                last image in stack is closest to saturation
            saturation_white(bool):
                bright pixels are saturated dark pixels are not yet switched
            quiet: bool
                choose whether to output status updates as print messages
            extra_info(bool):
                choose whether to return intermediate calculation steps as an extra dictionary
        Returns:
            (ImageFile): The map of field values for switching of each pixel in the stack
        """
        ks = self.clone
        if isinstance(baseimage, int):
            baseimage = self[baseimage].clone
        elif isinstance(baseimage, np.ndarray):
            baseimage = baseimage.view(np.ma.MaskedArray)
        if correct_drift:
            ks.apply_all("correct_drift", ref=baseimage, quiet=quiet)
            if not quiet:
                print("drift correct done")
        masks = self.denoise_thresh(denoise_weight=0.1, thresh=threshold, invert=not saturation_white)
        if not quiet:
            print("thresholding done")
        si, sp = masks.switch_index(saturation_end=saturation_end)
        Hcmap = ks.index_to_field(si)
        Hcmap[Hcmap == ks.fields[0]] = 0  # not switching does not give us a Hc value
        if extra_info:
            ei = {"switch_index": si, "switch_array": sp, "masks": masks}
            return Hcmap, ei
        return Hcmap

    def average_Hcmap(  # pylint: disable=invalid-name
        self: Self, weights: Optional[NDArray] = None, ignore_zeros: bool = False
    ) -> Self:
        """Get an array of average pixel values for the stack.

        Keyword Arguments:
            weights (array like):
                Weights to apply when averaging image.
            ignore_zeros (bool):
                Weight zero values in an image as 0 in the averaging.

        Returns:
            average(ImageFile):
                average values
        """
        if ignore_zeros:
            weights = self.clone
            weights.imarray = weights.imarray.astype(bool).astype(int)  # 1 if Hc isn't zero, zero otherwise
            condition = np.sum(weights, axis=0) == 0  # stop zero division error
            for m in range(self.shape[0]):
                weights[m] = np.select([condition, np.logical_not(condition)], [np.ones_like(weights[m]), weights[m]])
            # weights means we only account for non-zero values in average
        average = np.average(self.imarray, axis=0, weights=weights)
        return average.view(np.ma.MaskedArray)


class MaskStackMixin:
    """A Mixin for :py:class:`Stoner.Image.ImageStack` but made for stacks of boolean or binary images."""

    def __init__(self: Self, *args: Args, **kwargs: Kwargs):
        """Ensure the data is boolean."""
        super().__init__(*args, **kwargs)
        package = self.export_storage()
        package.dataset.intensity.data = package.dataset.intensity.values.astype(bool)
        package.fill_value = True
        for frame in package.frames:
            frame.fill_value = True
        self._stack_owner.replace(package)

    def switch_index(
        self: Self, saturation_end: bool = True, saturation_value: bool = True
    ) -> Tuple[NDArray, NDArray]:
        """Construct a map of switching points in a hysteresis stack.

        Given a stack of boolean masks representing a hystersis loop find the stack index of the saturation
        field for each pixel.

        Take the final mask as all switched (or the first mask if saturation_end
        is False). Work back through the masks taking the first time a pixel
        switches as its coercive field (ie the last time it switches before
        reaching saturation).
        Elements that start switched at the lowest measured field or never
        switch are given a zero index.

        At the moment it's set up to expect masks to be false when the sample is saturated
        at a high field

        Keyword Arguments:
            saturation_end(bool):
                True if the last image is closest to the fully saturated state.
                False if you want the first image
            saturation_value(bool):
                if True then a pixel value True means that switching has occurred
                (ie magnetic saturation would be all True)

        Returns:
            switch_ind: MxN ndarray of int
                index that each pixel switches at
            switch_progession: MxNx(P-1) ndarray of bool
                stack of masks showing when each pixel saturates

        """
        if not len(self):
            raise ValueError("Switching analysis requires at least one frame")
        values = self.to_numpy()
        if not saturation_end:
            values = values[::-1]
        if not saturation_value:
            values = ~values
        switch_ind = np.zeros(self.max_size, dtype=int)
        switch_prog = self.clone
        del switch_prog[-1]
        done = np.zeros(self.max_size, dtype=bool)
        if len(switch_prog):
            with switch_prog.edit_numpy() as draft:
                draft.data[:] = False
                for m in reversed(range(len(values) - 1)):
                    destination = m if saturation_end else len(values) - 2 - m
                    draft.data[destination] = done
                    changed = np.ma.filled((~values[m]) & values[m + 1], False) & ~done
                    switch_ind[changed] = m if saturation_end else len(values) - 1 - m
                    done |= changed
        switch_ind = numerical_image(np.ma.array(switch_ind, mask=np.ma.getmaskarray(values).any(axis=0)))
        return switch_ind, switch_prog


class KerrStack(KerrStackMixin, ImageStack):
    """Represent a stack of Kerr images."""


class MaskStack(MaskStackMixin, KerrStackMixin, ImageStack):
    """Represent a set of masks for Kerr images."""
