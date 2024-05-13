# -*- coding: utf-8 -*-
"""Kerr Image Processing Module.

Created on Fri Apr 21 17:29:08 2017
Derivatives of ImageArray and ImageStack specific to processing Kerr images.

@author: phyrct
"""
__all__ = ["KerrArray", "KerrStack", "MaskStack"]

import os

import numpy as np

from ..Image import ImageArray, ImageStack, ImageFile
from ..tools import make_Data
from ..tools.decorators import class_modifier, image_file_adaptor
from . import kerrfuncs

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


@class_modifier(kerrfuncs)
class KerrArray(ImageArray):
    """A subclass for Kerr microscopy specific image functions."""

    # useful_keys are metadata keys that we'd usually like to keep from a
    # standard kerr output.

    def __init__(self, *args, **kargs):
        """Initialise KerrArray as a subclasses ImageArray.

        Extra keyword arguments accepted are given below.

        Keyword Arguments:
            reduce_metadata(bool):
                if True reduce the metadata to useful bits and do some processing on it
            asfloat(bool)
                if True convert the image to float values between 0 and 1 (necessary
                for some forms of processing)
            crop_text(bool):
                whether to crop the bottom text area from the image
            ocr_metadata(bool):
                whether to try to use optical character recognition to get the
                metadata from the image (necessary for images taken pre 06/2016
                and so far field from hysteresis images)
            field_only(bool):
                if ocr_metadata is true, get field only (bit faster)
        """
        kerrdefaults = {
            "ocr_metadata": False,
            "field_only": False,
            "reduce_metadata": True,
            "asfloat": True,
            "crop_text": True,
        }
        kerrdefaults.update(kargs)
        super().__init__(*args, **kargs)
        self._tesseractable = None
        if kerrdefaults["reduce_metadata"]:
            self.reduce_metadata()
        if kerrdefaults["ocr_metadata"]:
            self.ocr_metadata(field_only=kerrdefaults["field_only"])
        if kerrdefaults["asfloat"]:
            self.asfloat()
        if kerrdefaults["crop_text"]:
            self.crop_text()

    @property
    def tesseractable(self):
        """Do a test call to tesseract to see if it is there and cache the result."""
        return _tesseractable

    def save(self, filename=None, **kargs):
        """Stub method for a save function."""
        raise NotImplementedError(f"Save is not implemented in {self.__class__}")


@class_modifier(kerrfuncs, adaptor=image_file_adaptor)
class KerrImageFile(ImageFile):
    """Subclass of ImageFile that keeps the data as a KerrArray so that extra functions are available."""

    priority = 16
    mime_type = ["image/png"]
    pattern = ["*.png"]

    def __init__(self, *args, **kargs):
        """Ensure that the image is a KerrImage."""
        super().__init__(*args, **kargs)
        self._image = self.image.view(KerrArray)

    @ImageFile.image.getter
    def image(self):  # pylint disable=invalid-overridden-method
        """Access the image data."""
        return self._image.view(KerrArray)

    @ImageFile.image.setter
    def image(self, v):  # pylint: disable=function-redefined
        """Ensure stored image is always an ImageArray."""
        filename = self.filename
        v = KerrArray(v)
        # ensure setting image goes into the same memory block if from stack
        if (
            hasattr(self, "_fromstack")
            and self._fromstack
            and self._image.shape == v.shape
            and self._image.dtype == v.dtype
        ):
            self._image[:] = v
            self._image = self._image.view(KerrArray)
        else:
            self._image = KerrArray(v)
        self.filename = filename


class KerrStackMixin:
    """A mixin for :py:class:`ImageStack` that adds some functionality particular to Kerr images.

    Attributes:
        fields(list):
            list of applied fields in stack. This is the most important metadata
            for things like hysteresis.
    """

    _defaults = {"type": KerrImageFile}

    @property
    def fields(self):
        """Produce an array of field values from the metadata."""
        if not hasattr(self, "_field"):
            if "field" not in self.metadata:
                self._field = np.arange(len(self))
            else:
                self._field = np.array(self.metadata["field"])
        return self._field

    def crop_text(self):
        """Crop the bottom text area from a standard Kermit image across the complete stack.

        Returns:
        (ImageArray):
            cropped image
        """
        images = self.shape[0]
        if self.shape[1:3] == IM_SIZE:
            return self
        if self.shape[1:3] != AN_IM_SIZE:
            raise ValueError(
                f"Need a full sized Kerr image to crop. Current size is {self.shape}"
            )  # check it's a normal image
        self._sizes = np.column_stack(
            (np.ones(images, dtype=int) * IM_SIZE[0], np.ones(images, dtype=int) * IM_SIZE[1])
        )
        new_size = self.max_size + (images,)
        self._resize_stack(new_size)
        return self

    def hysteresis(self, mask=None):
        """Make a hysteresis loop of the average intensity in the given images.

        Keyword Argument:
            mask(ndarray or list):
                boolean array of same size as an image or imarray or list of
                masks for each image. If True then don't include that area in
                the intensity averaging.

        Returns
        -------
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

    def index_to_field(self, index_map):
        """Convert an image of index values into an image of field values."""
        fieldvals = np.take(self.fields, index_map)
        return ImageArray(fieldvals)

    def denoise_thresh(self, denoise_weight=0.1, thresh=0.5, invert=False):
        """Apply denoise then threshold images.

        Return:
            (ndarray) MaskStack:
                True for values greater than thresh, False otherwise
                else return True for values between thresh and 1
        """
        masks = self.clone
        masks.each.denoise(weight=denoise_weight)
        masks.each.threshold_minmax(threshmin=thresh, threshmax=np.max(masks.imarray))
        masks = MaskStack(masks)
        if invert:
            masks.stack = ~masks.stack  # pylint: disable=attribute-defined-outside-init
        return masks

    def find_threshold(self, testim=None, mask=None):
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

    def stable_mask(self, tolerance=1e-2, comparison=(0, -1)):
        """Produce a mask of areas of the image that are changing little over the stack.

        comparison is an optional tuple that gives the index of two images
        to compare, otherwise first and last used. tolerance is the difference
        tolerance
        """
        first, last = comparison
        mask = np.zeros(self[0].shape, dtype=bool)
        mask[abs(self[last] - self[first]) < tolerance] = True
        return mask

    def HcMap(
        self,
        threshold=0.5,
        correct_drift=False,
        baseimage=0,
        quiet=True,
        saturation_end=True,
        saturation_white=True,
        extra_info=False,
    ):
        """Produce a map of the switching field at every pixel in the stack.

        It needs the stack to start saturated one way and end saturated the other way.

        Keyword Arguments:
            threshold(float):
                the threshold value for the intensity switching. This will need to
                be tuned for each stack
            correct_drift(bol):
                whether to correct drift on the image stack before proceding
            baseimage(int or ImageArray):
                we use drift correction from the baseimage.
            saturation_end(bool):
                last image in stack is closest to saturation
            saturation_white(bool):
                bright pixels are saturated dark pixels are not yet switched
            quiet: bool
                choose wether to output status updates as print messages
            extra_info(bool):
                choose whether to return intermediate calculation steps as an extra dictionary
        Returns:
            (ImageArray): The map of field values for switching of each pixel in the stack
        """
        ks = self.clone
        if isinstance(baseimage, int):
            baseimage = self[baseimage].clone
        elif isinstance(baseimage, np.ndarray):
            baseimage = baseimage.view(ImageArray)
        if correct_drift:
            ks.apply_all("correct_drift", ref=baseimage, quiet=quiet)
            if not quiet:
                print("drift correct done")
        masks = self.denoise_thresh(denoise_weight=0.1, thresh=threshold, invert=not (saturation_white))
        if not quiet:
            print("thresholding done")
        si, sp = masks.switch_index(saturation_end=saturation_end)
        Hcmap = ks.index_to_field(si)
        Hcmap[Hcmap == ks.fields[0]] = 0  # not switching does not give us a Hc value
        if extra_info:
            ei = {"switch_index": si, "switch_array": sp, "masks": masks}
            return Hcmap, ei
        return Hcmap

    def average_Hcmap(self, weights=None, ignore_zeros=False):
        """Get an array of average pixel values for the stack.

        Return average of pixel values in the stack.

        Keyword Arguments:
            ignore zeros(bool):
                Weight zero values in an image as 0 in the averaging.

        Returns:
            average(ImageArray):
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
        return average.view(ImageArray)


class MaskStackMixin:
    """A Mixin for :py:class:`Stoner.Image.ImageStack` but made for stacks of boolean or binary images."""

    def __init__(self, *args, **kargs):
        """Ensure the data is boolean."""
        super().__init__(*args, **kargs)
        self._stack = self._stack.astype(bool)

    def switch_index(self, saturation_end=True, saturation_value=True):
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
        ms = self.clone
        if not saturation_end:
            ms = ms.reverse()
        # arr1 = ms[0].astype(float) #find out whether True is at begin or end
        # arr2 = ms[-1].astype(float)
        # if np.average(arr1)>np.average(arr2): #OK so it's bright at the start
        if not saturation_value:
            self.imarray = np.invert(ms.imarray)  # Now it's bright (True) at end
        switch_ind = np.zeros(ms[0].shape, dtype=int)
        switch_prog = self.clone
        switch_prog.imarray = np.zeros(self.shape, dtype=bool)
        del switch_prog[-1]
        for m in reversed(range(len(ms) - 1)):  # go from saturation backwards
            already_done = np.copy(switch_ind).astype(dtype=bool)  # only change switch_ind if it hasn't already
            condition = np.logical_and(not ms[m], ms[m + 1])
            condition = np.logical_and(condition, np.invert(already_done))
            condition = [condition, np.logical_not(condition)]
            choice = [np.ones(switch_ind.shape) * m, switch_ind]  # index or leave as is
            switch_ind = np.select(condition, choice)
            switch_prog[m] = already_done
        if not saturation_end:
            switch_ind = -switch_ind + len(self) - 1  # should check this!
            switch_prog.reverse()
        switch_ind = ImageArray(switch_ind.astype(int))
        return switch_ind, switch_prog


class KerrStack(KerrStackMixin, ImageStack):
    """Represent a stack of Kerr images."""


class MaskStack(MaskStackMixin, KerrStackMixin, ImageStack):
    """Represent a set of masks for Kerr images."""
