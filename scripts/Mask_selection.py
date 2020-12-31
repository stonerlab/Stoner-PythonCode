# -*- coding: utf-8 -*-
# pylint: disable=no-member, invalid-name
"""Demonstrates the interactive mask selection code."""

from Stoner import __homepath__
from Stoner.HDF5 import STXMImage

# Load a single helicity STXM image
img = STXMImage(
    __homepath__ / ".." / "sample-data" / "Sample_Image_2017-10-15_100.hdf5"
)

# USe the interactive mask selection function
img.mask.select()

# Normalise the image based on the masked region
img.normalise(limits=(0.01, 0.99), scale_masked=True)
img.imshow()

# Quantize the image colours based on a threshold calculated from the image.
img.clone.quantize(output=[-1, 1], levels=[img.threshold_otsu()]).imshow()
