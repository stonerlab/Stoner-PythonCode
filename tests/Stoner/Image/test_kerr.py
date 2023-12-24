# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:09:04 2016

@author: phyrct
"""

from Stoner.Image.kerr import KerrArray, KerrImageFile, KerrStack
from Stoner import Data, __home__
import numpy as np
import pytest
import os
import matplotlib.pyplot as plt


import Stoner

Stoner.Options.multiprocessing = False

# data arrays for testing - some useful small images for tests

testdir = os.path.join(os.path.dirname(__file__), "kerr_testdata")
testdir2 = os.path.join(os.path.dirname(__file__), "coretestdata", "testims")
sample_data_dir = os.path.join(__home__, "../sample-data")


def shares_memory(arr1, arr2):
    """Check if two numpy arrays share memory"""
    # ret = arr1.base is arr2 or arr2.base is arr1
    ret = np.may_share_memory(arr1, arr2)
    return ret


selfimage = KerrArray(os.path.join(testdir, "kermit3.png"), ocr_metadata=True)
selfimage2 = KerrArray(os.path.join(testdir, "kermit3.png"))
selfimage3 = KerrImageFile(os.path.join(sample_data_dir, "testnormalsave.png"))
selfks = KerrStack(testdir2)


def test_kerr_ops():
    im = selfimage3.clone
    assert isinstance(im.image, KerrArray), "KerrImageFile not blessing the image property correctly"
    im1 = im.float_and_croptext()
    assert isinstance(im1.image, KerrArray), "Calling a crop routine without the _ argument returns a new KerrArray"
    im2 = im.float_and_croptext(_=True)
    assert im2 == im, "Calling crop method with _ argument changed the KerrImageFile"
    im = KerrImageFile(selfimage2.clone)
    im.float_and_croptext(_=True)
    mask = im.image.defect_mask_subtract_image()
    im.image[~mask] = np.mean(im.image[mask])
    _ = im
    assert mask.sum() == 343228, "Mask didn't work out right"
    im = KerrImageFile(selfimage2.clone)
    im.float_and_croptext(_=True)
    mask = im.image.defect_mask(radius=4)
    im.image[~mask] = np.mean(im.image[mask])
    selim2 = im
    assert mask.sum() == 342540, "Mask didn't work out right"
    selim2.level_image()
    selim2.remove_outliers()
    selim2.normalise()
    selim2.plot_histogram()
    selim2.imshow()
    assert len(plt.get_fignums()) == 2, "Didn't open the correct number of figures"
    plt.close("all")


def test_tesseract_ocr():
    # this incidentally tests get_metadata too
    if not selfimage.tesseractable:
        print("#" * 80)
        print("Skipping test that uses tesseract.")
        return None
    _ = selfimage.metadata

    # assert all((m['ocr_scalebar_length_microns']==50.0,
    #                     m['ocr_date']=='11/30/15',
    #                     m['ocr_field'] == -0.13)), 'Misread metadata {}'.format(m))
    _ = (
        "ocr_scalebar_length_pixels",
        "ocr_field_of_view_microns",
        "Loaded from",
        "ocr_microns_per_pixel",
        "ocr_pixels_per_micron",
    )
    # assert all([k in m.keys() for k in keys]), 'some part of the metadata didn\'t load {}'.format(m))
    m_un = selfimage2.metadata
    assert "ocr_field" not in m_un.keys(), "Unannotated image has wrong metadata"


def test_kerrstack():
    print("X" * 80 + "\n" + "Test Kerrstack")
    ks = selfks.clone
    ks.each.normalise(scale=(0, 1.0))
    assert np.min(ks.imarray) == 0.0 and np.max(ks.imarray) == 1.0, "KerrStack subtract failed min,max: {},{}".format(
        np.min(ks.imarray), np.max(ks.imarray)
    )
    d = ks.hysteresis()
    assert isinstance(d, Data), "hysteresis didn't return Data"
    assert d.data.shape == (len(ks), 2), "hysteresis didn't return correct shape"


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb", __file__])
