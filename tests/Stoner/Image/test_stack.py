#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 21:22:18 2018

@author: phygbu
"""
from Stoner.Image import ImageFile, ImageFolder, ImageStack
import numpy as np
import os
import Stoner
import pytest

Stoner.Options.multiprocessing = False

testdir = os.path.join(os.path.dirname(__file__), "coretestdata", "testims")

istack2 = ImageStack()
for theta in np.linspace(0, 360, 16):
    i = ImageFile(np.zeros((100, 100)))
    x, y = 10 * np.cos(np.pi * theta / 180) + 50, 10 * np.sin(np.pi * theta / 180) + 50
    i.draw.circle(x, y, 25)
    i.filename = "Angle {}".format(theta)
    istack2.insert(0, i)


selftd = ImageFolder(testdir, pattern="*.png")
selfks = ImageStack(testdir)
selfks = ImageStack(selftd)  # load in two ways
assert len(selfks) == len(os.listdir(testdir))
selfistack2 = istack2.clone


def test_ImageStack_align():
    assert selfistack2.shape == (16, 100, 100), "ImageStack.shape wrong at {}".format(selfistack2.shape)
    i = ImageFile(np.zeros((100, 100))).draw.circle(50, 50, 25)

    istack2 = selfistack2.clone
    istack2.align(i, method="chi2_shift")
    data = istack2.metadata.slice(["tvec"], output="Data")
    assert data.shape == (16, 2), "Slice metadata went a bit funny"

    istack2 = selfistack2.clone
    with pytest.raises(ValueError):
        istack2.align(0, 10, 20, method="imreg_dft")
    with pytest.raises(TypeError):
        istack2.align(np.zeros(5), method="imreg_dft")
    with pytest.raises(TypeError):
        istack2.align(object, method="imreg_dft")

    istack2.align(method="imreg_dft")
    assert istack2[0]["tvec"] == (0.0, 0.0), "stack didn't align to first image"

    istack2 = selfistack2.clone

    istack2.align(4)
    data = istack2.metadata.slice(["tvec", "angle", "scale"], output="Data")
    assert data.shape == (16, 4), "Slice metadata went a bit funny"
    assert sorted(data.column_headers) == [
        "angle",
        "scale",
        "tvec[0]",
        "tvec[1]",
    ], "slice metadata column headers wrong at {}".format(data.column_headers)


@pytest.mark.filterwarnings("ignore:.*:UserWarning")
def test_ImageStack_methods():
    istack2 = selfistack2.clone
    m1 = selfistack2.mean().crop(10)
    istack2.align(i, method="imreg_dft")
    data = istack2.metadata.slice(["tvec", "angle", "scale"], output="Data")
    assert data.shape == (16, 4), "Slice metadata went a bit funny"
    assert sorted(data.column_headers) == [
        "angle",
        "scale",
        "tvec[0]",
        "tvec[1]",
    ], "slice metadata column headers wrong at {}".format(data.column_headers)

    istack2.each.crop("translation_limits")
    m2 = istack2.mean()
    std = istack2.stddev()
    err = istack2.stderr()

    assert std.mean() < 0.01, "To big a standard deviation in stack after align"
    assert err.mean() < 0.003, "To big a standard error in stack after align"
    assert istack2.shape == (16, 80, 80), "Stack translation_limits and crop failed."

    assert np.abs(m1.mean() - m2.mean()) / m1.mean() < 1e-2, "Problem calculating means of stacks."
    s1 = selfistack2[:, 45:55, 45:55]
    s2 = selfistack2[:, 50, :]
    s3 = selfistack2[:, :, 50]
    s4 = selfistack2[5, :, :]
    assert s1.shape == (16, 10, 10), "3D slicing to produce 3D stack didn't work."
    assert s2.shape == (16, 100), "3D slicing to 2D section z-y plane failed."
    assert s3.shape == (16, 100), "3D slicing to 2D section z-x plane failed."
    assert s4.shape == (100, 100), "3D slicing to 2D section x-y plane failed."
    assert len(selfistack2) == 16, "len(ImageFolder.images) failed."
    sa = []
    for im in selfistack2.images:
        sa.append(im.shape)
    sa = np.array(sa)
    assert np.all(sa == np.ones((16, 2)) * 100), "Result from iterating over images failed."
    selfistack2.each.normalise()
    assert (np.array(selfistack2.each.min()).mean(), np.array(selfistack2.each.max()).mean()) == (
        -1.0,
        1.0,
    ), "Adjust contrast failure"
    im1 = selfistack2[0]
    im1.normalise()
    im1.convert(np.int32)
    im2 = im1.convert(np.float32, force_copy=True)
    conv_err = (selfistack2[0].image - im2.image).max()
    assert conv_err < 1e-7, "Problems double converting images:{}.".format(conv_err)
    im1 = selfistack2[0].convert(np.int64)
    im1 = im1.convert(np.int8)
    im2 = selfistack2[0].convert(np.int8)
    assert abs((im2 - im1).max()) <= 2.0, "Failed up/down conversion to integer images."


def test_init():
    # try to init with a few different call sequences
    listinit = []
    for i in range(10):
        listinit.append(np.arange(12).reshape(3, 4))
    npinit = np.arange(1000).reshape(5, 10, 20)
    listinit = ImageStack(listinit)
    assert listinit.shape == (10, 3, 4), "problem with initialising ImageStack with list of data"
    npinitist = ImageStack(npinit)
    assert np.allclose(npinitist.imarray, npinit), "problem initiating with 3d numpy array"
    ist2init = ImageStack(selfistack2)
    assert np.allclose(ist2init.imarray, selfistack2.imarray), "problem initiating with other ImageStack"
    assert all(
        [k in ist2init[0].metadata.keys() for k in selfistack2[0].metadata.keys()]
    ), "problem with metadata when initiating with other ImageStack"
    imfinit = ImageStack(selftd)  # init with another ImageFolder
    assert len(imfinit) == 8, "Couldn't load from another ImageFolder object"


def test_accessing():
    # ensure we can write and read to the stack in different ways
    ist2 = ImageStack(np.arange(60).reshape(4, 3, 5))
    im = np.zeros((3, 5), dtype=int)
    ist2[0].image = im
    ist2[1] = im
    ist2[2].image[:] = im
    ist2[3][0, 0] = 100
    ist2[3].image[0, 1] = 101
    # ist2[3,0,2] = 102 may want to support this type of index accessing in the future?
    for i in range(3):
        assert np.allclose(ist2[i].asarray(), im)
    assert np.allclose(ist2[3][0, 0], 100)
    assert np.allclose(ist2[3][0, 1], 101)
    # check imarray behaviour
    ist2.imarray = np.arange(60).reshape(4, 3, 5) * 3
    assert np.allclose(ist2.imarray, np.arange(60).reshape(4, 3, 5) * 3)
    ist2.imarray[1, 2, 3] = 500
    assert ist2.imarray[1, 2, 3] == 500
    # check slice access #not implemented yet
    # im2 = np.zeros((2,3,5))
    # ist2[3:5] = im2 #try setting two images at once
    # assert np.allclose(ist2[4],im))


def test_methods():
    # check function generator machinery works
    selfistack2.each.crop(0, 30, 0, 50)
    assert selfistack2.shape == (16, 50, 30), "Unexpected size of imagestack got {} for 16x50x30".format(
        selfistack2.shape
    )
    ist2 = ImageStack(np.arange(60).reshape(4, 3, 5))
    assert issubclass(
        ist2.imarray.dtype.type, np.integer
    ), "Unexpected dtype in image stack2 got {} not int32".format(ist2.imarray.dtype)
    t1 = ImageStack(np.arange(60).reshape(4, 3, 5))
    t1.asfloat(normalise=False, clip_negative=False)
    assert t1.imarray.dtype == np.float64, f"Type Failure {t1.imarray.dtype}"
    assert np.max(t1.imarray) == 59.0
    t2 = ImageStack(np.arange(60).reshape(4, 3, 5))
    t2.asfloat(normalise=True, clip_negative=True)
    # assert  np.max(t2.imarray) == (2*59+1)/(2**31-(-2**31)) )
    assert np.min(t2.imarray) >= 0
    ist3 = ist2.clone
    assert not np.may_share_memory(ist2.imarray, ist3.imarray)
    del ist3[-1]
    assert len(ist3) == len(ist2) - 1
    assert np.allclose(ist3[0], ist2[0])
    ist3.insert(1, np.arange(18).reshape(3, 6))
    assert ist3[1].shape == (3, 6), "inserting an image of different size to stack"
    im1 = ImageFile(np.zeros((100, 100)))
    im1.draw.circle(50, 50, 25)
    im2 = im1.clone
    pi = np.pi
    X, Y = np.mgrid[-pi : pi : pi / 50, -pi : pi : pi / 50]
    bground = ImageFile(np.sin(X) + np.cos(Y))
    im2.image += bground.image
    bground.mask = im1.image == 1.0
    ist = ImageStack() + im2
    ist.subtract(bground)
    assert np.all(np.isclose(ist[0].image, im1.image))


def test_clone():
    ist2 = ImageStack(np.arange(60).reshape(4, 3, 5))
    ist3 = ist2.clone
    assert not np.may_share_memory(ist2.imarray, ist3.imarray)  # Check new memory has been allocated for clone
    del ist3[-1]
    assert len(ist3) == len(ist2) - 1  # Check deletion operation only happens for clone
    assert np.allclose(ist3[0], ist2[0])  # Check values are the same
    ist3.insert(1, np.arange(18).reshape(3, 6))  # Insert a larger image
    assert ist3[1].shape == (3, 6), "inserting an image of different size to stack"


def test_operators():
    im1 = ImageFile(np.zeros((100, 100)) + 0.75)
    im2 = ImageFile(np.zeros((100, 100)) + 0.25)
    ist1 = ImageStack() + im1
    ist2 = ImageStack() + im2
    ist3 = ist1 // ist2
    assert np.all(np.isclose(ist3[0].image, np.ones((100, 100)) * 0.5))
    im1a = ImageFile(np.ones((100, 100), dtype=int))
    with pytest.raises(ValueError):
        ist1a = ImageStack() + im1a
        _ = ist1a // ist2
    im1b = im1.clone
    im1b.image = im1b.image[1:-1, 1:-1]
    ist1b = ImageStack() + im1b
    with pytest.raises(ValueError):
        _ = ist1b // ist2


def test_mask():
    im = ImageFile(np.arange(12).reshape(3, 4))
    im.mask = np.zeros(im.shape, dtype=bool)
    im.mask.data[1, 1] = True
    ist2 = ImageStack(np.arange(60).reshape(4, 3, 5))
    ist2.insert(1, im)  # Insert an image with a mask
    assert ist2[1].mask.data.shape == ist2[1].shape
    assert ist2[1].mask[1, 1], "inserting an image with a mask into an ImageStack has failed"
    ist2[3].mask = np.ones(im.shape, dtype=bool)
    assert np.all(ist2[3].mask), "setting mask on an image stack item not working"
    istack2 = selfistack2.clone
    mask = ImageFile(np.zeros_like(istack2[0].image)).mask.draw.circle(20, 20, 10).mask
    mask = ~mask
    istack2.each.mask = mask
    assert istack2[0].mask[0, 0], "Mask not set correctly in stack"
    assert not istack2[0].mask[20, 20], "Mask not set correctly in stack"


if __name__ == "__main__":
    pytest.main(["--pdb", __file__])
