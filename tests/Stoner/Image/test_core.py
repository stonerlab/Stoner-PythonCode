# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:09:04 2016

@author: phyrct
"""

from Stoner.Image import ImageArray, ImageFile
from Stoner.Core import TypeHintedDict
from Stoner import Data, __home__
import numpy as np
import pytest
from os import path
import tempfile
import os
import sys
import shutil
from PIL import Image

from scipy.version import version as spv

spv = [int(x) for x in spv.split(".")]


# data arrays for testing - some useful small images for tests

thisdir = path.dirname(__file__)


def shares_memory(arr1, arr2):
    """Check if two numpy arrays share memory"""
    # ret = arr1.base is arr2 or arr2.base is arr1
    ret = np.may_share_memory(arr1, arr2)
    return ret


def _has_method(obj, method):
    """Returns True if obj has a method."""
    return callable(getattr(obj, method, None))


# random array shape 3,4
selfarr = np.array(
    [
        [0.41674764, 0.66100043, 0.91755303, 0.33796703],
        [0.06017535, 0.1440342, 0.34441777, 0.9915282],
        [0.2984083, 0.9167951, 0.73820304, 0.7655299],
    ]
)
selfimarr = ImageArray(np.copy(selfarr))  # ImageArray object
selfimarrfile = ImageArray(os.path.join(thisdir, "coretestdata/im1_annotated.png"))
# ImageArray object from file

selfimarr_x = np.copy(selfimarr).view(ImageArray)
#####test loading with different datatypes  ####


def test_load_from_array():
    # from array
    assert np.array_equal(selfimarr, selfarr)
    # int type
    imarr = ImageArray(np.arange(12, dtype="int32").reshape(3, 4))
    assert imarr.dtype == np.dtype("int32"), "Failed to set correct dtype - actual dtype={}".format(imarr.dtype)


def test_load_from_ImageArray():
    # from ImageArray
    t = ImageArray(selfimarr)
    assert shares_memory(selfimarr, t), "no overlap on creating ImageArray from ImageArray"


def test_load_from_png():
    subpath = os.path.join("coretestdata", "im1_annotated.png")
    fpath = os.path.join(thisdir, subpath)
    anim = ImageArray(fpath)
    assert (
        os.path.normpath(anim.metadata["Loaded from"]).lower() == os.path.normpath(fpath).lower()
    ), "Failed with {os.path.normpath(anim.metadata['Loaded from'])} and {os.path.normpath(fpath)}"
    cwd = os.getcwd()
    os.chdir(thisdir)
    anim = ImageArray(subpath)
    # check full path is in loaded from metadata
    assert (
        os.path.normpath(anim.metadata["Loaded from"]).lower() == os.path.normpath(fpath).lower()
    ), "Full path not in metadata: {}".format(anim["Loaded from"])
    os.chdir(cwd)


def test_load_save_all():
    tmpdir = tempfile.mkdtemp()
    pth = path.join(__home__, "..")
    datadir = path.join(pth, "sample-data")
    image = ImageFile(path.join(datadir, "kermit.png"))
    ims = {}
    fmts = ["uint8", "uint16", "uint32", "float32"]
    modes = {"uint8": "L", "uint32": "RGBX", "float32": "F"}
    for fmt in fmts:
        ims[fmt] = image.clone.convert(fmt)
        ims[fmt].save_tiff(path.join(tmpdir, "kermit-{}.tiff".format(fmt)))
        ims[fmt].save_tiff(path.join(tmpdir, "kermit-forcetype{}.tiff".format(fmt)), forcetype=True)
        ims[fmt].save_npy(path.join(tmpdir, "kermit-{}.npy".format(fmt)))
        if fmt != "uint16":
            im = Image.fromarray((ims[fmt].image.view(np.ndarray)).astype(fmt), mode=modes[fmt])
            im.save(path.join(tmpdir, "kermit-nometadata-{}.tiff".format(fmt)))
        del ims[fmt]["Loaded from"]
    for fmt in fmts:
        iml = ImageFile(path.join(tmpdir, "kermit-{}.tiff".format(fmt)))
        del iml["Loaded from"]
        assert ims[fmt] == iml, f"{ims[fmt].metadata} {iml.metadata}"
        iml = ImageFile(path.join(tmpdir, f"kermit-{fmt}.npy"))
        del iml["Loaded from"]
        assert np.all(ims[fmt].data == iml.data), f"Round tripping npy with format {fmt} failed"
        if fmt != "uint16":
            im = ImageFile(path.join(tmpdir, f"kermit-nometadata-{fmt}.tiff"))
            assert np.all(im.data == ims[fmt].data), f"Loading from tif without metadata failed for {fmt}"
    shutil.rmtree(tmpdir)
    _ = image.convert("uint8")


def test_conversion_bitness():
    pth = path.join(__home__, "..")
    datadir = path.join(pth, "sample-data")
    image = ImageFile(path.join(datadir, "kermit.png"))
    img = image.clone
    img2 = img.clone // 256
    img.convert("uint8")  # Down conversion with prevision loss
    assert np.all(img.image == img2.image)
    img.convert("int16")  # Upconversion with unequal bit lengths
    assert img.max() == 32767 and img.min() == 0
    img3 = img2.clone.convert("uint8")  # Downconversion without precision loss
    assert np.all(img2.image == img3.image)


def test_load_from_ImageFile():
    # uses the ImageFile.im attribute to set up ImageArray. Memory overlaps
    imfi = ImageFile(selfarr)
    imarr = ImageArray(imfi)
    assert np.array_equal(imarr, imfi.image), "Initialising from ImageFile failed"
    assert shares_memory(imarr, imfi.image)


def test_load_from_list():
    t = ImageArray([[1, 3], [3, 2], [4, 3]])
    assert np.array_equal(t, np.array([[1, 3], [3, 2], [4, 3]])), "Initialising from list failed"


def test_load_1d_data():
    t = ImageArray(np.arange(10) / 10.0)
    assert len(t.shape) == 2  # converts to 2d


def test_load_no_args():
    # Should be a 2d empty array
    t = ImageArray()
    assert len(t.shape) == 2
    assert t.size == 0


def test_load_bad_data():
    def testload(arg):
        ImageArray(arg)

    # dictionary
    with pytest.raises(ValueError):
        testload({"a": 1})

    # bad filename
    with pytest.raises(ValueError):
        testload("sillyfile.xyz")


def test_load_kwargs():
    # metadata keyword arg
    t = ImageArray(selfarr, metadata={"a": 5, "b": 7})
    assert ("a" in t.metadata.keys()) and ("b" in t.metadata.keys())
    assert t.metadata["a"] == 5
    # asfloat
    t = ImageArray(np.arange(12).reshape(3, 4), asfloat=True)
    assert t.dtype == np.float64, "Initialising asfloat failed"
    assert "Loaded from" in t.metadata.keys(), "Loaded from should always be in metadata"


# ####test attributes ##


def test_filename():
    im = ImageArray(np.linspace(0, 1, 12).reshape(3, 4))
    fpath = os.path.join(thisdir, "coretestdata/im1_annotated.png")
    assert (
        os.path.normpath(selfimarrfile.filename).lower() == os.path.normpath(fpath).lower()
    ), f"Failed with {os.path.normpath(selfimarrfile.filename)} and {os.path.normpath(fpath)}"
    im = ImageArray(np.linspace(0, 1, 12).reshape(3, 4))
    im["Loaded from"]
    im.filename
    assert im.filename == "", "{}, {}".format(selfimarr.shape, im.filename)


def test_clone():
    selfimarr["abc"] = 123  # add some metadata
    selfimarr["nested"] = [1, 2, 3]  # add some nested metadata to check deepcopy
    selfimarr.userxyz = 123  # add a user attribute
    c = selfimarr.clone
    assert isinstance(c, ImageArray), "Clone not ImageArray"
    assert np.array_equal(c, selfimarr), "Clone not replicating elements"
    assert all([k in c.metadata.keys() for k in selfimarr.metadata.keys()]), "Clone not replicating metadata"
    assert not shares_memory(c, selfimarr), "memory overlap on clone"  # formal check
    selfimarr["bcd"] = 234
    assert "bcd" not in c.metadata.keys(), "memory overlap for metadata on clone"
    selfimarr["nested"][0] = 2
    assert selfimarr["nested"][0] != c["nested"][0], "deepcopy not working on metadata"
    assert c.userxyz == 123
    c.userxyz = 234
    assert c.userxyz != selfimarr.userxyz
    img = ImageFile(np.zeros((100, 100)))
    img.abcd = "Hello"
    img2 = img.clone
    assert img2.abcd == "Hello"
    del img["Loaded from"]
    assert len(img.metadata) == 0
    del img.abcd
    assert not hasattr(img, "abcd")


def test_metadata():
    assert isinstance(selfimarr.metadata, TypeHintedDict)
    selfimarr["testmeta"] = "abc"
    assert selfimarr["testmeta"] == "abc", "Couldn't change metadata"
    del selfimarr["testmeta"]
    assert "testmeta" not in selfimarr.keys(), "Couldn't delete metadata"

    # bad data
    def test(imarr):
        imarr.metadata = (1, 2, 3)

    with pytest.raises(TypeError):
        test(selfimarr)  # check it won't let you do this


### test numpy like creation behaviour #
def test_user_attributes():
    selfimarr.abc = "new att"
    assert hasattr(selfimarr, "abc")
    t = ImageArray(selfimarr)
    assert hasattr(t, "abc"), "problem copying new attributes"
    t = selfimarr.view(ImageArray)  # check array_finalize copies attribute over
    assert hasattr(t, "abc")
    t = selfimarr * np.random.random(selfimarr.shape)
    assert isinstance(t, ImageArray), "problem with ufuncs"
    assert hasattr(t, "abc"), "Ufunc lost attribute!"


#####  test functionality  ##
def test_save():
    testfile = path.join(thisdir, "coretestdata", "testsave")
    ext = [".png", ".npy"]
    keys = selfimarr.keys()
    for e in ext:
        selfimarr.save(testfile + e)
        load = ImageArray(testfile + e)
        assert all([k in load.keys() for k in keys]), "problem saving metadata {} {}".format(list(load.keys()), e)
        if e == ".npy":
            # tolerance is really poor for png which saves in 8bit format
            assert np.allclose(selfimarr, load), "data not the same for extension {}".format(e)
        os.remove(testfile + e)  # tidy up


def test_savetiff():
    testfile = path.join(thisdir, "coretestdata", "testsave.tiff")
    # create a few different data types
    testb = ImageArray(np.zeros((4, 5), dtype=bool))  # bool
    testb[0, :] = True
    testui = ImageArray(np.arange(20).reshape(4, 5))  # int32
    testi = ImageArray(np.copy(testui) - 10)
    testf = ImageArray(np.linspace(-1, 1, 20).reshape(4, 5))  # float64
    for im in [testb, testui, testi, testf]:
        im["a"] = [1, 2, 3]
        im["b"] = "abc"  # add some test metadata
        im.filename = testfile
        im.save()
        n = ImageArray(testfile)
        assert all([n["a"][i] == im["a"][i] for i in range(len(n["a"]))])
        assert n["b"] == im["b"]
        assert "ImageArray.dtype" in n.metadata.keys()  # check the dtype metadata got added
        assert im.dtype == n.dtype  # check the datatype
        assert np.allclose(im, n)  # check the data


def test_max_box():
    s = selfimarr.shape
    assert selfimarr.max_box == (0, s[1], 0, s[0])


def test_crop():
    c = selfimarr.crop((1, 3, 1, 4), copy=True)
    assert np.array_equal(c, selfimarr[1:4, 1:3]), "crop didn't work"
    assert not shares_memory(c, selfimarr), "crop copy failed"
    c2 = selfimarr.crop(1, 3, 1, 4, copy=True)
    assert np.array_equal(c2, c), "crop with separate arguments didn't work"
    c3 = selfimarr.crop(box=(1, 3, 1, 4), copy=False)
    assert np.array_equal(c3, c), "crop with no arguments failed"
    assert shares_memory(selfimarr, c3), "crop with no copy failed"


def test_asint():
    ui = selfimarr.asint()
    assert ui.dtype == np.uint16
    intarr = np.array(
        [[27311, 43318, 60131, 22148], [3943, 9439, 22571, 64979], [19556, 60082, 48378, 50169]], dtype=np.uint16
    )
    assert np.array_equal(ui, intarr)


def test_other_funcs():
    """test imagefuncs add owns. the functions themselves are not checked
    and should include a few examples in the doc strings for testing"""
    assert hasattr(selfimarr, "do_nothing"), "imagefuncs not being added to dir"
    assert hasattr(
        selfimarr, "Stoner__Image__imagefuncs__do_nothing"
    ), "Stoner image functions not added with full namke"
    assert hasattr(selfimarr, "img_as_float"), "skimage funcs not being added to dir"
    im = selfimarr.do_nothing()  # see if it can run
    assert np.allclose(im, selfimarr), "imagefuncs not working"
    assert shares_memory(im, selfimarr), "imagefunc failed to share memory"
    im0 = ImageArray(np.linspace(0, 1, 12).reshape(3, 4))
    im1 = im0.clone * 5
    im2 = im1.rescale_intensity()  # test skimage
    assert np.allclose(im2, im0), "skimage func failed"
    assert not shares_memory(im2, im1), "skimage failed to clone"
    im3 = im1.skimage__exposure__exposure__rescale_intensity()  # test call with module name
    assert np.allclose(im3, im0), "skimage call with module name failed"


def test_attrs():
    test = ImageArray([])
    assert _has_method(test, "gaussian_filter"), "Failed to get scipy.ndimage.gaussian_filter as ImageArray attr"
    assert _has_method(test, "gaussian"), "Failed to get skimage.filter.gaussian as ImageArray attr"
    assert _has_method(test, "gridimage"), "Failed to get imagfuncs.gridimage has ImageArray attr"


selfa = np.linspace(0, 5, 12).reshape(3, 4)
selfifi = ImageFile(selfa)
selfimgFile = ImageFile(os.path.join(thisdir, "coretestdata/im1_annotated.png"))


def test_constructors():
    selfimgFile = ImageFile(os.path.join(thisdir, "coretestdata/im1_annotated.png"))
    selfd = Data(selfimgFile)
    selfimgFile2 = ImageFile(selfd)
    selfimgFile2.pop("Stoner.class", "")
    selfimgFile.pop("Stoner.class", "")
    del selfimgFile2["x_vector"]
    del selfimgFile2["y_vector"]
    assert np.all(
        selfimgFile.image == selfimgFile2.image
    ), "Roundtripping constructor through Data failed to dupliocate data."
    assert (
        selfimgFile.metadata == selfimgFile2.metadata
    ), "Roundtripping constructor through Data failed to duplicate metadata: {} {}".format(
        selfimgFile2.metadata ^ selfimgFile.metadata, selfimgFile.metadata ^ selfimgFile2.metadata
    )


def test_properties():
    selfa = np.linspace(0, 5, 12).reshape(3, 4)
    assert np.allclose(selfifi.image, selfa)
    selfifi[0, 1] = 10.1
    assert selfifi.image[0, 1] == 10.1


def test_attrs2():
    """Check that creating new attributes puts them on ImageFile and not ImageArray and updates public_attrs."""
    assert selfifi["Loaded from"] == ""
    selfifi.abc = 123
    assert selfifi.abc == 123
    assert "abc" in selfifi.__dict__
    assert "abc" not in selfifi._image.__dict__
    assert "abc" in selfifi._public_attrs


def test_methods():
    b = np.arange(12).reshape(3, 4)
    ifi = ImageFile(b)
    ifi.asfloat(normalise=False, clip_negative=False)  # convert in place
    assert ifi.image.dtype.kind == "f"
    assert np.max(ifi) == 11.0
    ifi.image == ifi.image * 5
    ifi.rescale_intensity()
    assert np.allclose(ifi.image, np.linspace(0, 1, 12).reshape(3, 4))
    ifi.crop(0, 3, 0, None)
    assert ifi.shape == (3, 3)  # check crop is forced to overwrite ifi despite shape change
    datadir = path.join(__home__, "..", "sample-data")
    image = ImageFile(path.join(datadir, "kermit.png")).asfloat(normalise=False)
    i2 = image.clone.crop(5, _=True)
    assert i2.shape == (469, 349), "Failed to trim box by integer"
    i2 = image.clone.crop(0.25, _=True)
    assert i2.shape == (269, 269), "Failed to trim box by float"
    i2 = image.clone
    i2.crop([0.1, 0.2, 0.05, 0.1], _=True)
    assert i2.shape == (24, 36), "Failed to trim box by sequence of floats"
    assert image.aspect == pytest.approx(0.7494780793), "Aspect ratio failed"
    assert image.centre == (239.5, 179.5), "Failed to read image.centre"
    i2 = image.CW
    assert i2.shape == (359, 479), "Failed to rotate clockwise"
    i3 = i2.CCW
    assert i3.shape == (479, 359), "Failed to rotate counter-clockwise"
    i3 = i2.flip_h
    assert np.all(i3[:, 0] == i2[:, -1]), "Flip Horizontal failed"
    i3 = i2.flip_v
    assert np.all(i3[0, :] == i2[-1, :]), "Flip Horizontal failed"
    i2 = image.clone
    i3 = i2 - 127
    assert i3.mean() == pytest.approx(33940.72596111909, rel=1e-2), "Subtract integer failed."
    with pytest.raises(TypeError):
        i2 - "Gobble"
    test = i2
    assert _has_method(test, "gaussian_filter"), "Failed to get scipy.ndimage.gaussian_filter as ImageFile attr"
    assert _has_method(test, "gaussian"), "Failed to get skimage.filter.gaussian as ImageFile attr"
    assert _has_method(test, "gridimage"), "Failed to get imagfuncs.gridimage has ImageFile attr"
    assert image._repr_png_().startswith(b"\x89PNG\r\n"), "Failed to do ImageFile png representation"


def test_mask():
    i = np.ones((200, 200), dtype="uint8") * np.linspace(1, 200, 200).astype("uint8")
    i = ImageFile(i)
    i.mask.draw.rectangle(100, 50, 100, 100)
    assert i.mean() == pytest.approx(117.1666666666666, 1.0), "Mean after masked rectangle failed"
    i.mask.invert()
    assert i.mean() == pytest.approx(50.5, 1.0), "Mean after inverted masked rectangle failed"
    i.mask.clear()
    assert i.mean() == pytest.approx(100.5, 1.0), "Mean after clearing mask failed"
    i.mask.draw.square(100, 50, 100)
    assert i.mean() == pytest.approx(117.1666666666666, 1.0), "Mean after masked rectangle faile"
    i.mask.clear()
    i.mask.draw.annulus(100, 50, 35, 25)
    assert i.mean() == pytest.approx(102.96850393700, 1.0), "Mean after annular block mask failed"
    i.mask = False
    i.mask.draw.annulus(100, 50, 25, 35)
    assert i.mean() == pytest.approx(51.0), "Mean after annular pass mask failed"
    i.mask[:, :] = False
    assert not np.any(i.mask), "Setting Mask by index failed"
    i.mask = -i.mask
    assert np.all(i.mask), "Setting Mask by index failed"
    i.mask = ~i.mask
    assert not np.any(i.mask), "Setting Mask by index failed"
    i.mask.draw.circle(100, 100, 20)
    st = repr(i.mask)
    assert st.count("X") == i.mask.sum(), "String representation of mask failed"
    assert np.all(i.mask.image == i.mask._mask), "Failed to access mak data by image attr"
    i.mask = False
    i2 = i.clone
    i.mask.draw.rectangle(100, 100, 100, 50, angle=np.pi / 2)
    i2.mask.draw.rectangle(100, 100, 50, 100)
    assert np.all(i.mask.image == i2.mask.image), "Drawing rectangle with angle failed"
    assert i.mask._repr_png_().startswith(b"\x89PNG\r\n"), "Failed to do mask png representation"
    i = ImageFile(np.zeros((100, 100)))
    i2 = i.clone
    i2.draw.circle(50, 50, 25)
    i.mask = i2
    assert i.mask.sum() == i2.sum(), "Setting mask from ImageFile Failed"
    i2.mask = i.mask
    assert np.all(i.mask.image == i2.mask.image), "Failed to set mask by mask proxy"
    i = ImageFile(np.ones((100, 100)))
    i.mask.draw.square(50, 50, 10)
    i.mask.rotate(angle=np.pi / 4)
    i.mask.invert()
    i2 = ImageFile(np.zeros((100, 100)))
    i2.draw.square(50, 50, 10, angle=np.pi / 4)
    assert i.sum() == pytest.approx(i2.sum(), 1.5), "Check on rotated mask failed !"


def test_draw():
    i = ImageFile(np.zeros((200, 200)))
    attrs = [x for x in dir(i.draw) if not x.startswith("_")]
    expected = 21
    assert len(attrs) == expected, "Directory of DrawProxy failed"
    i2 = i.clone
    i2.draw.circle(10, 100, 10, value=1)
    assert i2.image.sum() == 305
    i2 = i.clone.draw.circle_perimeter_aa(100, 100, 10, value=1)
    assert np.isclose(i2.image.sum(), 56.5657137141714)
    i2 = i.clone.draw.random_shapes(i.shape, 1, 1, 30, 100)
    assert i2.label()[1] == 1
    i2 = i.clone.draw.rectangle_perimeter(100, 100, 20, 10, value=1)
    assert i2.image.sum() == 60.0
    i2.mask.threshold(0.5)
    assert i2.mask.sum() == 60


def test_operators():
    i = ImageFile(np.zeros((10, 10)))
    i = i + 1
    i += 4
    i += i.clone
    i += i.clone.data
    assert i.sum() == 2000, "Addition operators failed"
    i /= 4
    assert i.sum() == 500, "Division operators failed"
    i = i / 5
    assert i.sum() == 100, "Division operators failed"
    i2 = i.clone
    i -= 0.75
    i2 -= 0.25
    i3 = i2 // i
    assert i3.sum() == 50, "Division operators failed"
    i = ImageFile(np.zeros((10, 5)))
    i = ~i
    assert i.shape == (5, 10), "Invert to rotate failed"
    i.image = i.image.astype("uint8")
    i = -i
    assert i.sum() == 50 * 255, "Negate operators failed"


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main([__file__, "--pdb"])
