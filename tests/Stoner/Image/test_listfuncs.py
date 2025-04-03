# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:21:48 2016

@author: phyrct
"""
from types import MethodType
from Stoner.Image import ImageArray, ImageFile
from Stoner.HDF5 import STXMImage
from Stoner import __datapath__
import pytest
from os.path import dirname, join
import numpy as np
from lmfit.models import LorentzianModel
import matplotlib.pyplot as plt

thisdir = dirname(__file__)


def mag(x):
    return np.sqrt(np.dot(x, x))


def guess(self, data, **kwargs):
    """Function to guess the parameters of two Lorentzian peaks."""
    x = kwargs.get("x")
    length = len(data)
    i1 = np.argmax(data[: length // 2])  # Location of first peak
    i2 = np.argmax(data[length // 2 :]) + length // 2

    params = self.make_params()
    for k, p in zip(
        params,
        [
            data[i1] / np.sqrt(2),
            x[i1],
            0.25,
            0.5,
            data[i1],
            data[i2] / np.sqrt(2),
            x[i2],
            0.25,
            0.5,
            data[i2],
        ],
    ):
        params[k].value = p
    return params


def test_extra():
    img = ImageFile(__datapath__ / "kermit.png")
    img.normalise(limits=(0.01, 0.99))
    assert np.isclose(img.mean(), 0.04372557050412787)
    img.clip_neg()
    assert np.isclose(img.mean(), 0.28186646622726236)
    assert np.isclose(img.profile_line(x=120).y.max(), 1.0)
    assert np.isclose(img.profile_line(y=120).y.max(), 0.9450988802254867)
    assert np.isclose(img.profile_line(x=120, no_scale=True).y.max(), 1.0)
    assert np.isclose(img.profile_line(y=120, no_scale=True).y.max(), 0.9450988802254867)
    assert np.isclose(img.profile_line(10.0, -10.0).y.mean(), 0.013888888888888864)
    img = ImageFile(np.random.normal(size=(100, 100), scale=1.0))
    img.sgolay2d(points=5)
    assert np.sqrt(img.image**2).mean() < 0.2


def test_imagefile_ops():
    img_a2 = STXMImage(join(__datapath__, "Sample_Image_2017-10-15_100.hdf5"))
    img_a3 = STXMImage(join(__datapath__, "Sample_Image_2017-10-15_101.hdf5"))
    img_a2.gridimage()
    img_a3.gridimage()
    img_a2.crop(5, -15, 5, -5, _=True)
    img_a3.crop(5, -15, 5, -5, _=True)
    b = img_a2 // img_a3
    assert b.shape == (90, 80), "Failure to crop image correctly."
    assert b.max() > 0.047, "XMCD Ratio calculation failed"
    assert b.min() < -0.05, "XMCD Ratio calculation failed"
    b.normalise()
    assert b.max() == 1.0, "Normalise Image failed"
    assert b.min() == -1.0, "Normalise Image failed"
    profile = b.profile_line((0, 0), (100, 100))
    profile.plot()
    b.mask = img_a2.image > 25e3
    hist = b.hist(bins=200)
    hist.column_headers = ["XMCD Signal", "Frequency"]
    hist.labels = None

    # Construct a two Lorentzian peak model
    peak1 = LorentzianModel(prefix="peak1_")
    peak2 = LorentzianModel(prefix="peak2_")
    params = peak1.make_params()
    params.update(peak2.make_params())
    double_peak = peak1 + peak2
    # This shows how to replace the default guess (which is useless) with our function guess
    double_peak.guess = MethodType(guess, double_peak)

    print(peak1, peak2, params)
    res = hist.lmfit(double_peak, output="report")
    hist.add_column(res.init_fit, header="Initial Fit")
    hist.add_column(res.best_fit, header="Best Fit")
    hist.setas = "xyyy"
    hist.plot(fmt=["b+", "b--", "r-"])
    plt.close("all")

    # b.adnewjust_contrast((0.1,0.9),percent=True)z


def test_funcs():
    img_a = ImageArray(join(thisdir, "coretestdata/im2_noannotations.png"))
    img_a1 = ImageArray(join(thisdir, "coretestdata/im1_annotated.png"))
    b = img_a.translate((2.5, 3))
    c = b.correct_drift(ref=img_a)
    d = b.align(img_a, method="scharr")
    tv = np.array(d["tvec"]) * 10
    cd = np.array(c["correct_drift"]) * 10
    shift = np.array([25, 30])
    d1 = mag(cd - shift)
    d2 = mag(tv - (-shift[::-1]))
    assert d1 < 1.5, "Drift Correct off by more than 0.1 pxiels."
    assert d2 < 1.5, "Scharr Alignment off by more than 0.1 pxiels."

    a1 = ImageFile(img_a1.clone)
    a1.asfloat()
    a1.image = np.sqrt(a1.image) / 2 + 0.25
    a1.adjust_contrast()
    assert a1.span() == (0.0, 1.0), "Either adjust_contrast or span failed with an ImageFile"


#        print("#"*80)
#        print(a.metadata)
#        print(img_a1.metadata)
#        print(all([k in a.metadata.keys() for k in img_a1.metadata.keys()]))


def test_imagefuncs():
    img_a2 = STXMImage(join(__datapath__, "Sample_Image_2017-10-15_100.hdf5"))
    img_a2.subtract_image(img_a2.image, offset=0)
    assert np.all(img_a2.image <= 0.0001), "Failed to subtract image from itself"
    x = np.linspace(-3 * np.pi, 3 * np.pi, 101)
    X, Y = np.meshgrid(x, x)
    i = ImageFile(np.sin(X) * np.cos(Y))
    i2 = i.clone
    j = i.fft(window="hamming", remove_dc=True)
    assert j.image[47, 47] > 583.0
    j.imshow()
    assert len(plt.get_fignums()) == 1, "Imshow didn't open one window"
    plt.close("all")
    assert j.radial_profile().y.argmax() == 4
    assert j.radial_profile(angle=np.pi / 4).y.argmax() == 3
    img_a2.imshow(title=None, figure=None)
    img_a2.imshow(title="Hello", figure=1)
    assert len(plt.get_fignums()) == 1, "Imshow with arguments didn't open one window"
    plt.close("all")
    i = i2
    k = i + 0.2 * X - 0.1 * Y + 0.2
    k.level_image(mode="norm")
    j = k - i
    assert np.isclose(np.max(j), 0.004094023178423445), "Level Image failed"
    i2 = i.clone
    i2.quantize([-0.5, 0, 0.5])
    assert np.all(np.unique(i2.data) == np.array([-0.5, 0, 0.5])), "Quantise levels failed"
    i2 = i.clone
    i2.quantize([-0.5, 0, 0.5], levels=[-0.25, 0.25])
    assert np.all(np.unique(i2.data) == np.array([-0.5, 0, 0.5])), "Quantise levels failed"
    i2 = i.clone
    i2.rotate(np.pi / 4)
    i2.fft()
    assert np.all(np.unique(np.argmax(i2, axis=1)) == np.array([46, 47, 49, 50, 51, 53])), "FFT of image test failed"


if __name__ == "__main__":  # Run some tests manually to allow debugging
    pytest.main(["--pdb","-W","error", __file__])
