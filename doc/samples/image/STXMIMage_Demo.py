"""Demonstrate STXM Image Processing - G.Burnell Nov. 2017"""

# pylint: disable=invalid-name,no-member
from os.path import join, dirname
from types import MethodType

import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import LorentzianModel

from Stoner.Image import ImageFolder
from Stoner.HDF5 import STXMImage

# Load the images
thisdir = dirname(__file__)

imgs = ImageFolder(type=STXMImage)
for fname in [
    "Sample_Image_2017-10-15_100.hdf5",
    "Sample_Image_2017-10-15_101.hdf5",
]:
    # Load the image
    img = STXMImage(join(thisdir, "..", "..", "..", "sample-data", fname))
    a1 = id(img.metadata)
    img.gridimage()
    a2 = id(img.metadata)
    img.crop(5, -15, 5, -5, _=True)  # regularise grid and crop
    a3 = id(img.metadata)
    imgs += img
    a4 = [id(i.metadata) for i in imgs]

# Align the two images
imgs.align(method="imreg_dft", scale=10)
imgs.each.crop(2)

# Calculate the XMCD image
xmcd = imgs[0] // imgs[1]
xmcd.normalise()

imgs += xmcd
strctural = (imgs[0] + imgs[1]) / 2
imgs += strctural

# Mask out the background, using the structural image
xmcd.mask = strctural.image > strctural.threshold_otsu()

# Remove the background XMCD atio
xmcd.mask.invert()
mean = xmcd.mean()
xmcd.mask = False
xmcd -= mean
xmcd.mask = strctural.image > strctural.threshold_otsu()
xmcd.normalise()

# Create a profile and plot it
profile = xmcd.profile_line((0, 0), (100, 100))
profile.figure(figsize=(7, 6), no_axes=True)
profile.subplot(222)
profile.plot()
profile.title = "XMCD Cross Section"

# Show the images as well
profile.subplot(221)
strctural.imshow(figure=profile.fig, title="Structural Image")

profile.subplot(223)
xmcd.imshow(figure=profile.fig, title="XMCD Image")

# Make a histogram of the intensity values
hist = xmcd.hist(bins=200)
hist.column_headers = ["XMCD Signal", "Frequency"]
hist.labels = None
hist.fig = profile.fig

# Construct a two Lorentzian peak model
peak1 = LorentzianModel(prefix="peak1_")
peak2 = LorentzianModel(prefix="peak2_")
params = peak1.make_params()
params.update(peak2.make_params())
double_peak = peak1 + peak2


def guess(self, data, **kwargs):
    """Function to guess the parameters of two Lorentzian peaks."""
    x = kwargs.get("x")
    l = len(data)
    i1 = np.argmax(data[: l // 2])  # Location of first peak
    i2 = np.argmax(data[l // 2 :]) + l // 2

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


# This shows how to replace the default guess (which is useless) with our function guess
double_peak.guess = MethodType(guess, double_peak)

# Set the initial values with sensible starting points guessed from the data
# Fit the intensity values and add to thedata file
res = hist.lmfit(double_peak, output="report")
hist.add_column(res.init_fit, header="Initial Fit")
hist.add_column(res.best_fit, header="Best Fit")

# Plot the results as well
hist.setas = "xyyy"
profile.subplot(224)
hist.plot(fmt=["b+", "b--", "r-"])
hist.title = "Intensity histogram"
