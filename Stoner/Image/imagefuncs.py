# -*- coding: utf-8 -*-
"""Functions for manipulating Kerr (or any other) images.

All of these functions are accessible through the :class:`ImageArray` attributes e.g.:

    k=ImageArray('myfile'); k.level_image().

The first 'im' function argument is automatically added in this case.

If you want to add new functions that's great. There's a few important points:

    * Please make sure they take an image as the first argument

    * Don't give them the same name as functions from the numpy library or
          skimage library if you don't want to override them.

    * The function should not change the shape of the array. Please use crop_image
          before doing the function if you want to do that.

    * After that you're free to treat im as a ImageArray
          or numpy array, it should all behave the same.
"""
__all__ = [
    "adjust_contrast",
    "align",
    "convert",
    "correct_drift",
    "subtract_image",
    "fft",
    "filter_image",
    "gridimage",
    "hist",
    "imshow",
    "level_image",
    "normalise",
    "profile_line",
    "quantize",
    "radial_coordinates",
    "radial_profile",
    "remove_outliers",
    "rotate",
    "translate",
    "translate_limits",
    "plot_histogram",
    "threshold_minmax",
    "do_nothing",
    "denoise",
]
import warnings
import os
import io

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy import signal
from matplotlib.colors import to_rgba
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from skimage import measure, transform, filters
from PIL import Image, PngImagePlugin

from Stoner.tools import isTuple, isiterable, make_Data

# from .core import ImageArray
from ..core.base import metadataObject
from .util import sign_loss, _dtype2, _supported_types, prec_loss, dtype_range, _dtype, _scale as im_scale
from ..tools.decorators import changes_size, keep_return_type
from .widgets import LineSelect
from ..compat import string_types, get_filedialog  # Some things to help with Python2 and Python3 compatibility
from ..plot.utils import auto_fit_fontsize

try:
    from PyQt5.QtGui import QImage
    from PyQt5.QtWidgets import QApplication
except ImportError:
    QImage = None
    QApplication = None

try:  # Make OpenCV an optional import
    import cv2
except ImportError:
    cv2 = None

try:  # Make imreg_dfft2 optional
    import imreg_dft
except ImportError:
    imreg_dft = None

try:  # image_registration module
    from image_registration import chi2_shift
except ImportError:
    chi2_shift = None

IMAGE_FILES = [("Tiff File", "*.tif;*.tiff"), ("PNG files", "*.png", "Numpy Files", "*.npy")]


def _scale(coord, scale=1.0, to_pixel=True):
    """Convert pixel coordinates to scaled coordinates or visa versa.

    Args:
        coord(int,float or iterable): Coordinates to be scaled

    Keyword Arguments:
        scale(float): Microns per Pixel scale of image
        to_pixel(bool): Force the conversion to be to pixels

    Returns:
        scaled coordinates.
    """
    if isinstance(coord, int):
        if not to_pixel:
            coord = float(coord) * scale
    elif isinstance(coord, float):
        if to_pixel:
            coord = int(round(coord / scale))
    elif isiterable(coord):
        coord = tuple([_scale(c, scale, to_pixel) for c in coord])
    else:
        raise ValueError("coord should be an integer or a float or an iterable of integers and floats")
    return coord


def adjust_contrast(im, lims=(0.1, 0.9), percent=True):
    """Rescale the intensity of the image.

    Mostly a call through to skimage.exposure.rescale_intensity. The absolute limits of contrast are
    added to the metadata as 'adjust_contrast'

    Parameters
    ----------
    lims: 2-tuple
        limits of rescaling the intensity
    percent: bool
        if True then lims are the give the percentile of the image intensity
        histogram, otherwise lims are absolute

    Returns
    -------
    image: ImageArray
        rescaled image
    """
    if percent:
        vmin, vmax = np.percentile(im.view(np.ndarray), np.array(lims) * 100)
    else:
        vmin, vmax = lims[0], lims[1]
    im.metadata["adjust_contrast"] = (vmin, vmax)
    im = im.rescale_intensity(in_range=(vmin, vmax))  # clip the intensity
    return im


def _align_scharr(im, ref, **kargs):
    """Return the translation vector to shirt im to align to ref using the Scharr method."""
    scale = np.ceil(np.max(im.shape) / 500.0)
    ref1 = filters.edges.scharr(gaussian_filter(ref, sigma=scale, mode="wrap"))
    im1 = filters.edges.scharr(gaussian_filter(im, sigma=scale, mode="wrap"))
    return _align_imreg_dft(im1, ref1, **kargs)


def _align_chi2_shift(im, ref, **kargs):
    """Return the translation vector to shirt im to align to ref using the chi^2 shift method."""
    results = np.array(chi2_shift(ref, im, **kargs))
    return -results[1::-1], {}


def _align_imreg_dft(im, ref, **kargs):
    """Return the translation vector to shirt im to align to ref using the imreg_dft method."""
    constraints = kargs.pop("constraints", {"angle": [0.0, 0.0], "scale": [1.0, 0.0]})
    with warnings.catch_warnings():  # This causes a warning due to the masking
        warnings.simplefilter("ignore")
        result = imreg_dft.similarity(ref, im, constraints=constraints)
    tvec = result["tvec"]
    return np.array(tvec), result


def _align_cv2(im, ref, **kargs):  # pylint: disable=unused-argument
    """Return the translation vector to shirt im to align to ref using the cv2 method."""
    im1_gray = ref.convert("uint8", force_copy=True)
    im2_gray = im.convert("uint8", force_copy=True)

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (_, warp_matrix) = cv2.findTransformECC(
        im1_gray, im2_gray, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1
    )

    return -warp_matrix[1::-1, 2], {}


def align(im, ref, method="scharr", **kargs):
    """Use one of a variety of algroithms to align two images.

    Args:
        im (ndarray) image to align
        ref (ndarray) reference array

    Keyword Args:
        method (str or None):
            If given specifies which module to try and use.
            Options: 'scharr', 'chi2_shift', 'imreg_dft', 'cv2'
        _box (integer, float, tuple of images or floats):
            Used with ImageArray.crop to select a subset of the image to use for the aligning process.
        scale (int):
            Rescale the image and reference image by constant factor before finding the translation vector.
        prefilter (callable):
            A method to apply to the image before carrying out the translation to the align to the reference.
        **kargs (various): All other keyword arguments are passed to the specific algorithm.


    Returns
        (ImageArray or ndarray) aligned image

    Notes:
        Currently three algorithms are supported:
            - image_registration module's chi^2 shift: This uses a dft with an automatic
              up-sampling of the fourier transform for sub-pixel alignment. The metadata
              key *chi2_shift* contains the translation vector and errors.
            - imreg_dft module's similarity function. This implements a full scale, rotation, translation
              algorithm (by default constrained for just translation). It's unclear how much sub-pixel translation
              is accommodated.
            - cv2 module based affine transform on a gray scale image.
              from: http://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    """
    # To be consistent with x-y coordinate systems
    align_methods = {
        "scharr": (_align_scharr, imreg_dft),
        "chi2_shift": (_align_chi2_shift, chi2_shift),
        "imreg_dft": (_align_imreg_dft, imreg_dft),
        "cv2": (_align_cv2, cv2),
    }
    cls = type(im)
    for meth in list(align_methods.keys()):
        mod = align_methods[meth][1]
        if mod is None:
            del align_methods[meth]
    method = method.lower()
    if not len(align_methods):
        raise ImportError("align requires one of imreg_dft, chi2_shift or cv2 modules to be available.")
    if method not in align_methods:
        raise ValueError(f"{method} is not available either because it is not recognised or there is a missing module")

    if "box" in kargs or "_box" in kargs:
        box = kargs.pop("box", kargs.pop("_box"))
        if not isiterable(box):
            box = [box]
        working = im.crop(*box, copy=True)
        if ref.shape != working.shape:
            ref = ref.view(cls).crop(*box, copy=True)
    else:
        working = im

    scale = kargs.pop("scale", None)
    mode = kargs.pop("mode", "mirror")
    cval = kargs.pop("cval", im.mean())
    if mode == "mean":
        mode = "constant"
        cval = im.mean()

    if scale:
        working = working.rescale(scale, order=3)
        ref = transform.rescale(ref, scale, order=3)

    prefilter = kargs.pop("prefilter", True)
    try:
        tvec, data = align_methods[method][0](working, ref, **kargs)
    except (ValueError, TypeError):
        tvec = (0, 0)
        data = im

    data.pop("timg", None)

    if scale:
        tvec /= scale
    new_im = im.shift((tvec[1], tvec[0]), prefilter=prefilter, mode=mode, cval=cval)
    new_im.metadata.update(data)
    new_im["tvec"] = tuple(tvec)
    new_im["translation_limits"] = new_im.translate_limits("tvec")
    return new_im


def convert(image, dtype, force_copy=False, uniform=False, normalise=True):
    """Convert an image to the requested data-type.

    Warnings are issued in case of precision loss, or when negative values
    are clipped during conversion to unsigned integer types (sign loss).

    Floating point values are expected to be normalized and will be clipped
    to the range [0.0, 1.0] or [-1.0, 1.0] when converting to unsigned or
    signed integers respectively.

    Numbers are not shifted to the negative side when converting from
    unsigned to signed integer types. Negative values will be clipped when
    converting to unsigned integers.

    Parameters
    ----------
    image : ndarray
        Input image.
    dtype : dtype
        Target data-type.
    force_copy : bool
        Force a copy of the data, irrespective of its current dtype.
    uniform : bool
        Uniformly quantize the floating point range to the integer range.
        By default (uniform=False) floating point values are scaled and
        rounded to the nearest integers, which minimizes back and forth
        conversion errors.
    normalise : bool
        When converting from int types to float normalise the resulting array
        by the maximum allowed value of the int type.

    References
    ----------
    (1) DirectX data conversion rules.
        http://msdn.microsoft.com/en-us/library/windows/desktop/dd607323%28v=vs.85%29.aspx
    (2) Data Conversions.
        In "OpenGL ES 2.0 Specification v2.0.25", pp 7-8. Khronos Group, 2010.
    (3) Proper treatment of pixels as integers. A.W. Path.
        In "Graphics Gems I", pp 249-256. Morgan Kaufmann, 1990.
    (4) Dirty Pixels. J. Blinn.
        In "Jim Blinn's corner: Dirty Pixels", pp 47-57. Morgan Kaufmann, 1998.

    """
    image = np.asarray(image)
    dtypeobj = np.dtype(dtype)
    dtypeobj_in = image.dtype
    dtype = dtypeobj.type
    dtype_in = dtypeobj_in.type

    if dtype_in == dtype:
        if force_copy:
            image = image.clone
        return image

    if not (dtype_in in _supported_types and dtype in _supported_types):
        raise ValueError(f"can not convert {dtype_in} to {dtype}.")

    if dtypeobj.kind == "b":
        # to binary image
        if dtypeobj_in.kind in "fi":
            sign_loss(dtype_in, dtypeobj)
        prec_loss(dtypeobj_in, dtypeobj)
        return image > dtype_in(dtype_range[dtype_in][1] / 2)

    if dtypeobj_in.kind == "b":
        # from binary image, to float and to integer
        result = np.where(~image, *dtype_range[dtype])
        return result

    imax = imin = imax_in = imin_in = None
    if dtypeobj.kind in "ui":
        imin = np.iinfo(dtype).min
        imax = np.iinfo(dtype).max
    if dtypeobj_in.kind in "ui":
        imin_in = np.iinfo(dtype_in).min
        imax_in = np.iinfo(dtype_in).max

    if dtypeobj_in.kind == "f":
        if np.min(image) < -1.0 or np.max(image) > 1.0:
            raise ValueError("Images of type float must be between -1 and 1.")
        if dtypeobj.kind == "f":
            # floating point -> floating point
            if dtypeobj_in.itemsize > dtypeobj.itemsize:
                prec_loss(dtypeobj_in, dtypeobj)
            return image.astype(dtype)

        # floating point -> integer
        prec_loss(dtypeobj_in, dtypeobj)
        # use float type that can represent output integer type
        image = np.array(image, _dtype(dtypeobj.itemsize, dtype_in, np.float32, np.float64))
        if not uniform:
            if dtypeobj.kind == "u":
                image *= imax
            elif (
                dtypeobj_in.itemsize <= dtypeobj.itemsize and dtypeobj.itemsize == 8
            ):  # f64->int64 needs care to avoid overruns!
                image *= 2**54  # float64 has 52bits of mantissa, so this will avoid precission loss for a +/-1 range
                np.rint(image, out=image)
                image = image.astype(dtype)
                np.clip(image, -(2**54), 2**54 - 1, out=image)
                image *= 512
            else:
                image *= imax - imin
                image -= 1.0
                image /= 2.0
                np.rint(image, out=image)
                np.clip(image, imin, imax, out=image)
        elif dtypeobj.kind == "u":
            image *= imax + 1
            np.clip(image, 0, imax, out=image)
        else:
            image *= (imax - imin + 1.0) / 2.0
            np.floor(image, out=image)
            np.clip(image, imin, imax, out=image)
        return image.astype(dtype)

    if dtypeobj.kind == "f":
        # integer -> floating point
        if dtypeobj_in.itemsize >= dtypeobj.itemsize:
            prec_loss(dtypeobj_in, dtypeobj)
        # use float type that can exactly represent input integers
        image = np.array(image, _dtype(dtypeobj_in.itemsize, dtype, np.float32, np.float64))
        if normalise:  # normalise floats by maximum value of int type
            if dtypeobj_in.kind == "u":
                image /= imax_in
                # DirectX uses this conversion also for signed ints
                # if imin_in:
                #    np.maximum(image, -1.0, out=image)
            else:
                image *= 2.0
                image += 1.0
                image /= imax_in - imin_in
        return image.astype(dtype)

    if dtypeobj_in.kind == "u":
        if dtypeobj.kind == "i":
            # unsigned integer -> signed integer
            image = im_scale(image, 8 * dtypeobj_in.itemsize, 8 * dtypeobj.itemsize - 1, dtypeobj_in, dtypeobj)
            return image.view(dtype)
        # unsigned integer -> unsigned integer
        return im_scale(image, 8 * dtypeobj_in.itemsize, 8 * dtypeobj.itemsize, dtypeobj_in, dtypeobj)

    if dtypeobj.kind == "u":
        # signed integer -> unsigned integer
        sign_loss(dtype_in, dtypeobj)
        image = im_scale(image, 8 * dtypeobj_in.itemsize - 1, 8 * dtypeobj.itemsize, dtypeobj_in, dtypeobj)
        result = np.empty(image.shape, dtype)
        np.maximum(image, 0, out=result, dtype=image.dtype, casting="unsafe")
        return result

    # signed integer -> signed integer
    if dtypeobj_in.itemsize > dtypeobj.itemsize:
        return im_scale(image, 8 * dtypeobj_in.itemsize - 1, 8 * dtypeobj.itemsize - 1, dtypeobj_in, dtypeobj)
    image = image.astype(_dtype2("i", dtypeobj.itemsize * 8))
    image -= imin_in
    image = im_scale(image, 8 * dtypeobj_in.itemsize, 8 * dtypeobj.itemsize, dtypeobj_in, dtypeobj, copy=False)
    image += imin
    return image.astype(dtype)


def correct_drift(im, ref, **kargs):
    """Align images to correct for image drift.

    Args:
        ref (ImageArray): Reference image with assumed zero drift

    Keyword Arguments:
        threshold (float): threshold for detecting imperfections in images
            (see skimage.feature.corner_fast for details)
        upsample_factor (float): the resolution for the shift 1/upsample_factor pixels registered.
            see skimage.feature.register_translation for more details
        box (sequence of 4 ints): defines a region of the image to use for identifyign the drift
            defaults to the whol image. Use this to avoid drift calculations being confused by
            the scale bar/annotation region.
        do_shift (bool): Shift the image, or just calculate the drift and store in metadata (default True, shit)

    Returns:
        A shifted image with the image shift added to the metadata as 'correct drift'.

    Detects common features on the images and tracks them moving.
    Adds 'drift_shift' to the metadata as the (x,y) vector that translated the
    image back to it's origin.
    """
    do_shift = kargs.pop("do_shift", True)
    kargs["scale"] = kargs.pop("upscale", kargs.get("scale", 5.0))
    kargs.setdefault("meothd", "scharr")

    ret = align(im, ref, **kargs)
    if do_shift:
        im = ret
    im["correct_drift"] = -np.array(ret["tvec"])[::-1]

    return im


def subtract_image(im, background, contrast=16, clip=True, offset=0.5):
    """Subtract a background image from the ImageArray.

    Multiply the contrast by the contrast parameter.
    If clip is on then clip the intensity after for the maximum allowed data range.
    """
    im = im.asfloat(normalise=False, clip_negative=False)
    im = contrast * (im - background) + offset
    if clip:
        im = im.clip_intensity()
    return im


def fft(im, shift=True, phase=False, remove_dc=False, gaussian=None, window=None):
    """Perform a 2d fft of the image and shift the result to get zero frequency in the centre.

    Keyword Args:
        shift (bool):
            Shift the fft so that zero order is in the centre of the image. Default True
        phase (bool, None):
            If true, return the phase angle rather than the magnitude if False. If None, return the raw fft.
            Default False - return magnitude of fft.
        remove_dc(bool):
            Replace the points around the dc offset with the mean of the fft to avoid dc offset artefacts.
            Default False
        gaussian (None or float):
            Apply a gaussian blur to the fft where this parameter is the width of the blue in px. Default None for off.
        window (None or str):
            If not None (default) the image is multiplied by the given window function before the fft is calculated.
            This avpoids leaking some signal into the higher frequency bands due to discontinuities at the image edges.

    Return:
        fft of the image, preserving metadata.
    """
    if window:
        window = filters.window(window, im.shape)
        im = im.clone * window
    r = np.fft.fft2(im)

    if remove_dc:
        fill = r.mean()
        r[0, 0] = fill
        r[-1, 0] = fill
        r[-1, -1] = fill
        r[0, -1] = fill
    if shift:
        r = np.fft.fftshift(r)
    if phase is None:
        pass
    if not phase:
        r = np.abs(r)
    else:
        r = np.angle(r)
    r = r.view(type(im))
    if isinstance(gaussian, (float, int)):
        r.gaussian(gaussian)

    r.metadata.update(im.metadata)
    return r


def filter_image(im, sigma=2):
    """Alias for skimage.filters.gaussian."""
    return im.gaussian(sigma=sigma)


def gridimage(im, points=None, xi=None, method="linear", fill_value=None, rescale=False):
    """Use :py:func:`scipy.interpolate.griddata` to shift the image to a regular grid of coordinates.

    Args:
        points (tuple of (x-co-ords,yco-ordsa)):
            The actual sampled data coordinates
        xi (tupe of (2D array,2D array)):
            The regular grid coordinates (as generated by e.g. :py:func:`np.meshgrid`)

    Keyword Arguments:
        method ("linear","cubic","nearest"):
            how to interpolate, default is linear
        fill_value (folat, Callable, None):
            What to put when the coordinates go out of range (default is None). May be a callable
            in which case the initial image is presented as the only argument. If None, use the mean value.
        rescale (bool):
            If the x and y coordinates are very different in scale, set this to True.

    Returns:
        A copy of the modified image. The image data is interpolated and metadata kets "actual_x","actual_y","sample_
        x","samp[le_y" are set to give coordinates of new grid.

    Notes:
        If points and or xi are missed out then we try to construct them from the metadata. For points, the metadata
        keys "actual-x" and "actual_y" are looked for and for xi, the metadata keys "sample_x" and "sample_y" are
        used. These are set, for example, by the :py:class:`Stoner.HDF5.SXTMImage` loader if the interformeter stage
        data was found in the file.

        The metadata used in this case is then adjusted as well to ensure that repeated application of this method
        doesn't change the image after it has been corrected once.
    """
    if points is None:
        points = np.column_stack((im["actual_x"].ravel(), im["actual_y"].ravel()))
    if xi is None:
        xi = xi = (im["sample_x"], im["sample_y"])

    if fill_value is None:
        fill_value = np.mean

    if callable(fill_value):
        fill_value = fill_value(im)

    im2 = griddata(points, im.ravel(), xi, method, fill_value, rescale)
    im2 = type(im)(im2)
    im2.metadata = im.metadata
    im2.metadata["actual_x"] = xi[0]
    im2.metadata["actual_y"] = xi[1]
    return im2


def hist(im, *args, **kargs):
    """Pass through to :py:func:`matplotlib.pyplot.hist` function."""
    if isinstance(im, np.ma.MaskedArray):
        im_data = im[~im.mask]
    else:
        im_data = im.ravel()
    counts, edges = np.histogram(im_data, *args, **kargs)
    centres = (edges[1:] + edges[:-1]) / 2
    new = make_Data(np.column_stack((centres, counts)))
    new.column_headers = ["Intensity", "Frequency"]
    new.setas = "xy"
    return new


def imshow(im, **kwargs):
    """Quickly plot of image.

    Keyword Arguments:
        figure (int, str or matplotlib.figure):
            if int then use figure number given, if figure is 'new' then create a new figure, if None then use
            whatever default figure is available
        show_axis (bool):
            If True, show the axis otherwise don't (default)'
        title (str,None,False):
            Title for plot - defaults to False (no title). None will take the title from the filename if present
        title_args (dict):
            Arguments to pass to the title function to control formatting.
        cmap (str,matplotlib.cmap):
            Colour scheme for plot, defaults to gray

    Any masked areas are set to NaN which stops them being plotted at all.
    """
    figure = kwargs.pop("figure", "new")
    ax = kwargs.pop("ax", None)
    # Get a title - from keyword argument, from title attr or filename attr
    title = os.path.split(getattr(im, "title", getattr(im, "filename", "")))[-1]
    title = kwargs.pop("title", title)
    title_args = kwargs.pop("title_args", {})
    cmap = kwargs.pop("cmap", "gray")
    mask_alpha = kwargs.pop("mask_alpha", getattr(im, "_mask_alpha", 0.5))
    mask_col = kwargs.pop("mask_color", getattr(im, "_mask_color", "red"))
    if isinstance(cmap, string_types):
        cmap = getattr(cm, cmap)
    im_data = im
    if figure is not None and isinstance(figure, int):
        fig = plt.figure(figure)
    elif figure is not None and figure == "new":
        fig = plt.figure()
    elif figure is not None:  # matplotlib.figure instance
        fig = plt.figure(figure.number)
    else:
        fig = plt.figure()
    if ax is not None:
        plt.sca(ax)
    else:
        ax = plt.gca()
    ax.imshow(im_data.view(np.ndarray), cmap=cmap, **kwargs)
    if np.ma.is_masked(im):
        mask_col = list(to_rgba(mask_col))
        mask_col[-1] = mask_alpha
        mask_data = np.zeros(im.shape + (4,))
        mask_data[im.mask] = mask_col
        ax.imshow(mask_data)

    if title is None:
        if "filename" in im.metadata.keys():
            title = os.path.split(im["filename"])[1]
        elif hasattr(im, "filename"):
            title = os.path.split(im.filename)[1]
        else:
            title = " "
    elif isinstance(title, bool) and not title:
        title = ""

    txt = ax.set_title(title, **title_args)
    if not title_args.get("fontdict", {}).get("fontsize", False):
        gs = ax.get_gridspec()
        width = 0.9 / gs.ncols
        height = 0.09 / gs.nrows
        auto_fit_fontsize(txt, width, height)
    ax.axis("on" if kwargs.get("show_axis", False) else "off")
    try:
        im["ax"] = ax
        im["fig"] = fig
    except IndexError:
        pass

    if QImage is None:  # No Qt5
        return fig

    def add_figure_to_clipboard(event):
        if event.key == "ctrl+c":
            with io.BytesIO() as buffer:
                fig.savefig(buffer)
                QApplication.clipboard().setImage(QImage.fromData(buffer.getvalue()))

    fig.canvas.mpl_connect("key_press_event", add_figure_to_clipboard)
    return fig


def level_image(im, poly_vert=1, poly_horiz=1, box=None, poly=None, mode="clip"):
    """Subtract a polynomial background from image.

    Keyword Arguments:
        poly_vert (int): fit a polynomial in the vertical direction for the image of order
            given. If 0 do not fit or subtract in the vertical direction
        poly_horiz (int): fit a polynomial of order poly_horiz to the image. If 0 given
            do not subtract
        box (array, list or tuple of int): [xmin,xmax,ymin,ymax] define region for fitting. IF None use entire
            image
        poly (list or None): [pvert, phoriz] pvert and phoriz are arrays of polynomial coefficients
            (highest power first) to subtract in the horizontal and vertical
            directions. If None function defaults to fitting its own polynomial.
        mode (str): Either 'clip' or 'norm' - specifies how to handle intensitry values that end up being outside
            of the accepted range for the image.

    Returns:
        A new copy of the processed images.

    Fit and subtract a background to the image. Fits a polynomial of order
    given in the horizontal and vertical directions and subtracts. If box
    is defined then level the *entire* image according to the
    gradient within the box. The polynomial subtracted is added to the
    metadata as 'poly_vert_subtract' and 'poly_horiz_subtract'
    """
    if box is None:
        box = im.max_box
    cim = im.crop(box=box)
    (vertl, horizl) = cim.shape
    p_horiz = 0
    p_vert = 0
    if poly_horiz > 0:
        comp_vert = np.average(cim, axis=0)  # average (compress) the vertical values
        if poly is not None:
            p_horiz = poly[0]
        else:
            p_horiz = np.polyfit(np.arange(horizl), comp_vert, poly_horiz)  # fit to the horizontal
            av = np.average(comp_vert)  # get the average pixel height
            p_horiz[-1] = p_horiz[-1] - av  # maintain the average image height
        horizcoord = np.indices(im.shape)[1]  # now apply level to whole image
        for i, c in enumerate(p_horiz):
            im = im - c * horizcoord ** (len(p_horiz) - i - 1)
    if poly_vert > 0:
        comp_horiz = np.average(cim, axis=1)  # average the horizontal values
        if poly is not None:
            p_vert = poly[1]
        else:
            p_vert = np.polyfit(np.arange(vertl), comp_horiz, poly_vert)
            av = np.average(comp_horiz)
            p_vert[-1] = p_vert[-1] - av  # maintain the average image height
        vertcoord = np.indices(im.shape)[0]
        for i, c in enumerate(p_vert):
            im = im - c * vertcoord ** (len(p_vert) - i - 1)
    im.metadata["poly_sub"] = (p_horiz, p_vert)
    if mode == "clip":
        im = im.clip_intensity()  # saturate any pixels outside allowed range
    elif mode == "norm":
        im = im.normalise()
    return im


def normalise(im, scale=None, sample=False, limits=(0.0, 1.0), scale_masked=False):
    """Norm alise the data to a fixed scale.

    Keyword Arguments:
        scale (2-tuple):
            The range to scale the image to, defaults to -1 to 1.
        saple (box):
            Only use a section of the input image to calculate the new scale over. Default is False - whole image
        limits (low,high):
            Take the input range from the *high* and *low* fraction of the input when sorted.
        scale_masked (bool):
            If True then the masked region is also scaled, otherwise any masked region is ignored. Default, False.


    Returns:
        A scaled version of the data. The ndarray min and max methods are used to allow masked images
        to be operated on only on the unmasked areas.

    Notes:
        The *sample* keyword controls the area in which the range of input values is calculated, the actual scaling is
        done on the whole image.

        The *limits* parameter is used to set the input scale being normalised from - if an image has a few outliers
        then this setting can be used to clip the input range before normalising. The parameters in the limit are the
        values at the *low* and *high* fractions of the cumulative distribution functions.
    """
    mask = im.mask
    cls = type(im)
    im = im.astype(float)
    if scale is None:
        scale = (-1.0, 1.0)
    section = im[im._box(sample)]

    section = section[~section.mask]
    if limits != (0.0, 1.0):
        low, high = limits
        low = np.sort(section.ravel())[int(low * section.size)]
        high = np.sort(section.ravel())[int(high * section.size)]
        im.clip_intensity(limits=(low, high))
    else:
        high = section.max()
        low = section.min()

    if not isTuple(scale, float, float, strict=False):
        raise ValueError("scale should be a 2-tuple of floats.")
    scaled = (im.data - low) / (high - low)
    delta = scale[1] - scale[0]
    offset = scale[0]
    scaled = scaled * delta + offset
    if not scale_masked:
        im = np.where(im.mask, im, scaled).view(cls)
    else:
        im = scaled.view(cls)
    im.mask = mask
    return im


def clip_neg(im):
    """Clip negative pixels to 0.

    Most useful for float where pixels above 1 are reduced to 1.0 and -ve pixels
    are changed to 0.
    """
    im[im < 0] = 0
    return im


def profile_line(img, src=None, dst=None, linewidth=1, order=1, mode="constant", cval=0.0, constrain=True, **kargs):
    """Wrap sckit-image method of the same name to get a line_profile.

    Parameters:
        img(ImageArray):
            Image data to take line section of

    Keyword Parameters:
        src, dst (2-tuple of int or float):
            start and end of line profile. If the coordinates
            are given as integers then they are assumed to be pxiel coordinates, floats are
            assumed to be real-space coordinates using the embedded metadata.
        linewidth (int):
            the wideth of the profile to be taken.
        order (int 1-3):
            Order of interpolation used to find image data when not aligned to a point
        mode (str):
            How to handle data outside of the image.
        cval (float):
            The constant value to assume for data outside of the image is mode is "constant"
        constrain (bool):
            Ensure the src and dst are within the image (default True).
        no_scale (bool):
            Do not attempt to scale values by the image scale and work in pixels throughout. (default False)


    Returns:
        A :py:class:`Stoner.Data` object containing the line profile data and the metadata from the image.
    """
    scale = 1.0 if kargs.get("no_scale", False) else img.get("MicronsPerPixel", 1.0)
    r, c = img.shape
    fast_mode = False
    if src is None and dst is None:
        if "x" in kargs:
            src = (0, kargs["x"])
            dst = (r, kargs["x"])
            fast_mode = kargs.get("no_scale", False)
        if "y" in kargs:
            src = (kargs["y"], 0)
            dst = (kargs["y"], c)
            fast_mode = kargs.get("no_scale", False)
        if src is None and dst is None:
            src, dst = LineSelect()(img)
    if isinstance(src, float):
        src = (src, src)
    if isinstance(dst, float):
        dst = (dst, dst)
    dst = _scale(dst, scale)
    src = _scale(src, scale)
    if not isTuple(src, int, int):
        raise ValueError("src coordinates are not a 2-tuple of ints.")
    if not isTuple(dst, int, int):
        raise ValueError("dst coordinates are not a 2-tuple of ints.")

    if constrain:
        fix = lambda x, mx: int(round(sorted([0, x, mx])[1]))
        r, c = img.shape
        src = list(src)
        src = (fix(src[0], r), fix(src[1], c))
        dst = (fix(dst[0], r), fix(dst[1], c))

    if fast_mode:
        if "x" in kargs:
            result = img[:, src[1]]
            points = np.vstack((np.ones(img.shape[0]) * src[1], np.arange(img.shape[0])))
        else:
            result = img[src[0], :]
            points = np.vstack((np.arange(img.shape[1]), np.ones(img.shape[1]) * src[0]))
    else:
        result = measure.profile_line(img, src, dst, linewidth, order, mode, cval)
        points = measure.profile._line_profile_coordinates(src, dst, linewidth)[:, :, 0]
    ret = make_Data()
    ret.data = points.T
    ret.setas = "xy"
    x, y = points
    x -= x[0]
    y -= y[0]
    ret &= np.sqrt(x**2 + y**2) * scale
    ret &= result
    ret.column_headers = ["X", "Y", "Distance", "Intensity"]
    ret.setas = "..xy"
    ret.metadata = img.metadata.copy()
    return ret


def radial_coordinates(im, centre=(None, None), pixel_size=(1, 1), angle=False):
    """Rerurn a map of the radial coordinates of an image from a given centre, with adjustments for pixel size.

    Keyword Arguments:
        centre (2-tuple):
            Coordinates of centre point in terms of the original pixels. Defaults to(None,None) for the middle of the
            image.
        pixel_size (2-tuple):
            The size of one pixel in (dx by dy) - defaults to 1,1
        angle (bool, None):
            Whether to return the angles (in radians, True), distances (False) o a complex number (None).

    Returns:
        An array of the same class as the input, but with values corresponding to the radial coordinates.
    """
    cx, cy = centre
    r, c = im.shape
    dx, dy = pixel_size
    cx = c / 2 if cx is None else cx
    cy = r / 2 if cy is None else cy
    x_r = dx * (np.linspace(0, c - 1, c) - cx)
    y_r = dy * (np.linspace(0, r - 1, r) - cy)
    Y, X = np.meshgrid(x_r, y_r)
    Z = -Y + (0 + 1j) * X
    if angle is None:
        pass
    elif not angle:
        Z = np.abs(Z)
    else:
        Z = np.angle(Z)
    Z = Z.view(type(im))
    Z.metadata = im.metadata
    return Z


def radial_profile(im, angle=None, r=None, centre=(None, None), pixel_size=(1, 1)):
    """Extract a radial  profile line from an image.

    Keyword Parameters:
        angle (float, tuple, None):
            Select the radial angle to include:
                - float selects a single angle
                - tuple selects a range of angles
                - None integrates over all angles
        r (array, None):
            Edges of the bins in the radual direction - will return r.size-1 points. Default is None which uses the
            minimum r value found on the edges of the image.
        centre (2-tuple):
            Coordinates of centre point in terms of the original pixels. Defaults to(None,None) for the middle of the
            image.
        pixel_size (2-tuple):
            The size of one pixel in (dx by dy) - defaults to 1,1

    Returns:
        (Data):
            A py:class:`Stoner.Data` object with a column for r and columns for mean, std, and number of pixels.
    """
    coords = im.radial_coordinates(centre=centre, pixel_size=pixel_size, angle=None)
    if r is None:  # Identify the minimum edge value
        edges = np.append(coords[:, 0], coords[-1, :])
        edges = np.append(edges, coords[:, -1])
        edges = np.append(edges, coords[0, :])
        r_limit = np.abs(edges).min()
        r = np.linspace(0, np.ceil(r_limit), int(np.ceil(r_limit) + 1))
    r_l = r[:-1]
    r_h = r[1:]
    r_m = (r_l + r_h) / 2
    if angle is None:
        angle_select = np.ones_like(coords).astype(bool)
    elif isinstance(angle, tuple):
        angle_low = np.angle(coords) >= angle[0]
        angle_high = np.angle(coords) <= angle[1]
        angle_select = np.logical_and(angle_low, angle_high)
    elif isinstance(angle, (int, float)):
        angle_select = np.isclose(np.angle(coords), angle)
    else:
        raise TypeError(f"angle should be a float, tuple of two floats or None not a {type(angle)}")
    ret = make_Data()
    for low, high, mid in zip(r_l, r_h, r_m):
        r_select = np.logical_and(np.abs(coords) >= low, np.abs(coords) < high)
        data = im[np.logical_and(r_select, angle_select)]
        if data.size == 0:
            continue
        if data.size == 1:
            avg = data[0]
            std = np.nan
        else:
            avg = data.mean()
            std = data.std()
        num = data.size
        ret += np.array([low, mid, high, avg, std, num])
    ret.column_headers = ["Low_r", "r", "high_r", "mean", "std", "number"]
    ret.setas = ".x.ye."
    ret.metadata = im.metadata
    ret.filename = im.filename[:-4] + "_profile.txt"
    return ret


def quantize(im, output, levels=None):
    """Quantise the image data into fixed levels given by a mapping.

    Args:
        output (list,array,tuple): Output levels to return.

    Keyword Arguments:
        levels (list, array or None): The input band markers. If None is constructed from the data.

    The number of levels should be one less than the number of output levels given.

    Notes:
        The routine will ignore all masked pixels and will preserve the mask.
    """
    section = im[~im.mask]
    mask = im.mask
    lmin, lmax = section.min(), section.max()  # Dudge to ensure that the bottom and top elements are included.
    delta = (lmax - lmin) / 100

    if levels is None:
        levels = np.linspace(lmin - delta, lmax + delta, len(output) + 1)
    elif len(levels) == len(output) + 1:
        pass
    elif len(levels) == len(output) - 1:
        lvl = np.zeros(len(output) + 1)
        lvl[1:-1] = levels
        lvl[0] = section.min() - delta
        lvl[-1] = section.max() + delta
        levels = lvl
    else:
        raise RuntimeError(f"{len(output)} output levels and {len(levels)} input levels")

    ret = im.clone
    ret.mask = False
    for lvl, lvh, val in zip(levels[:-1], levels[1:], output):
        select = np.logical_and(np.less_equal(im, lvh), np.greater(im, lvl))
        ret[select] = val
    ret[mask] = im.view(np.ndarray)[mask]  # Put back the original image values where they are masked.
    ret.mask = mask  # restore mask
    return ret


def remove_outliers(im, percentiles=(0.01, 0.99), replace=None):
    """Find values of the data that are beyond a percentile of the overall distribution and replace them.

    Keyword Parameters:
        percentile (2 tuple):
            Fraction percentiles to consider to be outliers (default is (0.01,0.99) for 1% limits)
        replace (2 tuple or None):
            Values to set outliers to. If None, then the pixel values at the percentile limits are used.

    Returns:
        (ndarray):
            The modified array.

    Use this method if you have an image with a small number of pixels with extreme values that are
    out of range.
    """
    from scipy.interpolate import interp1d

    cdf, bins = im.cumulative_distribution(nbins=min(1000, im.count()))
    cdf = interp1d(cdf, bins, kind="linear")
    limits = [cdf(x) for x in percentiles]
    if replace is None:
        replace = limits
    im[im < limits[0]] = replace[0]
    im[im > limits[1]] = replace[1]
    return im


def rotate(im, angle, resize=False, center=None, order=1, mode="constant", cval=0, clip=True, preserve_range=False):
    """Rotate image by a certain angle around its center.

    Parameters:
        angle  (float):
            Rotation angle in **radians** in clockwise direction.

    Keyword Parameters:
        resize (bool):
            Determine whether the shape of the output image will be automatically
            calculated, so the complete rotated image exactly fits. Default is
            False.
        center (iterable of length 2):
            The rotation center. If ``center=None``, the image is rotated around
            its center, i.e. ``center=(cols / 2 - 0.5, rows / 2 - 0.5)``.  Please
            note that this parameter is (cols, rows), contrary to normal skimage
            ordering.
        order (int):
            The order of the spline interpolation, default is 1. The order has to
            be in the range 0-5. See `skimage.transform.warp` for detail.
        mode ({'constant', 'edge', 'symmetric', 'reflect', 'wrap'}):
            Points outside the boundaries of the input are filled according
            to the given mode.  Modes match the behaviour of `numpy.pad`.
        cval  (float):
            Used in conjunction with mode 'constant', the value outside
            the image boundaries.
        clip (bool):
            Whether to clip the output to the range of values of the input image.
            This is enabled by default, since higher order interpolation may
            produce values outside the given input range.
        preserve_range (bool):
            Whether to keep the original range of values. Otherwise, the input
            image is converted according to the conventions of `Stpomer.Image.ImageArray.as_float`.

    Returns:
        (ImageFile/ImageArray):
            Rotated image

    Notes:
        (pass through to the skimage.transform.warps.rotate function)
    """
    ang = np.rad2deg(-angle)
    data = transform.rotate(
        im,
        ang,
        resize=resize,
        center=center,
        order=order,
        mode=mode,
        cval=cval,
        clip=clip,
        preserve_range=preserve_range,
    )
    print(type(im), type(data))
    ret = data.view(type(im))
    try:
        ret.metadata = im.metadata
        ret.metadata["transform:rotation"] = angle
    except AttributeError:
        pass
    return ret


def sgolay2d(img, points=15, poly=1, derivative=None):
    """Implements a 2D Savitsky Golay Filter for a 2D array (e.g. image).

    Arguments:
        img (ImageArray or ImageFile):
            image to be filtered

    Keyword Arguments:
        points (int):
            The number of points in the window aperture. Must be an odd number. (default 15)
        poly (int):
            Degree of polynomial to use in the filter. (default 1)
        derivative (str or None):
            Type of defivative to calculate. Can be:
                None - smooth only (default)
                "x","y" - calculate dIntentity/dx or dIntensity/dy
                "both" - calculate the full derivative and return magnitud and angle.

    ReturnsL
        (imageArray or ImageFile):
            filtered image.

    Raises:
        ValueError if points, order or derivative are incorrect.

    Notes:
        Adapted from code on the scipy cookbook : https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    """
    # number of terms in the polynomial expression
    n_terms = (poly + 1) * (poly + 2) / 2.0

    if points % 2 == 0:
        raise ValueError("window_size must be odd")

    if points**2 < n_terms:
        raise ValueError("order is too high for the window size")

    half_size = points // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [(k - n, n) for k in range(poly + 1) for n in range(k + 1)]

    # coordinates of points
    ind = np.arange(-half_size, half_size + 1, dtype=np.float64)
    dx = np.repeat(ind, points)
    dy = np.tile(ind, [points, 1]).reshape(points**2)

    # build matrix of system of equation
    A = np.empty((points**2, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx ** exp[0]) * (dy ** exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = img.shape[0] + 2 * half_size, img.shape[1] + 2 * half_size
    Z = np.zeros((new_shape))
    # top band
    band = img[0, :]
    Z[:half_size, half_size:-half_size] = band - np.abs(np.flipud(img[1 : half_size + 1, :]) - band)
    # bottom band
    band = img[-1, :]
    Z[-half_size:, half_size:-half_size] = band + np.abs(np.flipud(img[-half_size - 1 : -1, :]) - band)
    # left band
    band = np.tile(img[:, 0].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs(np.fliplr(img[:, 1 : half_size + 1]) - band)
    # right band
    band = np.tile(img[:, -1].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, -half_size:] = band + np.abs(np.fliplr(img[:, -half_size - 1 : -1]) - band)
    # central band
    Z[half_size:-half_size, half_size:-half_size] = img

    # top left corner
    band = img[0, 0]
    Z[:half_size, :half_size] = band - np.abs(np.flipud(np.fliplr(img[1 : half_size + 1, 1 : half_size + 1])) - band)
    # bottom right corner
    band = img[-1, -1]
    Z[-half_size:, -half_size:] = band + np.abs(
        np.flipud(np.fliplr(img[-half_size - 1 : -1, -half_size - 1 : -1])) - band
    )

    # top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] = band - np.abs(np.flipud(Z[half_size + 1 : 2 * half_size + 1, -half_size:]) - band)
    # bottom left corner
    band = Z[-half_size:, half_size].reshape(-1, 1)
    Z[-half_size:, :half_size] = band - np.abs(np.fliplr(Z[-half_size:, half_size + 1 : 2 * half_size + 1]) - band)

    # solve system and convolve
    if derivative is None:
        m = np.linalg.pinv(A)[0].reshape((points, -1))
        ret = signal.fftconvolve(Z, m, mode="valid").view(type(img))
    elif derivative == "x":
        c = np.linalg.pinv(A)[1].reshape((points, -1))
        ret = signal.fftconvolve(Z, -c, mode="valid").view(type(img))
    elif derivative == "y":
        r = np.linalg.pinv(A)[2].reshape((points, -1))
        ret = signal.fftconvolve(Z, -r, mode="valid").view(type(img))
    elif derivative == "both":
        c = np.linalg.pinv(A)[1].reshape((points, -1))
        r = np.linalg.pinv(A)[2].reshape((points, -1))
        ret = signal.fftconvolve(Z, -r, mode="valid"), signal.fftconvolve(Z, -c, mode="valid").view(type(img))
    else:
        raise ValueError(f"Unknown derivative mode {derivative}")
    ret.metadata.update(img.metadata)
    return ret


def span(im):
    """Return the minimum and maximum values in the image."""
    return np.min(im), np.max(im)


def translate(im, translation, add_metadata=False, order=3, mode="wrap", cval=None):
    """Translate the image.

    Areas lost by move are cropped, and areas gained are made black (0)
    The area not lost or cropped is added as a metadata parameter
    'translation_limits'

    Args:
        translate (2-tuple):
            translation (x,y)

    Keyword Arguments:
        add_metadata (bool):
            Record the shift in the image metadata order (int): Interpolation order (default, 3, bi-cubic)
        mode (str):
            How to handle points outside the original image. See :py:func:`skimage.transform.warp`.  Defaults to "wrap"
        cval (float):
            The value to fill with if *mode* is constant. If not specified or None, defaults to the mean pixcel value.

    Returns:
        im (ImageArray): translated image
    """
    translation = [-x for x in translation]
    trans = transform.SimilarityTransform(translation=translation)
    if cval is None:
        cval = im.mean()
    im = im.warp(trans, order=order, mode=mode, cval=cval)
    if add_metadata:
        im.metadata["translation"] = translation
        im.metadata["translation_limits"] = translate_limits(im, translation)
    return im


def translate_limits(im, translation, reverse=False):
    """Find the limits of an image after a translation.

    After using ImageArray.translate some areas will be black,
    this finds the max area that still has original pixels in

    Args:
        translation: 2-tuple
            the (x,y) translation applied to the image

    Keyword Args:
        reverse (bool):
            whether to reverse the translation vector (default False, no)

    Returns:
        limits: 4-tuple
            (xmin,xmax,ymin,ymax) the maximum coordinates of the image with original
            information
    """
    if isinstance(translation, string_types):
        translation = im[translation]

    translation = np.array(translation)
    if reverse:
        translation *= -1

    shape = im.shape

    xmin = max(0, translation[0])
    xmax = min(shape[0], shape[0] + translation[0])
    ymin = max(0, translation[1])
    ymax = min(shape[1], shape[1] + translation[1])

    return (xmin, xmax, ymin, ymax)


def plot_histogram(im, bins=256):
    """Plot the histogram and cumulative distribution for the image."""
    hist, bins = np.histogram(im, bins)
    cum, bins = im.cumulative_distribution(nbins=bins)
    cum = cum * np.max(hist) / np.max(cum)
    plt.figure()
    plt.plot(bins, hist, "k-")
    plt.plot(bins, cum, "r-")


def threshold_minmax(im, threshmin=0.1, threshmax=0.9):
    """Return a boolean array which is thresholded between threshmin and  threshmax.

    (ie True if value is between threshmin and threshmax)
    """
    im = im.convert(float)
    return np.logical_and(im > threshmin, im < threshmax)


def denoise(im, weight=0.1):
    """Rename the skimage restore function."""
    return im.denoise_tv_chambolle(weight=weight)


def do_nothing(self):
    """Nulop function for testing the integration into ImageArray."""
    return self


@changes_size
def crop(self, *args, **kargs):
    """Crop the image according to a box.

    Args:
        box(tuple) or 4 separate args or None:
            (xmin,xmax,ymin,ymax)
            If None image will be shown and user will be asked to select
            a box (bit experimental)

    Keyword Arguments:
        copy(bool):
            If True return a copy of ImageFile with the cropped image
    Returns:
        (ImageArray):
            view or copy of array asked for

    Notes:
        This is essentially like taking a view onto the array
        but uses image x,y coords (x,y --> col,row)
        Returns a view according to the coords given. If box is None it will
        allow the user to select a rectangle. If a tuple is given with None
        included then max extent is used for that coord (analogous to slice).
        If copy then return a copy of self with the cropped image.

        The box can be specified in a number of ways:
            -   (int):
                A border around all sides of the given number pixels is ignored.
            -   (float 0.0-1.0):
                A border of the given fraction of the images height and width is ignored
            -   (string):
                A corresponding item of metadata is located and used  to specify the box
            -   (tuple of 4 ints or floats):
                For each item in the tuple it is interpreted as follows:
                    -   (int):
                        A pixel coordinate in either the x or y direction
                    -   (float 0.0-1.0):
                        A fraction of the width or height in from the left, right, top, bottom sides
                    -   (float > 1.0):
                        Is rounded to the nearest integer and used a pixel coordinate.
                    -   None:
                        The extent of the image is used.

    Example:
        a=ImageFile(np.arange(12).reshape(3,4))

        a.crop(1,3,None,None)
    """
    box = self._box(*args, **kargs)
    ret = self[box]
    if "copy" in kargs.keys() and kargs["copy"]:
        ret = ret.clone
    return ret


def dtype_limits(self, clip_negative=True):
    """Return intensity limits, i.e. (min, max) tuple, of the image's dtype.

    Args:
        image(ndarray):
            Input image.
        clip_negative(bool):
            If True, clip the negative range (i.e. return 0 for min intensity)
            even if the image dtype allows negative values.

    Returns:
        (imin, imax : tuple): Lower and upper intensity limits.
    """
    if clip_negative is None:
        clip_negative = True
    imin, imax = dtype_range[self.dtype.type]
    if clip_negative:
        imin = 0
    return imin, imax


@keep_return_type
def asarray(self):
    """Provide a consistent way to get at the underlying array data in both ImageArray and ImageFile objects."""
    return self


def asfloat(self, normalise=True, clip=False, clip_negative=False):
    """Return the image converted to floating point type.

    If currently an int type and normalise then floats will be normalised
    to the maximum allowed value of the int type.
    If currently a float type then no change occurs.
    If clip then clip values outside the range -1,1
    If clip_negative then further clip values to range 0,1

    Keyword Arguments:
        normalise(bool):
            normalise the image to the max value of current int type
        clip_negative(bool):
            clip negative intensity to 0
    """
    if self.dtype.kind == "f":
        ret = self
    else:
        ret = self.convert(dtype=np.float64, normalise=normalise).view(type(self))  # preserve metadata
        tmp = metadataObject.__new__(metadataObject)
        for k, v in tmp.__dict__.items():
            if k not in ret.__dict__:
                ret.__dict__[k] = v
        c = self.clone  # copy formatting and apply to new array
        for k, v in c._optinfo.items():
            setattr(ret, k, v)
    if clip or clip_negative:
        ret = ret.clip_intensity(clip_negative=clip_negative)
    return ret


def clip_intensity(self, clip_negative=False, limits=None):
    """Clip intensity outside the range -1,1 or 0,1.

    Keyword Arguments:
        clip_negative(bool):
            if True clip to range 0,1 else range -1,1
        limits (low,high):
            Clip the intensity between low and high rather than zero and 1.

    Ensure data range is -1 to 1 or 0 to 1 if clip_negative is True.

    """
    if limits is None:
        dl = self.dtype_limits(clip_negative=clip_negative)
    else:
        dl = list(limits)
    np.clip(self, dl[0], dl[1], out=self)
    return self


def asint(self, dtype=np.uint16):
    """Convert the image to unsigned integer format.

    May raise warnings about loss of precision.
    """
    if self.dtype.kind == "f" and (np.max(self) > 1 or np.min(self) < -1):
        self = self.normalise()
    ret = self.convert(dtype)
    ret = ret.view(type(self))
    tmp = metadataObject.__new__(metadataObject)
    for k, v in tmp.__dict__.items():
        if k not in ret.__dict__:
            ret.__dict__[k] = v

    for k, v in self._optinfo.items():
        setattr(ret, k, v)
    return ret


def save(self, filename=None, **kargs):
    """Save the image into the file 'filename'.

    Args:
        filename (string, bool or None):
            Filename to save data as, if this is None then the current filename for the object is used
            If this is not set, then then a file dialog is used. If filename is False then a file dialog is forced.

    Keyword Args:
        fmt (string or list):
            format to save data as. 'tif', 'png' or 'npy' or a list of them. If not included will guess from
            filename.
        forcetype (bool):
            integer data will be converted to np.float32 type for saving. if forcetype then preserve and save as
            int type (will be unsigned).

    Notes:
        Metadata will be preserved in .png and .tif format.

        fmt can be 'png', 'npy', 'tif', 'tiff'  or a list of more than one of those.
        tif is recommended since metadata is lost in .npy format but data is
        converted to integer format for png so that definition cannot be
        saved.

    Since Stoner.Image is meant to be a general 2d array often with negative
    and floating point data this poses a problem for saving images. Images
    are naturally saved as 8 or more bit unsigned integer values representing colour.
    The only obvious way to save an image and preserve negative data
    is to save as a float32 tif. This has the advantage over the npy
    data type which cannot be opened by external programs and will not
    save metadata.
    """
    # Standard filename block
    if filename is None:
        filename = getattr(self, "filename", None)
    if filename is None or (isinstance(filename, bool) and not filename):
        # now go and ask for one
        filename = get_filedialog(what="file", filetypes=IMAGE_FILES)

    def_fmt = os.path.splitext(filename)[1][1:]  # Get a default format from the filename
    if def_fmt not in self.fmts:  # Default to png if nothing else
        def_fmt = "png"
    fmt = kargs.pop("fmt", [def_fmt])

    if not isinstance(fmt, list):
        fmt = [fmt]
    if set(fmt) & set(self.fmts) == set([]):
        raise ValueError(f"fmt must be {','.join(self.fmts)}")
    fmt = ["tiff" if f == "tif" else f for f in fmt]
    self.filename = filename
    for fm in fmt:
        saver = getattr(self, f"save_{fm}", "save_tif")
        if fm == "tiff":
            forcetype = kargs.pop("forcetype", False)
            saver(filename, forcetype)
        else:
            saver(filename)


def save_png(self, filename):
    """Save the ImageArray with metadata in a png file.

    This can only save as 8bit unsigned integer so there is likely
    to be a loss of precision on floating point data"""
    pngname = os.path.splitext(filename)[0] + ".png"
    meta = PngImagePlugin.PngInfo()
    info = self.metadata.export_all()
    info = [(i.split("=")[0], "=".join(i.split("=")[1:])) for i in info]
    for k, v in info:
        meta.add_text(k, v)
    s = (self - self.min()) * 256 / (self.max() - self.min())
    im = Image.fromarray(s.astype("uint8"), mode="L")
    im.save(pngname, pnginfo=meta)


def save_npy(self, filename):
    """Save the ImageArray as a numpy array."""
    npyname = os.path.splitext(filename)[0] + ".npy"
    np.save(npyname, np.array(self))


def save_tiff(self, filename, forcetype=False):
    """Save the ImageArray as a tiff image with metadata.

    Args:
        filename (str):
            Filename to save file as.

    Keyword Args:
        forcetype(bool):
            (deprecated) if forcetype then preserve data type as best as possible on save.
            Otherwise we let the underlying pillow library choose the best data type.

    Note:
        PIL can save in modes "L" (8bit unsigned int), "I" (32bit signed int),
        or "F" (32bit signed float). In general max info is preserved for "F"
        type so if forcetype is not specified then this is the default. For
        boolean type data mode "L" will suffice and this is chosen in all cases.
        The type name is added as a string to the metadata before saving.

    """
    from PIL.TiffImagePlugin import ImageFileDirectory_v2
    import json

    dtype = np.dtype(self.dtype).name  # string representation of dtype we can save
    self["ImageArray.dtype"] = dtype  # add the dtype to the metadata for saving.
    if forcetype:  # PIL supports uint8, int32 and float32, try to find the best match
        if self.dtype == np.uint8 or self.dtype.kind == "b":  # uint8 or boolean
            im = Image.fromarray(self, mode="L")
        elif self.dtype.kind in ["i", "u"]:
            im = Image.fromarray(self.astype("int32"), mode="I")
        else:  # default to float32
            im = Image.fromarray(self.astype(np.float32), mode="F")
    else:
        if (
            self.dtype.kind == "b"
        ):  # boolean we're not going to lose data by saving as unsigned int	            im = Image.fromarray(self)
            im = Image.fromarray(self, mode="L")
        else:  # try to convert everything else to float32 which can has maximum preservation of info
            try:
                im = Image.fromarray(self)
            except TypeError:
                im = Image.fromarray(self.astype("float32"))
    ifd = ImageFileDirectory_v2()
    ifd[270] = json.dumps(
        {"type": type(self).__name__, "module": type(self).__module__, "metadata": self.metadata.export_all()}
    )
    ext = os.path.splitext(filename)[1]
    if ext in [".tif", ".tiff"]:  # ensure extension is preserved in save
        pass
    else:  # default to tiff
        ext = ".tiff"
    tiffname = os.path.splitext(filename)[0] + ext
    im.save(tiffname, tiffinfo=ifd)
