# -*- coding: utf-8 -*-
"""Functions for manipulating Kerr (or any other) images

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
    "rotate",
    "translate",
    "translate_limits",
    "plot_histogram",
    "threshold_minmax",
    "defect_mask",
    "do_nothing",
    "float_and_croptext",
    "denoise",
]
from Stoner.compat import string_types
import warnings
import numpy as np, matplotlib.pyplot as plt, os
from Stoner.tools import istuple, isiterable
from scipy.interpolate import griddata
from skimage import feature, measure, transform
from matplotlib.colors import Normalize
import matplotlib.cm as cm

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
    from image_registration import fft_tools
except ImportError:
    chi2_shift = None

from .core import ImageArray
from Stoner import Data


def _scale(coord, scale=1.0, to_pixel=True):
    """Convert pixel cordinates to scaled co-ordinates or visa versa.

    Args:
        coord(int,float or iterable): Coordinates to be scaled

    Keyword Arguments:
        scale(float): Microns per Pixel scale of image
        to_pixel(bool): Force the conversion to be to pixels

    Returns:
        scaled co-ordinates.
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
    """rescale the intensity of the image.

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
        vmin, vmax = np.percentile(im, np.array(lims) * 100)
    else:
        vmin, vmax = lims[0], lims[1]
    im.metadata["adjust_contrast"] = (vmin, vmax)
    im = im.rescale_intensity(in_range=(vmin, vmax))  # clip the intensity
    return im


def align(im, ref, method=None, **kargs):
    """Use one of a variety of algroithms to align two images.

    Args:
        im (ndarray) image to align
        ref (ndarray) reference array

    Keyword Args:
        method (str or None):
            If given specifies which module to try and use.
            Options: 'scharr', 'chi2_shift', 'imreg_dft', 'cv2'
        **kargs (various): All other keyword arguments are passed to the specific algorithm.

    Returns
        (ImageArray or ndarray) aligned image

    Notes:
        Currently three algorithms are supported:
            - image_registration module's chi^2 shift: This uses a dft with an automatic
              up-sampling of the fourier transform for sub-pixel alignment. The metadata
              key *chi2_shift* contains the translation vector and errors.
            - imreg_dft module's similarity function. This implements a full scale, rotation, translation
              algorithm (by default cosntrained for just translation). It's unclear how much sub-pixel translation
              is accomodated.
            - cv2 module based affine transform on a gray scale image.
              from: http://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    """
    # To be consistent with x-y co-ordinate systems
    if all([m is None for m in [imreg_dft, chi2_shift, cv2]]):
        raise ImportError("align requires one of imreg_dft, chi2_shift or cv2 modules to be available.")
    if method == "scharr" and imreg_dft is not None:
        im = im.T
        ref = ref.T
        scale = np.ceil(np.max(im.shape) / 500.0)
        ref1 = ref.gaussian_filter(sigma=scale, mode="wrap").scharr()
        im1 = im.gaussian_filter(sigma=scale, mode="wrap").scharr()
        im1 = im1.align(ref1, method="imreg_dft")
        tvec = np.array(im1["tvec"])
        new_im = im.shift(tvec)
        new_im["tvec"] = tuple(-tvec)
        new_im = new_im.T
    elif (method is None and chi2_shift is not None) or method == "chi2_shift":
        kargs["zeromean"] = kargs.get("zeromean", True)
        result = np.array(chi2_shift(ref, im, **kargs))
        new_im = im.__class__(fft_tools.shiftnd(im, -result[0:2]))
        new_im.metadata.update(im.metadata)
        new_im.metadata["chi2_shift"] = result
    elif (method is None and imreg_dft is not None) or method == "imreg_dft":
        constraints = kargs.pop("constraints", {"angle": [0.0, 0.0], "scale": [1.0, 0.0]})
        cls = im.__class__
        with warnings.catch_warnings():  # This causes a warning due to the masking
            warnings.simplefilter("ignore")
            result = imreg_dft.similarity(ref, im, constraints=constraints)
        new_im = (result.pop("timg")).view(type=cls)
        new_im.metadata.update(im.metadata)
        new_im.metadata.update(result)
    elif (method is None and cv2 is not None) or method == "cv2":
        im1_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Find size of image1
        sz = im.shape

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
        (_, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

        # Use warpAffine for Translation, Euclidean and Affine
        new_im = cv2.warpAffine(im, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    else:  # No cv2 available so don't do anything.
        raise RuntimeError("Couldn't find an image alignment algorithm to use")
    return new_im.T


def correct_drift(im, ref, threshold=0.005, upsample_factor=50, box=None, do_shift=True):
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
        A shifted iamge with the image shift added to the metadata as 'correct drift'.

    Detects common features on the images and tracks them moving.
    Adds 'drift_shift' to the metadata as the (x,y) vector that translated the
    image back to it's origin.
    """
    if box is None:
        box = im.max_box
    cim = im.crop_image(box=box)

    refed = ImageArray(ref, get_metadata=False)
    refed = refed.crop_image(box=box)
    refed = refed.filter_image(sigma=1)
    refed = refed > refed.threshold_otsu()
    refed = refed.corner_fast(threshold=threshold)

    imed = cim.clone
    imed = imed.filter_image(sigma=1)
    imed = imed > imed.threshold_otsu()
    imed = imed.corner_fast(threshold=threshold)

    shift = feature.register_translation(refed, imed, upsample_factor=upsample_factor)[0]
    if do_shift:
        im = im.translate(translation=(-shift[1], -shift[0]))  # x,y
    im.metadata["correct_drift"] = (-shift[1], -shift[0])
    return im


def subtract_image(im, background, contrast=16, clip=True):
    """subtract a background image from the ImageArray

    Multiply the contrast by the contrast parameter.
    If clip is on then clip the intensity after for the maximum allowed data range.
    """
    im = im.convert_float()
    im = contrast * (im - background) + 0.5
    if clip:
        im = im.clip_intensity()
    return im


def fft(im, shift=True, phase=False):
    """Perform a 2d fft of the image and shift the result to get zero frequency in the centre."""
    r = np.fft.fft2(im)

    if shift:
        r = np.fft.fftshift(r)

    if not phase:
        r = np.abs(r)
    else:
        r = np.angle(r)

    r = im.__class__(r)
    r.metadata.update(im.metadata)
    return r


def filter_image(im, sigma=2):
    """Alias for skimage.filters.gaussian"""
    return im.gaussian(sigma=sigma)


def gridimage(im, points=None, xi=None, method="linear", fill_value=1.0, rescale=False):
    """Use :py:func:`scipy.interpolate.griddata` to shift the image to a regular grid of co-ordinates.

    Args:
        points (tuple of (x-co-ords,yco-ordsa)): The actual sampled data co-ordinates
        xi (tupe of (2D array,2D array)): The regular grid co-ordinates (as generated by e.g. :py:func:`np.meshgrid`)

    Keyword Arguments:
        method ("linear","cubic","nearest"): how to interpolate, default is linear
        fill_value (folat): What to put when the co-ordinates go out of range (default is 1.0). May be a callable in which
            case the initial image is presented as the only argument
        rescale (bool): If the x and y co-ordinates are very different in scale, set this to True.

    Returns:
        A copy of the modified image. The image data is interpolated and metadata kets "actual_x","actual_y","sample_x","samp[le_y" are
        set to give co-ordinates of new grid.

    Notes:
        If points and or xi are missed out then we try to construct them from the metadata. For points, the metadata keys
        "actual-x" and "actual_y" are looked for and for xi, the metadata keys "sample_x" and "sample_y" are used. These are set
        , for example, by the :py:class:`Stoner.HDF5.SXTMImage` loader if the interformeter stage data was found in the file.

        The metadata used in this case is then adjusted as well to ensure that repeated application of this method doesn't change the image
        after it has been corrected once.
    """
    if points is None:
        points = np.column_stack((im["actual_x"].ravel(), im["actual_y"].ravel()))
    if xi is None:
        xi = xi = (im["sample_x"], im["sample_y"])

    if callable(fill_value):
        fill_value = fill_value(im)

    im2 = griddata(points, im.ravel(), xi, method, fill_value, rescale)
    im2 = im.__class__(im2)
    im2.metadata = im.metadata
    im2.metadata["actual_x"] = xi[0]
    im2.metadata["actual_y"] = xi[1]
    return im2


def hist(im, *args, **kargs):
    """Pass through to :py:func:`matplotlib.pyplot.hist` function."""
    counts, edges = np.histogram(im.ravel(), *args, **kargs)
    centres = (edges[1:] + edges[:-1]) / 2
    new = Data(np.column_stack((centres, counts)))
    new.column_headers = ["Intensity", "Frequency"]
    new.setas = "xy"
    return new


def imshow(im, **kwargs):
    """quick plot of image

    Keyword Arguments:
        figure (int, str or matplotlib.figure): if int then use figure number given,
            if figure is 'new' then create a new figure, if None then use whatever default
            figure is available
        title (str,None,False): Title for plot - defaults to False (no title). None will take the title from the filename if present
        cmap (str,matplotlib.cmap): Colour scheme for plot, defaults to gray

    Any masked areas are set to NaN which stops them being plotted at all.
    """
    figure = kwargs.pop("figure", "new")
    title = kwargs.pop("title", False)
    cmap = kwargs.pop("cmap", "gray")
    if isinstance(cmap, string_types):
        cmap = getattr(cm, cmap)
    if np.ma.is_masked(im):
        im_data = im.data
        vmax = np.max(im_data.data)
        vmin = np.min(im_data.data)
        alpha = np.where(im.mask, 0.15, 1.0)
        colors = cmap(Normalize(vmin, vmax)(im_data))
        colors[..., -1] = alpha
        im_data = colors
    else:
        im_data = im
    if figure is not None and isinstance(figure, int):
        fig = plt.figure(figure)
        plt.imshow(im_data, figure=fig, cmap=cmap, **kwargs)
    elif figure is not None and figure == "new":
        fig = plt.figure()
        plt.imshow(im_data, figure=fig, cmap=cmap, **kwargs)
    elif figure is not None:  # matplotlib.figure instance
        fig = plt.imshow(im_data, figure=figure, cmap=cmap, **kwargs)
    else:
        fig = plt.imshow(im_data, cmap=cmap, **kwargs)
    if title is None:
        if "filename" in im.metadata.keys():
            plt.title(os.path.split(im["filename"])[1])
        elif hasattr(im, "filename"):
            plt.title(os.path.split(im.filename)[1])
        else:
            plt.title(" ")
    elif isinstance(title, bool) and not title:
        pass
    else:
        plt.title(title)
    plt.axis("off")
    return fig


def level_image(im, poly_vert=1, poly_horiz=1, box=None, poly=None, mode="clip"):
    """Subtract a polynomial background from image

    Keword Arguments:
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
    cim = im.crop_image(box=box)
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


def normalise(im, scale=None, sample=None, limits=(0.0, 1.0)):
    """Norm alise the data to a fixed scale.

    Keyword Arguements:
        scale (2-tuple): The range to scale the image to, defaults to -1 to 1.
        saple (box): Only use a section of the input image to calculate the new scale over.
        limits (low,high): Take the input range from the *high* and *low* fraction of the input when sorted.

    Returns:
        A scaled version of the data. The ndarray min and max methods are used to allow masked images
        to be operated on only on the unmasked areas.

    Notes:
        The *sample* keyword controls the area in which the range of input values is calculated, the actual scaling is
        done on the whole image.

        The *limits* parameter is used to set the input scale being normalised from - if an image has a few outliers then
        this setting can be used to clip the input range before normalising. The parameters in the limit are the values at
        the *low* and *high* fractions of the cumulative distribution functions.
    """
    mask = im.mask
    cls = im.__class__
    im = im.astype(float)
    if scale is None:
        scale = (-1.0, 1.0)
    if sample is not None:
        section = im[im._box(sample)]
    else:
        section = im
    if limits != (0.0, 1.0):
        low, high = limits
        low = np.sort(section.ravel())[int(low * section.size)]
        high = np.sort(section.ravel())[int(high * section.size)]
        im.clip_intensity(limits=(low, high))
    else:
        high = section.max()
        low = section.min()

    if not istuple(scale, float, float, strict=False):
        raise ValueError("scale should be a 2-tuple of floats.")
    scaled = (im.data - low) / (high - low)
    delta = scale[1] - scale[0]
    offset = scale[0]
    im = scaled * delta + offset
    im = im.view(cls)
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
    """Wrapper for sckit-image method of the same name to get a line_profile.

    Parameters:
        img(ImageArray): Image data to take line section of
        src, dst (2-tuple of int or float): start and end of line profile. If the co-ordinates
            are given as intergers then they are assumed to be pxiel co-ordinates, floats are
            assumed to be real-space co-ordinates using the embedded metadata.
        linewidth (int): the wideth of the profile to be taken.
        order (int 1-3): Order of interpolation used to find image data when not aligned to a point
        mode (str): How to handle data outside of the image.
        cval (float): The constant value to assume for data outside of the image is mode is "constant"
        constrain (bool): Ensure the src and dst are within the image (default True).

    Returns:
        A :py:class:`Stoner.Data` object containing the line profile data and the metadata from the image.
    """
    scale = img.get("MicronsPerPixel", 1.0)
    r, c = img.shape
    if src is None and dst is None:
        if "x" in kargs:
            src = (kargs["x"], 0)
            dst = (kargs["x"], r)
        if "y" in kargs:
            src = (0, kargs["y"])
            dst = (c, kargs["y"])
    if isinstance(src, float):
        src = (src, src)
    if isinstance(dst, float):
        dst = (dst, dst)
    dst = _scale(dst, scale)
    src = _scale(src, scale)
    if not istuple(src, int, int):
        raise ValueError("src co-ordinates are not a 2-tuple of ints.")
    if not istuple(dst, int, int):
        raise ValueError("dst co-ordinates are not a 2-tuple of ints.")

    if constrain:
        fix = lambda x, mx: int(round(sorted([0, x, mx])[1]))
        r, c = img.shape
        src = list(src)
        src = (fix(src[0], r), fix(src[1], c))
        dst = (fix(dst[0], r), fix(dst[1], c))

    result = measure.profile_line(img, src, dst, linewidth, order, mode, cval)
    points = measure.profile._line_profile_coordinates(src, dst, linewidth)[:, :, 0]
    ret = Data()
    ret.data = points.T
    ret.setas = "xy"
    ret &= np.sqrt(ret.x ** 2 + ret.y ** 2) * scale
    ret &= result
    ret.column_headers = ["X", "Y", "Distance", "Intensity"]
    ret.setas = "..xy"
    ret.metadata = img.metadata.copy()
    return ret


def quantize(im, output, levels=None):
    """Quantise the image data into fixed levels given by a mapping

    Args:
        output (list,array,tuple): Output levels to return.

    Keyword Arguments:
        levels (list, array or None): The input band markers. If None is constructed from the data.

    The number of levels should be one less than the number of output levels given.
    """
    if levels is None:
        levels = np.linspace(im.min(), im.max(), len(output) + 1)
    elif len(levels) == len(output) + 1:
        pass
    elif len(levels) == len(output) - 1:
        lvl = np.zeros(len(output) + 1)
        lvl[1:-1] = levels
        lvl[0] = im.min()
        lvl[-1] = im.max()
        levels = lvl
    else:
        raise RuntimeError("{} output levels and {} input levels".format(len(output), len(levels)))

    ret = im.clone
    for lvl, lvh, val in zip(levels[:-1], levels[1:], output):
        select = np.logical_and(np.less_equal(im, lvh), np.greater(im, lvl))
        ret[select] = val
    return ret


def rotate(im, angle, mode="constant", cval=None):
    """Rotates the image.

    Areas lost by move are cropped, and areas gained are made black (0)

    Args:
        rotation: float
            clockwise rotation angle in radians (rotated about top right corner)

    Keyword Argum,ents:
        mode (str): How to handle points outside the original image. See :py:func:`skimage.transform.warp`. Defaults to "constant"
        cval (float): The value to fill with if *mode* is constant. If not speficied or None, defaults to the mean pixcel value.

    Returns:
        im: ImageArray
            rotated image
    """
    rot = transform.SimilarityTransform(rotation=angle)
    if cval is None:
        cval = im.mean()
    im.warp(rot, mode=mode, cval=cval)
    im.metadata["transform:rotation"] = angle
    return im


def translate(im, translation, add_metadata=False, order=3, mode="wrap", cval=None):
    """Translates the image.

    Areas lost by move are cropped, and areas gained are made black (0)
    The area not lost or cropped is added as a metadata parameter
    'translation_limits'

    Args:
        translate (2-tuple): translation (x,y)

    Keyword Arguments:
        add_metadata (bool): Record the shift in the image metadata
        order (int): Interpolation order (default, 3, bi-cubic)
        mode (str): How to handle points outside the original image. See :py:func:`skimage.transform.warp`. Defaults to "wrap"
        cval (float): The value to fill with if *mode* is constant. If not speficied or None, defaults to the mean pixcel value.

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


def translate_limits(im, translation):
    """Find the limits of an image after a translation

    After using ImageArray.translate some areas will be black,
    this finds the max area that still has original pixels in

    Args:
        translation: 2-tuple
            the (x,y) translation applied to the image

    Returns:
        limits: 4-tuple
            (xmin,xmax,ymin,ymax) the maximum coordinates of the image with original
            information
    """
    t = translation
    s = im.shape
    if t[0] <= 0:
        xmin, xmax = 0, s[1] - t[0]
    else:
        xmin, xmax = t[0], s[1]
    if t[1] <= 0:
        ymin, ymax = 0, s[0] - t[1]
    else:
        ymin, ymax = t[1], s[0]
    return (xmin, xmax, ymin, ymax)


def plot_histogram(im):
    """plot the histogram and cumulative distribution for the image"""
    hist, bins = im.histogram()
    cum, bins = im.cumulative_distribution()
    cum = cum * np.max(hist) / np.max(cum)
    plt.figure()
    plt.plot(bins, hist, "k-")
    plt.plot(bins, cum, "r-")


def threshold_minmax(im, threshmin=0.1, threshmax=0.9):
    """returns a boolean array which is thresholded between threshmin and  threshmax.

    (ie True if value is between threshmin and threshmax)
    """
    im = im.convert_float()
    return np.logical_and(im > threshmin, im < threshmax)


def defect_mask(im, thresh=0.6, corner_thresh=0.05, radius=1, return_extra=False):
    """Tries to create a boolean array which is a mask for typical defects found in Image images.

    Best for unprocessed raw images. (for subtract images
    see defect_mask_subtract_image)
    Looks for big bright things by thresholding and small and dark defects using
    skimage's corner_fast algorithm

    Parameters
    ----------
    thresh float
        brighter stuff than this gets removed (after image levelling)
    corner_thresh float
        see corner_fast skimage
    radius:
        radius of pixels around corners that are added to mask

    return_extra dict
        this returns a dictionary with some of the intermediate steps of the
        calculation

    Returns
    -------
    totmask ndarray of bool
        mask
    info (optional) dict
        dictionary of intermediate calculation steps
    """
    im = im.convert_float()
    im = im.level_image(poly_vert=3, poly_horiz=3)
    th = im.threshold_minmax(0, thresh)
    # corner fast good at finding the small black dots
    cor = im.corner_fast(threshold=corner_thresh)
    blobs = cor.blob_doh(min_sigma=1, max_sigma=20, num_sigma=3, threshold=0.01)
    q = np.zeros_like(im)
    for y, x, _ in blobs:
        q[y - radius : y + radius, x - radius : x + radius] = 1.0
    totmask = np.logical_or(q, th)
    if return_extra:
        info = {"flattened_image": im, "corner_fast": cor, "corner_points": blobs, "corner_mask": q, "thresh_mask": th}
        return totmask, info
    return totmask


def defect_mask_subtract_image(im, threshmin=0.25, threshmax=0.9, denoise_weight=0.1, return_extra=False):
    """Create a mask array for a typical subtract Image image
    Uses a denoise algorithm followed by simple thresholding.

    Returns
    -------
    totmask: ndarray of bool
        the created mask
    info (optional) dict:
        the intermediate denoised image
    """

    p = im.denoise_tv_chambolle(weight=denoise_weight)
    submask = p.threshold_minmax(threshmin, threshmax)
    if return_extra:
        info = {"denoised_image": p}
        return submask, info
    return submask


def do_nothing(im):
    """exactly what it says on the tin"""
    return im


def float_and_croptext(im):
    """convert image to float and crop_text
    Just to group typical functions together
    """
    k = im.convert_float()
    k = k.crop_text()
    return k


def denoise(im, weight=0.1):
    """just a rename of the skimage restore function"""
    return im.denoise_tv_chambolle(weight=weight)
