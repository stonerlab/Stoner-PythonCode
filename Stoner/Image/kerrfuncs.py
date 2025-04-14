# -*- coding: utf-8 -*-
"""Specialist functions for dealing with Kerr Images."""

import os
import subprocess  # nosec
import tempfile

import numpy as np
from skimage import exposure, io, transform

from ..core.base import TypeHintedDict
from ..compat import which
from ..core.exceptions import assertion, StonerAssertionError

GRAY_RANGE = (0, 65535)  # 2^16
IM_SIZE = (512, 672)  # Standard Kerr image size
AN_IM_SIZE = (554, 672)  # Kerr image with annotation not cropped
pattern_file = os.path.join(os.path.dirname(__file__), "kerr_patterns.txt")

_useful_keys = [
    "X-B-2d",
    "field: units",
    "MicronsPerPixel",
    "Comment:",
    "Contrast Shift",
    "HorizontalFieldOfView",
    "Images to Average",
    "Lens",
    "Magnification",
    "Subtraction Std",
]
_test_keys = ["X-B-2d", "field: units"]  # minimum keys in data to assert that it is a standard file output


def _parse_text(text, key=None):
    """Parse text which has been recognised from an image if key is given specific hints may be applied."""
    # strip any internal white space
    text = [t.strip() for t in text.split()]
    text = "".join(text)

    # replace letters that look like numbers
    errors = [
        ("s", "5"),
        ("S", "5"),
        ("O", "0"),
        ("f", "/"),
        ("::", "x"),
        ("Z", "2"),
        ("l", "1"),
        ("\xe2\x80\x997", "7"),
        ("?", "7"),
        ("I", "1"),
        ("].", "1"),
        ("'", ""),
    ]
    for item in errors:
        text = text.replace(item[0], item[1])

    # apply any key specific corrections
    if key in ["ocr_field", "ocr_scalebar_length_microns"]:
        try:
            text = float(text)
        except ValueError:
            pass  # leave it as string
    # print '{} after processing: \'{}\''.format(key,data)

    return text


def crop_text(kerr_im, copy=False):
    """Crop the bottom text area from a standard Kermit image.

    KeywordArguments:
        copy(bool):
            Whether to return a copy of the data or the original data

    Returns:
    (ImageArray):
        cropped image
    """
    if kerr_im.shape == IM_SIZE:
        return kerr_im
    if kerr_im.shape != AN_IM_SIZE:
        raise ValueError(
            f"Need a full sized Kerr image to crop. Current size is {kerr_im.shape}"
        )  # check it's a normal image
    return kerr_im.crop(None, None, None, IM_SIZE[0], copy=copy)


def reduce_metadata(kerr_im):
    """Reduce the metadata down to a few useful pieces and do a bit of processing.

    Returns:
        (:py:class:`TypeHintedDict`): the new metadata
    """
    newmet = {}
    if not all([k in kerr_im.keys() for k in _test_keys]):
        return kerr_im.metadata  # we've not got a standard Labview output, not safe to reduce
    for key in _useful_keys:
        if key in kerr_im.keys():
            newmet[key] = kerr_im[key]
    newmet["field"] = newmet.pop("X-B-2d")  # rename
    if "Subtraction Std" in kerr_im.keys():
        newmet["subtraction"] = newmet.pop("Subtraction Std")
    if "Averaging" in kerr_im.keys():
        if kerr_im["Averaging"]:  # averaging was on
            newmet["Averaging"] = newmet.pop("Images to Average")
        else:
            newmet["Averaging"] = 1
            newmet.pop("Images to Average")
    kerr_im.metadata = TypeHintedDict(newmet)
    return kerr_im.metadata


def _tesseract_image(kerr_im, key):
    """Ocr image with tesseract tool.

    im is the cropped image containing just a bit of text
    key is the metadata key we're trying to find, it may give a
    hint for parsing the text generated.
    """
    # first set up temp files to work with
    tmpdir = tempfile.mkdtemp()
    textfile = os.path.join(tmpdir, "tmpfile.txt")
    stdoutfile = os.path.join(tmpdir, "logfile.txt")
    imagefile = os.path.join(tmpdir, "tmpim.tif")
    with open(textfile, "w", encoding="utf-8") as tf:  # open a text file to export metadata to temporarily
        pass

    # process image to make it easier to read
    i = 1.0 * kerr_im / np.max(kerr_im)  # change to float and normalise
    i = exposure.rescale_intensity(i, in_range=(0.49, 0.5))  # saturate black and white pixels
    i = exposure.rescale_intensity(i)  # make sure they're black and white
    i = transform.rescale(i, 5.0, mode="constant")  # rescale to get more pixels on text
    io.imsave(
        imagefile,
        (255.0 * i).astype("uint8"),
    )  # python imaging library will save according to file extension

    # call tesseract
    if kerr_im.tesseractable:
        tesseract = which("tesseract")
        with open(stdoutfile, "w", encoding="utf-8") as stdout:
            subprocess.call(  # nosec
                [tesseract, imagefile, textfile[:-4]], stdout=stdout, stderr=subprocess.STDOUT
            )  # adds '.txt' extension itkerr_im
        os.unlink(stdoutfile)
    with open(textfile, "r", encoding="utf-8") as tf:
        data = tf.readline()

    # delete the temp files
    os.remove(textfile)
    os.remove(imagefile)
    os.rmdir(tmpdir)

    # parse the reading
    if len(data) == 0:
        print(f"No data read for {key}")
    data = _parse_text(data, key=key)
    return data


def get_scalebar(kerr_im):
    """Get the length in pixels of the image scale bar."""
    im = kerr_im[519:520, :419]
    im = im.astype(float)
    im = (im - im.min()) / (im.max() - im.min())
    im = exposure.rescale_intensity(im, in_range=(0.49, 0.5))  # saturate black and white pixels
    im = exposure.rescale_intensity(im)  # make sure they're black and white
    im = np.diff(im[0])  # 1d numpy array, differences
    lim = [np.where(im > 0.9)[0][0], np.where(im < -0.9)[0][0]]  # first occurrence of both cases
    assertion(len(lim) == 2, "Couldn't find scalebar")
    return lim[1] - lim[0]


def float_and_croptext(kerr_im):
    """Convert image to float and crop_text.

    Just to group typical functions together
    """
    ret = kerr_im.asfloat()
    ret = ret.crop_text()
    return ret


def ocr_metadata(kerr_im, field_only=False):
    """Use image recognition to try to pull the metadata numbers off the image.

    Requirements:
        This function uses tesseract to recognise the image, therefore
        tesseract file1 file2 must be valid on your command line.
        Install tesseract from
        https://sourceforge.net/projects/tesseract-ocr-alt/files/?source=navbar

    KeywordArguments:
        field_only(bool):
            only try to return a field value

    Returns:
        metadata: dict
            updated metadata dictionary
    """
    if kerr_im.shape != AN_IM_SIZE:
        pass  # can't do anything without an annotated image

    # now we have to crop the image to the various text areas and try tesseract
    elif field_only:
        fbox = (110, 165, 527, 540)  # (This is just the number area not the unit)
        im = kerr_im.crop(box=fbox, copy=True)
        field = kerr_im._tesseract_image(im, "ocr_field")
        kerr_im.metadata["ocr_field"] = field
    else:
        text_areas = {
            "ocr_field": (110, 165, 527, 540),
            "ocr_date": (542, 605, 512, 527),
            "ocr_time": (605, 668, 512, 527),
            "ocr_subtract": (237, 260, 527, 540),
            "ocr_average": (303, 350, 527, 540),
        }
        try:
            sb_length = kerr_im.get_scalebar()
        except (StonerAssertionError, AssertionError):
            sb_length = None
        if sb_length is not None:
            text_areas.update(
                {
                    "ocr_scalebar_length_microns": (sb_length + 10, sb_length + 27, 514, 527),
                    "ocr_lens": (sb_length + 51, sb_length + 97, 514, 527),
                    "ocr_zoom": (sb_length + 107, sb_length + 149, 514, 527),
                }
            )

        metadata = {}  # now go through and process all keys
        for key in text_areas:
            im = kerr_im.crop(box=text_areas[key], copy=True)
            metadata[key] = _tesseract_image(kerr_im, key)
        metadata["ocr_scalebar_length_pixels"] = sb_length
        if isinstance(metadata["ocr_scalebar_length_microns"], float):
            metadata["ocr_microns_per_pixel"] = metadata["ocr_scalebar_length_microns"] / sb_length
            metadata["ocr_pixels_per_micron"] = 1 / metadata["ocr_microns_per_pixel"]
            metadata["ocr_field_of_view_microns"] = np.array(IM_SIZE) * metadata["ocr_microns_per_pixel"]
        kerr_im.metadata.update(metadata)
    if "ocr_field" in kerr_im.metadata.keys() and not isinstance(kerr_im.metadata["ocr_field"], (int, float)):
        kerr_im.metadata["ocr_field"] = np.nan  # didn't read the field properly
    return kerr_im.metadata


def defect_mask(kerr_im, thresh=0.6, corner_thresh=0.05, radius=1, return_extra=False):
    """Try to create a boolean array which is a mask for typical defects found in Image images.

    Best for unprocessed raw images. (for subtract images
    see defect_mask_subtract_image)
    Looks for big bright things by thresholding and small and dark defects using
    skimage's corner_fast algorithm

    Parameters:
    thresh (float):
        brighter stuff than this gets removed (after image levelling)
    corner_thresh (float):
        see corner_fast (skimage):
    radius (float):
        radius of pixels around corners that are added to mask
    return_extra (bool):
        this returns a dictionary with some of the intermediate steps of the
        calculation

    Returns:
        totmask (ndarray of bool):
            mask
    info (*optional* dict):
        dictionary of intermediate calculation steps
    """
    im = kerr_im.asfloat()
    im = im.level_image(poly_vert=3, poly_horiz=3)
    th = im.threshold_minmax(0, thresh)
    # corner fast good at finding the small black dots
    cor = im.corner_fast(threshold=corner_thresh)
    blobs = cor.blob_doh(min_sigma=1, max_sigma=20, num_sigma=3, threshold=0.01)
    q = np.zeros_like(im)
    for y, x, _ in blobs:
        q[
            int(np.round(y - radius)) : int(np.round(y + radius)),
            int(np.round(x - radius)) : int(np.round(x + radius)),
        ] = 1.0
    totmask = np.logical_or(q, th)
    if return_extra:
        info = {"flattened_image": im, "corner_fast": cor, "corner_points": blobs, "corner_mask": q, "thresh_mask": th}
        return totmask, info
    return totmask


def defect_mask_subtract_image(kerr_im, threshmin=0.25, threshmax=0.9, denoise_weight=0.1, return_extra=False):
    """Create a mask array for a typical subtract Image image.

    Uses a denoise algorithm followed by simple thresholding.

    Returns:
        totmask (ndarray of bool):
            the created mask
        info (*optional* dict):
            the intermediate denoised image
    """
    p = kerr_im.denoise_tv_chambolle(weight=denoise_weight)
    submask = p.threshold_minmax(threshmin, threshmax)
    if return_extra:
        info = {"denoised_image": p}
        return submask, info
    return submask
