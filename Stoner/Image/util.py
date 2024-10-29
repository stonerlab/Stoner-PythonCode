# -*- coding: utf-8 -*-
"""Code adapted from skimage module."""

from __future__ import division

__all__ = ["sign_loss", "prec_loss", "dtype_range", "_dtype", "_dtype2"]
from warnings import warn

import numpy as np

from ..tools.classes import Options
from ..compat import np_version

dtype_range = {
    np.bool_: (False, True),
    np.uint8: (0, 255),
    np.uint16: (0, 65535),
    np.int8: (-128, 127),
    np.int16: (-32768, 32767),
    np.int64: (-(2**63), 2**63 - 1),
    np.uint64: (0, 2**64 - 1),
    np.int32: (-(2**31), 2**31 - 1),
    np.uint32: (0, 2**32 - 1),
    np.float32: (-1.0, 1.0),
    np.float64: (-1.0, 1.0),
}

if np_version.major == 1 and np_version.minor < 24:
    dtype_range[np.bool8] = (False, True)

integer_types = (np.uint8, np.uint16, np.int8, np.int16)

_supported_types = (
    np.bool_,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
)

dtype_range[np.float16] = (-1, 1)
_supported_types += (np.float16,)


def sign_loss(dtypeobj_in, dtypeobj):
    """Warn over loss of sign information when converting image."""
    if Options().warnings:
        warn(
            "Possible sign loss when converting negative image of type "
            "%s to positive image of type %s." % (dtypeobj_in, dtypeobj)
        )


def prec_loss(dtypeobj_in, dtypeobj):
    """Warn over precision loss when converting image."""
    if Options().warnings:
        warn("Possible precision loss when converting from " "%s to %s" % (dtypeobj_in, dtypeobj))


def _dtype(itemsize, *dtypes):
    """Return first of `dtypes` with itemsize greater than `itemsize."""
    try:
        ret = next(dt for dt in dtypes if itemsize < np.dtype(dt).itemsize)
    except StopIteration:
        ret = dtypes[0]
    return ret


def _dtype2(kind, bits, itemsize=1):
    """Return dtype of `kind` that can store a `bits` wide unsigned int."""
    c = lambda x, y: x <= y if kind == "u" else x < y
    s = next(i for i in (itemsize,) + (2, 4, 8) if c(bits, i * 8))
    return np.dtype(kind + str(s))


def _scale(image, src_bits, dest_bits, dtypeobj_in, dtypeobj, copy=True):
    """Scale unsigned/positive integers from src_bitsto dest_bits_bits.

    Numbers can be represented exactly only if dest_bits_is a multiple of n
    Output array is of same kind as input.
    """
    if src_bits == dest_bits:  # Trivial case no scale necessary
        return image.copy() if copy else image

    if src_bits > dest_bits:
        return _down_scale(image, src_bits, dest_bits, dtypeobj_in, dtypeobj)

    return _up_scale(image, src_bits, dest_bits, dtypeobj_in, dtypeobj)


def _down_scale(image, src_bits, dest_bits, dtypeobj_in, dtypeobj, copy=True):
    """Downscale and image from src_bits to dest_bits."""
    kind = image.dtype.kind
    if image.max() <= 2**dest_bits:
        prefix = ["uint", "int"][dest_bits % 2]
        dest_bits += dest_bits % 2
        dtype = f"{prefix}{dest_bits}"
        src_bits += src_bits % 2
        if Options().warnings:
            warn(
                f"Downcasting {image.dtype} to {dtype} without scaling"
                + f"because max value {image.max()} fits in {dtype}."
            )
        return image.astype(_dtype2(kind, dest_bits))
    prec_loss(dtypeobj_in, dtypeobj)
    if copy:
        image2 = np.empty(image.shape, _dtype2(kind, dest_bits))
    else:
        image2 = image
    np.floor_divide(image, 2 ** (src_bits - dest_bits), out=image2, dtype=image.dtype, casting="unsafe")
    return image2


def _up_scale(image, src_bits, dest_bits, dtypeobj_in, dtypeobj, copy=True):
    """Upscale an image from src_bits to dest_bits."""
    kind = image.dtype.kind
    if dest_bits % src_bits == 0:
        # exact upscale to a multiple of src_bitsbits
        if copy:
            image2 = np.empty(image.shape, _dtype2(kind, dest_bits))
        else:
            image2 = np.array(image, _dtype2(kind, dest_bits, image.dtype.itemsize), copy=False)
        np.multiply(image, (2**dest_bits - 1) // (2**src_bits - 1), out=image2, dtype=image2.dtype)
        return image2
    # upscale to a multiple of src_bitsbits,
    # then downscale with precision loss
    prec_loss(dtypeobj_in, dtypeobj)
    upscale_factor = (dest_bits // src_bits + 1) * src_bits
    if copy:
        image2 = np.empty(image.shape, _dtype2(kind, upscale_factor))
    else:
        image2 = image
    np.multiply(image, (2**upscale_factor - 1) // (2**src_bits - 1), out=image2, dtype=image2.dtype)
    image2 //= 2 ** (upscale_factor - dest_bits)
    return image2
