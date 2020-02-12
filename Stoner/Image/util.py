# -*- coding: utf-8 -*-
"""Code adapted from skimage module
"""
from __future__ import division

__all__ = ["sign_loss", "prec_loss", "dtype_range", "_dtype", "_dtype2", "build_funcs_proxy"]

import numpy as np
from warnings import warn
from ..core.base import regexpDict

dtype_range = {
    np.bool_: (False, True),
    np.bool8: (False, True),
    np.uint8: (0, 255),
    np.uint16: (0, 65535),
    np.int8: (-128, 127),
    np.int16: (-32768, 32767),
    np.int64: (-2 ** 63, 2 ** 63 - 1),
    np.uint64: (0, 2 ** 64 - 1),
    np.int32: (-2 ** 31, 2 ** 31 - 1),
    np.uint32: (0, 2 ** 32 - 1),
    np.float32: (-1.0, 1.0),
    np.float64: (-1.0, 1.0),
}

integer_types = (np.uint8, np.uint16, np.int8, np.int16)

_supported_types = (
    np.bool_,
    np.bool8,
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
    warn(
        "Possible sign loss when converting negative image of type "
        "%s to positive image of type %s." % (dtypeobj_in, dtypeobj)
    )


def prec_loss(dtypeobj_in, dtypeobj):
    """Warn over precision loss when converting image."""
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


def _scale(a, n, m, dtypeobj_in, dtypeobj, copy=True):
    """cale unsigned/positive integers from n to m bits

    Numbers can be represented exactly only if m is a multiple of n
    Output array is of same kind as input."""
    kind = a.dtype.kind
    if n > m and a.max() < 2 ** m:
        mnew = int(np.ceil(m / 2) * 2)
        if mnew > m:
            dtype = "int%s" % mnew
        else:
            dtype = "uint%s" % mnew
        n = int(np.ceil(n / 2) * 2)
        msg = "Downcasting %s to %s without scaling because max " "value %s fits in %s" % (
            a.dtype,
            dtype,
            a.max(),
            dtype,
        )
        warn(msg)
        return a.astype(_dtype2(kind, m))
    elif n == m:
        return a.copy() if copy else a
    elif n > m:
        # downscale with precision loss
        prec_loss(dtypeobj_in, dtypeobj)
        if copy:
            b = np.empty(a.shape, _dtype2(kind, m))
            np.floor_divide(a, 2 ** (n - m), out=b, dtype=a.dtype, casting="unsafe")
            return b
        else:
            a //= 2 ** (n - m)
            return a
    elif m % n == 0:
        # exact upscale to a multiple of n bits
        if copy:
            b = np.empty(a.shape, _dtype2(kind, m))
            np.multiply(a, (2 ** m - 1) // (2 ** n - 1), out=b, dtype=b.dtype)
            return b
        else:
            a = np.array(a, _dtype2(kind, m, a.dtype.itemsize), copy=False)
            a *= (2 ** m - 1) // (2 ** n - 1)
            return a
    else:
        # upscale to a multiple of n bits,
        # then downscale with precision loss
        prec_loss(dtypeobj_in, dtypeobj)
        o = (m // n + 1) * n
        if copy:
            b = np.empty(a.shape, _dtype2(kind, o))
            np.multiply(a, (2 ** o - 1) // (2 ** n - 1), out=b, dtype=b.dtype)
            b //= 2 ** (o - m)
            return b
        else:
            a = np.array(a, _dtype2(kind, o, a.dtype.itemsize), copy=False)
            a *= (2 ** o - 1) // (2 ** n - 1)
            a //= 2 ** (o - m)
            return a


def build_funcs_proxy():
    """Build a dictionary of references to Image manipulation functions."""
    # Delayed imports
    from . import imagefuncs

    from skimage import (
        color,
        exposure,
        feature,
        io,
        measure,
        filters,
        graph,
        util,
        restoration,
        morphology,
        segmentation,
        transform,
        viewer,
    )
    import scipy.ndimage as ndi

    # Cache is a regular expression dictionary - keys matched directly and then by regular expression
    func_proxy = regexpDict()
    short_names = []

    # Get the Stoner.Image.imagefuncs mopdule first
    for d in dir(imagefuncs):
        if not d.startswith("_"):
            func = getattr(imagefuncs, d)
            if callable(func) and getattr(func, "__module__", "") == imagefuncs.__name__:
                name = "{}__{}".format(func.__module__, d).replace(".", "__")
                func_proxy[name] = func
                short_names.append((d, func))

    # Add scipy.ndimage functions
    _sp_mods = [ndi.interpolation, ndi.filters, ndi.measurements, ndi.morphology, ndi.fourier]
    for mod in _sp_mods:
        for d in dir(mod):
            if not d.startswith("_"):
                func = getattr(mod, d)
                if callable(func) and getattr(func, "__module__", "") == mod.__name__:
                    func.transpose = True
                    name = "{}__{}".format(func.__module__, d).replace(".", "__")
                    func_proxy[name] = func
                    short_names.append((d, func))

    # Add the scikit images modules
    _ski_modules = [
        color,
        exposure,
        feature,
        io,
        measure,
        filters,
        filters.rank,
        graph,
        util,
        restoration,
        morphology,
        segmentation,
        transform,
        viewer,
    ]
    for mod in _ski_modules:
        for d in dir(mod):
            if not d.startswith("_"):
                func = getattr(mod, d)
                if callable(func):
                    name = f"{getattr(func,'__module__','')}__{d}".replace(".", "__")
                    func_proxy[name] = func
                    short_names.append((d, func))

    for n, f in reversed(short_names):
        func_proxy[n] = f

    return func_proxy
