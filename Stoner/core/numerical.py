"""Prepare ordinary NumPy results from owner-held column descriptions."""

import numpy as np

from ..tools import AttributeStore, isiterable, isnone
from .setas import Setas


def numerical_result(values, *, setas=None, column_headers=None, isrow=False, **kwargs):
    """Return a detached NumPy masked array with explicit result annotations.

    Annotations describe this result only. Native NumPy slicing and arithmetic
    follow NumPy's rules; they do not propagate Stoner schema or create storage.
    """
    result = np.ma.array(values, copy=True, **kwargs)
    roles = setas.clone if hasattr(setas, "clone") else Setas(row=isrow)
    roles.shape = result.shape
    width = result.shape[-1] if result.ndim > 1 or isrow else 1
    if column_headers is not None:
        roles.column_headers = list(column_headers)
    elif len(roles.column_headers) != width:
        roles.column_headers = [f"Column {i}" for i in range(width)]
    if setas is not None and not hasattr(setas, "clone"):
        roles(setas)
    result.setas = result._setas = roles
    result.column_headers = list(roles.column_headers)
    result.isrow = isrow
    result.i = 0 if isrow else np.arange(result.shape[0] if result.ndim else 1)
    for role, field in zip("xyzdefuvw", ("xcol", "ycol", "zcol", "xerr", "yerr", "zerr", "ucol", "vcol", "wcol")):
        position = roles.cols[field]
        if isinstance(position, list):
            position = position[0] if position else None
        if position is not None and (result.ndim > 1 or isrow):
            setattr(result, role, result[..., position])
    return result


def column_arguments(
    self,
    scalar=True,
    xcol=None,
    ycol=None,
    zcol=None,
    ucol=None,
    vcol=None,
    wcol=None,
    xerr=None,
    yerr=None,
    zerr=None,
    **kwargs,
):  # pylint: disable=unused-argument
    """Create an object which has keys  based either on arguments or setas attribute."""
    cols = {
        "xcol": xcol,
        "ycol": ycol,
        "zcol": zcol,
        "ucol": ucol,
        "vcol": vcol,
        "wcol": wcol,
        "xerr": xerr,
        "yerr": yerr,
        "zerr": zerr,
    }
    no_guess = kwargs.get("no_guess", True)
    for i in cols.values():
        if i is not None:  # User specification wins out
            break
    else:  # User didn't set any values, setas will win
        no_guess = kwargs.get("no_guess", False)
    ret = AttributeStore(self.setas._get_cols(no_guess=no_guess, startx=kwargs.get("startx", 0)))
    force_list = kwargs.get("force_list", not scalar)
    for c in list(cols.keys()):
        if isnone(cols[c]):  # Not defined, fallback on setas
            del cols[c]
            continue
        if isinstance(cols[c], bool) and not cols[c]:  # False, delete column altogether
            del cols[c]
            if c in ret:
                del ret[c]
            continue
        if c in ret and isinstance(ret[c], list):
            if isinstance(cols[c], float) or (isinstance(cols[c], np.ndarray) and cols[c].size == len(self)):
                continue
        if isinstance(cols[c], float):
            continue
        cols[c] = self.setas.find_col(cols[c], force_list=force_list)
    ret.update(cols)
    if scalar:
        for c in ret:
            if isinstance(ret[c], list):
                if ret[c]:
                    ret[c] = ret[c][0]
                else:
                    ret[c] = None
    elif isinstance(scalar, bool) and not scalar:
        for c in ret:
            if c.startswith("x") or c.startswith("has_"):
                continue
            if not isiterable(ret[c]) and ret[c] is not None:
                ret[c] = list([ret[c]])
            elif ret[c] is None:
                ret[c] = []
    for n in ["xcol", "xerr", "ycol", "yerr", "zcol", "zerr", "ucol", "vcol", "wcol", "axes"]:
        ret[f"has_{n}"] = n in ret and not (ret[n] is None or (isinstance(ret[n], list) and not ret[n]))

    return ret
