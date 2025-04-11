# -*- coding: utf-8 -*-
"""Module to work with scan files from an AttocubeSPM running Daisy."""
import re

from ...tools.file import FileManager, get_filename
from ...core.exceptions import StonerLoadError
from ..decorators import register_loader

PARAM_RE = re.compile(r"^([\d\\.eE\+\-]+)\s*([\%A-Za-z]\S*)?$")
SCAN_NO = re.compile(r"SC_(\d+)")


def parabola(X, cx, cy, a, b, c):
    """Parabola in the X-Y plane for levelling an image."""
    x, y = X
    return a * (x - cx) ** 2 + b * (y - cy) ** 2 + c


def plane(X, a, b, c):
    """Plane equation for levelling an image."""
    x, y = X
    return a * x + b * y + c


@register_loader(patterns=(".txt", 32), mime_types=("text/plain", 32), name="AttocubeScanParametersFile", what="Data")
def load_attocube_parameters(new_data, *args, **kargs):
    """Load the scan parameters text file as the metadata for a Data File.

    Args:
        root_name (str):
            The scan prefix e.g. SC_###

    Returns:
        new_data:
            The modififed scan stack.
    """
    filename, args, kargs = get_filename(args, kargs)
    new_data.filename = filename
    with FileManager(filename, "r") as parameters:
        if not parameters.readline().startswith("Daisy Parameter Snapshot"):
            raise StonerLoadError("Parameters file exists but does not have correct header")
        for line in parameters:
            if not line.strip():
                continue
            parts = [x.strip() for x in line.strip().split(":")]
            key = parts[0]
            value = ":".join(parts[1:])
            units = PARAM_RE.match(value)
            if units and units.groups()[1]:
                key += f" [{units.groups()[1]}]"
                value = units.groups()[0]
            new_data[key] = value
    return new_data
