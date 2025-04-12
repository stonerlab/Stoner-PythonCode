#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implement DataFile classes for some generic file formats."""
import contextlib
import io
import re
import sys
import logging

from ...core.exceptions import StonerLoadError
from ...Image import ImageArray
from ...formats.decorators import register_loader
from ...tools.file import get_filename


class _refuse_log(logging.Filter):
    """Refuse to log all records."""

    def filter(self, record):
        """Do not log anything."""
        return False


@contextlib.contextmanager
def catch_sysout(*args):
    """Temporarily redirect sys.stdout and.sys.stdin."""
    stdout, stderr = sys.stdout, sys.stderr
    out = io.StringIO()
    sys.stdout, sys.stderr = out, out
    logger = logging.getLogger("hyperspy.io")
    logger.addFilter(_refuse_log)
    yield None
    logger.removeFilter(_refuse_log)
    sys.stdout, sys.stderr = stdout, stderr
    return


def _delim_detect(line):
    """Detect a delimiter in a line.

    Args:
        line(str):
            String to search for delimiters in.

    Returns:
        (str):
            Delimiter to use.

    Raises:
        StnerLoadError:
            If delimiter cannot be located.
    """
    quotes = re.compile(r"([\"\'])[^\1]*\1")
    line = quotes.sub("", line)  # Remove quoted strings first
    current = (None, len(line))
    for delim in "\t ,;":
        try:
            idx = line.index(delim)
        except ValueError:
            continue
        if idx < current[1]:
            current = (delim, idx)
    if current[0] is None:
        raise StonerLoadError("Unable to find a delimiter in the line")
    return current[0]


@register_loader(
    patterns=[(".tif", 8), (".tiff", 8), (".png", 8), (".npy", 8)],
    mime_types=[("image/tiff", 8), ("image/png", 8), ("application/octet-stream", 8)],
    name="ImageFile",
    what="Image",
)
def load_imagefile(new_image, *args, **kargs):
    """Load an ImageFile by calling the ImageArray method instead."""
    filename, args, kargs = get_filename(args, kargs)
    new_image._image = ImageArray(filename, *args, **kargs)
    for k in new_image._image._public_attrs:
        setattr(new_image, k, getattr(new_image._image, k, None))
    return new_image
