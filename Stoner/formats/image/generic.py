#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implement DataFile classes for some generic file formats."""

import contextlib
import io
import logging
import re
import sys
from importlib import import_module

from ...core.exceptions import StonerLoadError
from ...formats.decorators import register_loader
from ...tools.decorators import make_Class
from ...tools.file import get_filename

try:
    import rsciio
except ImportError:
    rsciio = None


class _refuse_log(logging.Filter):
    """Refuse to log all records."""

    def filter(self, record):
        """Do not log anything."""
        return False


@contextlib.contextmanager
def catch_sysout():
    """Temporarily redirect sys.stdout and.sys.stdin."""
    stdout, stderr = sys.stdout, sys.stderr
    out = io.StringIO()
    sys.stdout, sys.stderr = out, out
    logger = logging.getLogger("hyperspy.io")
    log_filter = _refuse_log()
    logger.addFilter(log_filter)
    try:
        yield None
    finally:
        logger.removeFilter(log_filter)
        sys.stdout, sys.stderr = stdout, stderr


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
def load_imagefile(new_image, *args, **kwargs):
    """Read numerical pixels and publish them through the image owner."""
    filename, args, kwargs = get_filename(args, kwargs)
    from ...Image.numerical import load_pixels
    try:
        loaded = load_pixels(filename, **kwargs)
    except (OSError, ValueError, TypeError) as error:
        raise StonerLoadError(f"Cannot load image pixels from {filename}") from error
    new_image.image = loaded
    for k in loaded._public_attrs:
        setattr(new_image, k, getattr(loaded, k, None))
    return new_image


if rsciio:
    for plugin in rsciio.IO_PLUGINS:
        patterns = [(f".{ext}", 64) for ext in plugin["file_extensions"]]
        mime_types = []
        name = plugin["name"]
        WHAT = "Image"

        CODE = f'''
def load_{name}file(new_image, *args, **kwargs):
    """Use a RosettaSciIO plugin to load an image file."""
    filename, args, kwargs = get_filename(args, kwargs)
    try:
        api=import_module(plugin["api"])
        data=api.file_reader(filename)[0]
    except Exception as err:
        print(name,err)
        raise StonerLoadError
    new_image.image=data["data"]
    new_image.metadata.update(data["metadata"])
    new_image.metadata["axes"]=data["axes"]
    return new_image
        '''
        namespace = {
            "get_filename": get_filename,
            "import_module": import_module,
            "plugin": plugin,
            "name": name,
            "make_Class": make_Class,
            "StonerLoadError": StonerLoadError,
        }
        exec(CODE, namespace)  # nosec pylint: disable=exec-used
        func = namespace[f"load_{name}file"]
        register_loader(patterns=patterns, mime_types=mime_types, name=name, what=WHAT)(func)
        setattr(sys.modules[__name__], f"load_{name}file", func)
