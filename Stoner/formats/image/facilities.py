#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implements DataFile like classes for various large scale facilities."""

# Standard Library imports
from ...core.exceptions import StonerLoadError
from ...tools.file import get_filename

from ..decorators import register_loader

try:
    import fabio
except ImportError:
    fabio = None

if fabio:

    @register_loader(
        patterns=(".edf", 32),
        mime_types=[("text/plain", 32), ("application/octet-stream", 32)],
        name="FabioImage",
        what="Image",
    )
    def load_fabio(new_data, *args, **kargs):
        """Load function. File format has space delimited columns from row 3 onwards."""
        filename, args, kargs = get_filename(args, kargs)
        if filename is None or not filename:
            new_data.get_filename("r")
        else:
            new_data.filename = filename
        try:
            img = fabio.open(new_data.filename)
            new_data.image = img.data
            new_data.metadata.update(img.header)
            return new_data
        except (OSError, ValueError, TypeError, IndexError) as err:
            raise StonerLoadError("Not a Fabio Image file !") from err
