"""Provide registered loaders and savers for supported file formats.

Users normally construct :py:class:`Stoner.core.data.Data` or
:py:class:`Stoner.Image.core.ImageFile` objects and let Stoner select a suitable
format handler. Loader and saver functions are registered with
:py:func:`Stoner.formats.decorators.register_loader` and
:py:func:`Stoner.formats.decorators.register_saver`.

Handlers can advertise filename patterns and MIME types. When several handlers
match, lower priority numbers are tried first. A loader must positively identify
its input where possible and raise
:py:exc:`Stoner.core.exceptions.StonerLoadError` if the file is not in the
expected format, allowing the next candidate to be tried.
"""

# __all__ = ["instruments", "generic", "rigs", "facilities", "simulations", "attocube", "maximus"]
# from . import instruments, generic, rigs, facilities, simulations, attocube, maximus
__all__ = ["data", "image"]
from . import data, image
