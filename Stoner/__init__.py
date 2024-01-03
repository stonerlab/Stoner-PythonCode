"""Provides utility classes for writing simple data analysis scripts more easily.

It has been developed by members of the `Condensed Matter Group<http://www.stoner.leeds.ac.uk/>` at the
`University  of Leeds<http://www.leeds.ac.uk>`.
"""
import importlib
import pathlib

from .tools import Options as _Options

__all__ = [
    "core",
    "analysis",
    "formats",
    "plot",
    "tools",
    "Image",
    "Data",
    "DataFolder",
    "ImageFile",
    "ImageFolder",
    "ImageStack",
    "set_option",
    "get_option",
    "Options",
]
_sub_imports = {
    "Data": "core.data",
    "DataFolder": "Folders",
    "ImageFile": "Image",
    "ImageFolder": "Image",
    "ImageStack": "Image",
    "set_option": "tools",
    "get_optiopns": "tools",
}
# These fake the old namespace if you do an import Stoner


Options = _Options()


__version_info__ = ("0", "11", "0")
__version__ = ".".join(__version_info__)

__homepath__ = pathlib.Path(__file__).parent.resolve()
__datapath__ = (__homepath__ / ".." / "sample-data").resolve()
__home__ = str(__homepath__)


def __getattr__(name):
    """Lazy import required module."""
    if name in __all__:
        if name in _sub_imports:
            ret = importlib.import_module(f".{_sub_imports[name]}", __name__)
            return getattr(ret, name)
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
