# -*- coding: utf-8 -*-
"""
Decoiratoprs for manging loader and saver functions.
"""
from typing import Optional, List, Tuple, Callable
from importlib import import_module
from inspect import signature, _empty
from pathlib import Path
import sys

from ..core.base import SortedMultivalueDict
from ..core.exceptions import StonerLoadError

_loaders_by_type = SortedMultivalueDict()
_loaders_by_pattern = SortedMultivalueDict()
_loaders_by_name = dict()
_savers_by_pattern = SortedMultivalueDict()
_savers_by_name = dict()

LoadQualifier = Optional[str | Tuple[str, int] | List[str] | List[Tuple[str, int]]]


def register_loader(
    mime_types: LoadQualifier = None,
    patterns: LoadQualifier = None,
    priority: Optional[int] = 256,
    name: Optional[str] = None,
    what: Optional[str] = None,
) -> Callable:
    """Decorator to store a loader function indexed by mime_type or extension and priority.

    Keyword Arguments:
        mime_type(str, or list of str, tuple(str, int), list pf tuple(str, int) or None):
            (default None) - Mime-types of files that this loader might work with
        patterns (str or list of str, tuple(str, int), list pf tuple(str, int)  or None):
            (default None) - filenbame extensions of files that this loader might handle
        priority (int):
            (default 256) - sets default order in which loaders are tried when multiple loaders might work
            with a given pattern or mime-type if per pattern/type priorities are not specified.
            Lower priority numbers are tried first.
        name (str, None):
            (default None) - The human readable name for this loader (or the function name if None)
        what (str, None):
            (default None) - The type of object that this loader is going to return - if None then it will try to get
            a return type annotation.

    Notes:
        If not pattern is set, then the default "*" pattern is used.
    """

    def innner(func: Callable) -> Callable:
        """Actual decorator for laoder functions.

        Args:
            func (callable):
                Loader function to be registered.
        """
        if name is None:
            inner_name = func.__name__
        else:
            inner_name = name
        if patterns and not isinstance(patterns, list):
            inner_patterns = [patterns]
        else:
            inner_patterns = patterns
        if mime_types and not isinstance(mime_types, list):
            inner_mime_types = [mime_types]
        else:
            inner_mime_types = mime_types
        func.patterns = inner_patterns
        func.mime_types = inner_mime_types
        func.priority = priority
        func.name = inner_name
        if what is None:
            inner_what = signature(func).return_annotation
            if inner_what == _empty:
                inner_what = None
        else:
            inner_what = what
        func.what = str(inner_what)
        if inner_mime_types:
            for mime_type in inner_mime_types:
                match mime_type:  # Accept either a tuple of (mime_type,priority) or just string mime_type
                    case (str(mime_type), int(p)):  # per mime_type priority
                        _loaders_by_type[mime_type] = (p, func)
                    case str(mime_type):  # Use global priority
                        _loaders_by_type[mime_type] = (func.priority, func)
                    case _:  # Unrecognised
                        raise TypeError("Mime-Type should be a str or (str, int)")
        if inner_patterns:
            for pattern in inner_patterns:
                match pattern:  # Like mime-type, accept tuple or pattern, priority or just string pattern
                    case (str(pattern), int(p)):
                        _loaders_by_pattern[pattern] = (p, func)
                    case str(pattern):  # Bare pattenr - use gloabl priority
                        _loaders_by_pattern[pattern] = (func.priority, func)
                    case _:  # Unrecognised
                        raise TypeError("Pattern shpuld be either a str or (str, int)")
        # All loaders register on the default
        _loaders_by_pattern["*"] = (func.priority, func)
        _loaders_by_name[inner_name] = func
        return func

    return innner


def register_saver(
    patterns: LoadQualifier = None, priority: int = 256, name: Optional[str] = None, what: Optional[str] = None
) -> Callable:
    """Decorator to store a loader function indexed by mime_type or extension and priority.

    Keyword Arguments:
        patterns (str or list of str, tuple(str, int), list pf tuple(str, int)  or None):
            (default None) - filenbame extensions of files that this loader might handle
        priority (int):
            (default 256) - sets default order in which loaders are tried when multiple savers might work
            with a given pattern  if per pattern priorities are not specified.
            Lower priority numbers are tried first.
        name (str, None):
            (default None) - The human readable name for this loader (or the function name if None)
        what (str, None):
            (default None) - The type of object that this loader is going to return - if None then it will try to get
            a return type annotation.

    Notes:
        If not pattern is set, then the default "*" pattern is used.
    """

    def innner(func: Callable) -> Callable:
        """Actual decorator for laoder functions.

        Args:
            func (callable):
                Loader function to be registered.
        """
        if name is None:
            inner_name = func.__name__
        else:
            inner_name = name
        if patterns and not isinstance(patterns, list):
            inner_patterns = [patterns]
        else:
            inner_patterns = patterns
        func.patterns = inner_patterns
        func.priority = priority
        func.name = inner_name
        if what is None:
            inner_what = signature(func).return_annotation
            if inner_what == _empty:
                inner_what = None
        else:
            inner_what = what
        func.what = str(inner_what)
        if inner_patterns:
            for pattern in inner_patterns:
                match pattern:  # Like mime-type, accept tuple or pattern, priority or just string pattern
                    case (str(pattern), int(p)):
                        _savers_by_pattern[pattern] = (p, func)
                    case str(pattern):  # Bare pattenr - use gloabl priority
                        _savers_by_pattern[pattern] = (func.priority, func)
                    case _:  # Unrecognised
                        raise TypeError("Pattern shpuld be either a str or (str, int)")
        # All savers register on the default
        _savers_by_pattern["*"] = (func.priority, func)
        _savers_by_name[inner_name] = func

        return func

    return innner


def next_loader(pattern: Optional[str] = "*", mime_type: Optional[str] = None, what: Optional[str] = None) -> Callable:
    """Find possible loaders and yield them in turn.

    Keyword Args:
        pattern (str, None):
            (deault None) - if the file to load has an extension, use this.
        mime-type (str,None):
            (default None) - if we have a mime-type for the file, use this.
        what (str, None):
            (default None) - limit the return values to things that can load the specified class. If None, then
            this check is skipped.

    Yields:
        func (callable):
            The next loader function to try.

    Notes:
        If avaialbe, iterate through all loaders that match that particular mime-type, but also match the pattern
        if it is available (which is should be!). If mime-type is not specified, then just match by pattern and if
        neither are specified, then yse the default no pattern "*".
    """
    if mime_type is not None and mime_type in _loaders_by_type:  # Prefer mime-type if available
        for func in _loaders_by_type.get_value_list(mime_type):
            if pattern and pattern != "*" and pattern not in func.patterns:
                # If we have both pattern and type, match patterns too.
                continue
            if what and what != func.what:  # If we are limiting what we can load, do that check
                continue
            yield func
    if pattern in _loaders_by_pattern:  # Fall back to specific pattern
        for func in _loaders_by_pattern.get_value_list(pattern):
            if what and what != func.what:  # If we are limiting what we can load, do that check
                continue
            yield func
    if pattern != "*":  # Fall back again to generic pattern
        for func in _loaders_by_pattern.get_value_list("*"):
            if what and what != func.what:  # If we are limiting what we can load, do that check
                continue
            yield func
    return StopIteration


def best_saver(filename: str, name: Optional[str], what: Optional[str] = None) -> Callable:
    """Figure out the best saving routine registerd with the package."""
    if name and name in _savers_by_name:
        return _savers_by_name[name]
    extension = Path(filename).suffix
    if extension in _savers_by_pattern:
        for _, func in _savers_by_pattern[extension]:
            if what is None or (what and what == func.what):
                return func
    for _, func in _savers_by_pattern["*"]:
        if what is None or (what and what == func.what):
            return func
    raise ValueError(f"Unable to find a saving routine for {filename}")


def get_loader(filetype, silent=False):
    """Return the loader function by name.

    Args:
        filetype (str): Filetype to get loader for.
        silent (bool): If False (default) raise a StonerLoadError if filetype doesn't have a loade.

    Returns:
        (callable): Matching loader.

    Notes:
        If the filetype is not found and contains a . then it tries to import a module with the same name int he
        hope that that defines the missing loader. If that fails to work, then either raises StonerLoadError or
        returns None, depending on *silent*.
    """
    try:
        return _loaders_by_name[filetype]
    except KeyError as err:
        if "." in filetype:
            try:
                module_name = ".".join(filetype.split(".")[:-1])
                if module_name not in sys.modules:
                    import_module(module_name)
                ret = _loaders_by_name.get(filetype, _loaders_by_name.get(filetype.split(".")[-1], None))
                if ret:
                    return ret
            except ImportError:
                pass
        if not silent:
            raise StonerLoadError(f"Cannot locate a loader function for {filetype}") from err


def get_saver(filetype, silent=False):
    """Return the saver function by name.

    Args:
        filetype (str): Filetype to get saver for.
        silent (bool): If False (default) raise a Stonersaverror if filetype doesn't have a loade.

    Returns:
        (callable): Matching saver.

    Notes:
        If the filetype is not found and contains a . then it tries to import a module with the same name int he
        hope that that defines the missing saver. If that fails to work, then either raises Stonersaverror or
        returns None, depending on *silent*.
    """
    try:
        return _savers_by_name[filetype]
    except KeyError as err:
        if "." in filetype:
            try:
                module_name = ".".join(filetype.split(".")[:-1])
                if module_name not in sys.modules:
                    import_module(module_name)
                ret = _savers_by_name.get(filetype, _savers_by_name.get(filetype.split(".")[-1], None))
                if ret:
                    return ret
            except ImportError:
                pass
        if not silent:
            raise StonerLoadError(f"Cannot locate a loader function for {filetype}") from err


def clear_routine(name, loader=True, saver=True):
    """Remove the routine with the specified name from the registered loaders and/or savers.

    Args:
        name (str): Name of routine to remove

    Keyword Arguments:
        loader (bool):
            Whether to remove tyhe loader (default True)
        saver (bool):
            Whether to remove the saver routne (default True)

    Returns:
        (dict):
            Removed kiader and saver routines.
    """
    ret = {}
    for k, lookup, pattern_lookup, type_lookup in zip(
        ["loader", "saver"],
        [_loaders_by_name, _savers_by_name],
        [_loaders_by_pattern, _savers_by_pattern],
        [_loaders_by_type, None],
    ):
        if not locals()[k]:
            continue
        func = lookup.pop(name, None)
        if func is None:
            continue
        ret[k] = func
        for lookup_dict in [pattern_lookup, type_lookup]:
            if not isinstance(lookup_dict, dict):
                continue
            for _, values in _loaders_by_pattern.items():
                remove = []
                for ix, (priority, loader) in enumerate(values):
                    if loader is func:
                        remove.append(ix)
                if remove:
                    remove.reverse()
                    for ix in remove:
                        del values[ix]
    return ret
