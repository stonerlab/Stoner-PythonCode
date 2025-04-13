# -*- coding: utf-8 -*-
"""Decoiratoprs for manging loader and saver functions."""
from typing import Optional, List, Tuple, Callable
from inspect import signature, _empty

from ..tools.file import _loaders_by_type, _loaders_by_pattern, _loaders_by_name, _savers_by_pattern, _savers_by_name


LoadQualifier = Optional[str | Tuple[str, int] | List[str] | List[Tuple[str, int]]]


def register_loader(
    mime_types: LoadQualifier = None,
    patterns: LoadQualifier = None,
    priority: Optional[int] = 256,
    name: Optional[str] = None,
    what: Optional[str] = None,
) -> Callable:
    """Store a loader function indexed by mime_type or extension and priority.

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
    """Store a loader function indexed by mime_type or extension and priority.

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
