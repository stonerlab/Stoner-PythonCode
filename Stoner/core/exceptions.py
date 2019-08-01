#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stoner Package specific exceptions
"""


class StonerLoadError(Exception):

    """An exception thrown by the file loading routines in the Stoner Package.

    This special exception is thrown when one of the subclasses of :py:class:`Stoner.Core.DataFile`
    attmpts and then fails to load some data from disk. Generally speaking this is not a real
    error, but simply indicates that the file format is not recognised by that particular subclass,
    and thus another subclass should have a go instead.
    """

    pass


class StonerUnrecognisedFormat(IOError):

    """An exception thrown by the file loading routines in the Stoner Package.

    This special exception is thrown when none of the subclasses was able to load the specified file.
    """

    pass


class StonerSetasError(AttributeError):

    """An exception tjrown when we try to access a column in data without setas being set."""

    pass


class StonerAssertionError(RuntimeError):

    """An exception raised when the library thinks an assertion has failed."""

    pass


def assertion(condition, message="Library Assertion Error set"):
    """A utility functiuon to be used when assert might have been."""
    if not condition:
        raise StonerAssertionError(message)
