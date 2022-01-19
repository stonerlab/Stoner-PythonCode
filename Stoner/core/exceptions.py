#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stoner Package specific exceptions."""


class StonerLoadError(Exception):

    """An exception thrown by the file loading routines in the Stoner Package.

    This special exception is thrown when one of the loader functions of :py:class:`Stoner.Core.DataFile`
    attmpts and then fails to load some data from disk. Generally speaking this is not a real
    error, but simply indicates that the file format is not recognised by that particular function,
    and thus another loader function should have a go instead.
    """


class StonerSaveError(Exception):

    """An exception thrown by the file saving routines in the Stoner Package.

    This special exception is thrown when one of the saver functions  of :py:class:`Stoner.Core.DataFile`
    attmpts and then fails to save some data to disk. This is likely to be a real error in most cases, but
    is non-fatal to the saving process until we reach a point where there are no more routines to try to use.
    """


class StonerUnrecognisedFormat(IOError):

    """An exception thrown by the file loading routines in the Stoner Package.

    This special exception is thrown when none of the subclasses was able to load the specified file.
    """


class StonerSetasError(AttributeError):

    """An exception tjrown when we try to access a column in data without setas being set."""


class StonerAssertionError(RuntimeError):

    """An exception raised when the library thinks an assertion has failed."""


def assertion(condition, message="Library Assertion Error set"):
    """Raise an error when condition is false.

    A utility functiuon to be used when assert might have been."""
    if not condition:
        raise StonerAssertionError(message)
