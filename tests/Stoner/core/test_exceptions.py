# -*- coding: utf-8 -*-
"""Test Stoner.core.exceptions module"""
import pytest

from Stoner.core.exceptions import (
    StonerAssertionError,
    StonerLoadError,
    StonerSetasError,
    StonerUnrecognisedFormat,
    assertion,
)


def test_assertion():
    with pytest.raises(RuntimeError):
        assertion(False, "Triggered an assertion")


def test_assertion_true_does_not_raise():
    # Calling assertion with a truthy condition should not raise
    assertion(True, "Should not raise")


def test_StonerLoadError_is_exception():
    err = StonerLoadError("test load error")
    assert isinstance(err, Exception), "StonerLoadError should be an Exception"
    with pytest.raises(StonerLoadError):
        raise StonerLoadError("could not load file")


def test_StonerUnrecognisedFormat_is_IOError():
    err = StonerUnrecognisedFormat("unknown format")
    assert isinstance(err, IOError), "StonerUnrecognisedFormat should be an IOError"
    with pytest.raises(StonerUnrecognisedFormat):
        raise StonerUnrecognisedFormat("no loader found")


def test_StonerSetasError_is_AttributeError():
    err = StonerSetasError("setas not set")
    assert isinstance(err, AttributeError), "StonerSetasError should be an AttributeError"
    with pytest.raises(StonerSetasError):
        raise StonerSetasError("column not accessible")


def test_StonerAssertionError_is_RuntimeError():
    err = StonerAssertionError("assertion failed")
    assert isinstance(err, RuntimeError), "StonerAssertionError should be a RuntimeError"
    with pytest.raises(StonerAssertionError):
        assertion(False)


if __name__ == "__main__":
    pytest.main(["--pdb", __file__])

