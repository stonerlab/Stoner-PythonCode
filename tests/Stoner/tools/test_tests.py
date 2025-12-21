# -*- coding: utf-8 -*-
"""Test Stoner.tools.tests module."""


import numpy as np
import pytest

from Stoner import Data
from Stoner.tools import tests


def test_all_size():
    x1 = [1] * 5
    x2 = [2] * 5
    assert tests.all_size((x1, x2), 5), "Failed all_size with lists"
    x1 = np.array(x1)
    x2 = np.array(x2)
    assert tests.all_size((x1, x2)), "Failed all_size with arrays"


def test_all_type():
    x1 = [1] * 5
    assert tests.all_type(x1, int), "Failed all_type with list"
    x1 = np.array(x1)
    assert tests.all_type(x1, int), "Failed all type with array"


def test_is_AnyNone():
    x1 = [1] * 5
    x1[3] = None
    assert tests.isanynone(*x1), "is_AbyNione failed."


def test_is_Comparable():
    x1 = np.ones(10)
    x2 = np.zeros(10)
    assert tests.iscomparable(x1, x2), "is_Comparable with arrays failed."
    x1 = "Hello"
    x2 = "World"
    assert tests.iscomparable(x1, x2), "Failed is_Comparable with type error."
    x1 = 3
    x2 = 5
    assert tests.iscomparable(x1, x2), "is_Comparable failed with integers"


def test_is_Iterable():
    assert tests.isiterable(list()), "Is Iterable failed"


def test_is_like_list():
    assert tests.islistlike(tuple()), "islistlike failed."


def test_is_None():
    assert tests.isnone(None), "isnone with None failed"
    assert tests.isnone([None, None, None]), "isnone with list oif None failed"

    def g():
        while True:
            yield None

    assert tests.isnone(g()), "isnone with a generator failed"


def test_is_property():
    d = Data()
    assert tests.isproperty(Data, "data"), "Failed to test isProperty with class"
    assert tests.isproperty(d, "data"), "Failed to test isProperty with instance"


def test_is_tuple():
    assert not tests.istuple(5), "istuple though 5 was a tuple"
    assert not tests.istuple((5,), int, int), "istuple failed to short circuit length check"
    assert tests.istuple((5,)), "istuple didn't think (5,) is a tuple"
    assert tests.istuple((4, "Hi"), int, str), "istuple failed to match types"
    assert not tests.istuple((4, "Hi"), str, int), "istuple failed to match types as bad"


if __name__ == "__main__":
    pytest.main()
