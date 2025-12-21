# -*- coding: utf-8 -*-
"""Tests for Stoner.tools.classes"""


import pytest

from Stoner import Options
from Stoner.tools import classes


def test_TypedList_create():
    tl = classes.TypedList()
    assert tl._type == str, "Empty TypedList should default to str"
    tl = classes.TypedList(int)
    assert tl._type == int and len(tl) == 0, "TypedList with single type constructor failed"
    tl = classes.TypedList(int, [1, 2, 3, 4, 5])
    assert tl._type == int and len(tl) == 5, "TypedList with type and iterator constructor failed"
    try:
        tl = classes.TypedList(int, 1, 2, 3)
    except SyntaxError:
        pass
    else:
        assert False, "TypedList with too many arguments didn't throw error"
    try:
        tl = classes.TypedList(int, ["A", 1, 2, 3])
    except TypeError:
        pass
    else:
        assert False, "TypedList with with bad types didn't throw error"


def test_TypedList_operators():
    tl1 = classes.TypedList(int, (1, 2, 3))
    tl2 = classes.TypedList(int, (4, 5, 6))
    tl3 = classes.TypedList(float, (1.1, 1.2, 1.3))
    assert len(tl1 + tl2) == 6, "Failed to add similar typed lists"
    try:
        _ = tl1 + tl3
    except TypeError:
        pass
    else:
        assert False, "Adding unlike typed lists didn't raise TyperError"
    tl1 += tl2
    assert len(tl1) == 6, "In place add of TypedList failed"
    try:
        tl1 += 2
    except TypeError:
        pass
    else:
        assert False, "Adding non list to a typed lists didn't raise TyperError"
    assert [True, 2] + tl3 == [True, 2, 1.1, 1.2, 1.3], "Right adding TypedList failed."
    tl1 = classes.TypedList(int, (1, 2, 3))
    tl2 = classes.TypedList(int, (1, 2, 3))
    assert tl1 == tl2, "TypedList equality check failed"

    del tl1[0]
    assert tl1 == [2, 3], "Failed to delete and item from the list"
    assert tl1[0] == 2, "TypedList get item failed"
    assert repr(tl1) == "[2, 3]", "TypedList representation failed"
    try:
        tl1[[0, 1]] = 3
    except TypeError:
        pass
    else:
        assert (
            False
        ), "TypedList setitem with mismatch between iterable key and scalar value failed to rasise a TypeError"
    try:
        tl1[[0, 1]] = 3.0
    except TypeError:
        pass
    else:
        assert False, "TypedList setitem with from type of value failed to rasise a TypeError"


def test_TypedList_methods():
    tl1 = classes.TypedList(int, (1, 2, 3))
    try:
        tl1.extend(1)
    except TypeError:
        pass
    else:
        assert False, "TypedList failed to raise typeError when extending with a scale"
    assert tl1.index(2) == 1, "TypedList index failed to give right answer"
    tl1.insert(0, 0)
    assert tl1 == [0, 1, 2, 3], "TypedList insert at start failed."
    try:
        tl1.insert(0, "G")
    except TypeError:
        pass
    else:
        assert False, "TypedList failed to raise typeError when inserting with the wrong type"


def test_Options():
    try:
        classes.get_option("Random")
    except IndexError:
        pass
    else:
        assert False, "get_option didn't raise an IndexError for a bad option name"
    try:
        classes.set_option("Random", False)
    except IndexError:
        pass
    else:
        assert False, "set_option didn't raise an IndexError for a bad option name"
    try:
        classes.set_option("no_figs", "Hi")
    except ValueError:
        pass
    else:
        assert False, "set_option didn't raise a ValueError for a bad option value"

    try:
        Options.Random = False
    except AttributeError:
        pass
    else:
        assert False, "Set Option attribute didn't raise an IndexError for a bad option name"
    try:
        Options.no_figs = "Hi"
    except ValueError:
        pass
    else:
        assert False, "Set Options attribute didn't raise a ValueError for a bad option value"
    Options.no_figs = False
    assert not Options.no_figs, "Setting Options attribute didn't stick"
    del Options.no_figs
    assert Options.no_figs, "Deleting Options attrkibute didn't clear option"
    assert dir(Options) == [
        "multiprocessing",
        "no_figs",
        "short_data_repr",
        "short_folder_rrepr",
        "short_img_repr",
        "short_repr",
        "threading",
        "warnings",
    ], "Directory of Options failed"
    opt_repr = (
        "Stoner Package Options\n~~~~~~~~~~~~~~~~~~~~~~\n"
        + "multiprocessing : False\nno_figs : True\nshort_data_repr : False\nshort_folder_rrepr : True\n"
        + "short_img_repr : True\nshort_repr : False\nthreading : False\nwarnings : False\n"
    )
    assert repr(Options) == opt_repr, "Representation of Options failed"


if __name__ == "__main__":
    pytest.main(["--pdb", __file__])
