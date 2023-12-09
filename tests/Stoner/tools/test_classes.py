# -*- coding: utf-8 -*-
"""Tests for Stoner.tools.classes"""


import pytest
from Stoner.tools import classes
from Stoner import Options


def test_typedList_create():
    tl = classes.typedList()
    assert tl._type == str, "Empty typedList should default to str"
    tl = classes.typedList(int)
    assert tl._type == int and len(tl) == 0, "typedList with single type constructor failed"
    tl = classes.typedList(int, [1, 2, 3, 4, 5])
    assert tl._type == int and len(tl) == 5, "typedList with type and iterator constructor failed"
    try:
        tl = classes.typedList(int, 1, 2, 3)
    except SyntaxError:
        pass
    else:
        assert False, "typedList with too many arguments didn't throw error"
    try:
        tl = classes.typedList(int, ["A", 1, 2, 3])
    except TypeError:
        pass
    else:
        assert False, "typedList with with bad types didn't throw error"


def test_typedList_operators():
    tl1 = classes.typedList(int, (1, 2, 3))
    tl2 = classes.typedList(int, (4, 5, 6))
    tl3 = classes.typedList(float, (1.1, 1.2, 1.3))
    assert len(tl1 + tl2) == 6, "Failed to add similar typed lists"
    try:
        _ = tl1 + tl3
    except TypeError:
        pass
    else:
        assert False, "Adding unlike typed lists didn't raise TyperError"
    tl1 += tl2
    assert len(tl1) == 6, "In place add of typedList failed"
    try:
        tl1 += 2
    except TypeError:
        pass
    else:
        assert False, "Adding non list to a typed lists didn't raise TyperError"
    assert [True, 2] + tl3 == [True, 2, 1.1, 1.2, 1.3], "Right adding typedList failed."
    tl1 = classes.typedList(int, (1, 2, 3))
    tl2 = classes.typedList(int, (1, 2, 3))
    assert tl1 == tl2, "typedList equality check failed"

    del tl1[0]
    assert tl1 == [2, 3], "Failed to delete and item from the list"
    assert tl1[0] == 2, "typedList get item failed"
    assert repr(tl1) == "[2, 3]", "typedList representation failed"
    try:
        tl1[[0, 1]] = 3
    except TypeError:
        pass
    else:
        assert (
            False
        ), "typedList setitem with mismatch between iterable key and scalar value failed to rasise a TypeError"
    try:
        tl1[[0, 1]] = 3.0
    except TypeError:
        pass
    else:
        assert False, "typedList setitem with from type of value failed to rasise a TypeError"


def test_typedList_methods():
    tl1 = classes.typedList(int, (1, 2, 3))
    try:
        tl1.extend(1)
    except TypeError:
        pass
    else:
        assert False, "typedList failed to raise typeError when extending with a scale"
    assert tl1.index(2) == 1, "typedList index failed to give right answer"
    tl1.insert(0, 0)
    assert tl1 == [0, 1, 2, 3], "typedList insert at start failed."
    try:
        tl1.insert(0, "G")
    except TypeError:
        pass
    else:
        assert False, "typedList failed to raise typeError when inserting with the wrong type"


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
