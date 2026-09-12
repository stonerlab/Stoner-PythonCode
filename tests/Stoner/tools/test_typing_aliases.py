"""Ensure deferred typing aliases resolve to the public package classes."""

from typing import get_args, get_type_hints

import numpy as np

from Stoner import Data, ImageFile
from Stoner.Image.core import ImageArray
from Stoner.core.setas import Setas
from Stoner.tools import typing as aliases


def test_runtime_class_aliases():
    def annotated(data: aliases.Data, image: aliases.ImageArray, file: aliases.ImageFile, roles: aliases.Setas):
        pass

    hints = get_type_hints(annotated)
    assert hints["data"] is Data
    assert hints["image"] is ImageArray
    assert hints["file"] is ImageFile
    assert Setas in get_args(hints["roles"])
    assert get_type_hints(Data.add)["return"] is Data


def test_threshold_annotation_and_returns():
    hints = get_type_hints(Data.threshold)
    assert Data not in get_args(hints["return"])
    assert np.ndarray in get_args(hints["return"])
    data = Data(np.column_stack((np.arange(5.), np.arange(5.))), setas="xy")
    assert np.isclose(data.threshold(1.5), 1.5)
    assert np.isclose(data.threshold(1.5, all_vals=True), 1.5)
    assert isinstance(data.threshold([1.5, 2.5]), np.ndarray)
    assert isinstance(data.threshold(1.5, xcol=False), np.ndarray)
    assert data.threshold(10.) == []
