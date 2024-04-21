# -*- coding: utf-8 -*-
"""Test Stoner.core.exceptions module"""
import pytest
from Stoner.core.exceptions import assertion


def test_assertion():
    with pytest.raises(RuntimeError):
        assertion(False, "Triggered an assertion")


if __name__ == "__main__":
    pytest.main(["--pdb", __file__])
