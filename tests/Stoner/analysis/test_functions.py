# -*- coding: utf-8 -*-
"""Tests for Stoner.analysis.functions, including Data.decompose."""

import numpy as np
import pytest

from Stoner import Data
from Stoner.analysis.functions import decompose


def test_decompose_defined_in_analysis_functions():
    """Data.decompose is defined in Stoner.analysis.functions."""
    import inspect
    import Stoner.analysis.functions as funcs

    assert hasattr(funcs, "decompose"), "decompose not found in Stoner.analysis.functions"
    assert callable(funcs.decompose), "decompose is not callable"
    assert hasattr(Data, "decompose"), "Data does not have a decompose method"


def test_decompose_symmetric():
    """decompose correctly extracts the symmetric (even) component."""
    x = np.linspace(-5, 5, 201)
    y_even = x**2  # pure even function
    d = Data(x, y_even, setas="xy", column_headers=["X", "Y"])
    d.decompose()
    # Symmetric component should recover y_even; asymmetric should be ~0
    assert d.shape[1] == 4, "decompose should add two new columns"
    assert d.column_headers[2] == "Symmetric Data"
    assert d.column_headers[3] == "Asymmetric Data"
    np.testing.assert_allclose(d // "Symmetric Data", y_even, atol=0.1)
    np.testing.assert_allclose(d // "Asymmetric Data", 0.0, atol=0.1)


def test_decompose_antisymmetric():
    """decompose correctly extracts the antisymmetric (odd) component."""
    x = np.linspace(-5, 5, 201)
    y_odd = x**3  # pure odd function
    d = Data(x, y_odd, setas="xy", column_headers=["X", "Y"])
    d.decompose()
    # Symmetric component should be ~0; asymmetric should recover y_odd
    np.testing.assert_allclose(d // "Symmetric Data", 0.0, atol=1.0)
    np.testing.assert_allclose(d // "Asymmetric Data", y_odd, atol=1.0)


def test_decompose_mixed():
    """decompose separates mixed even/odd contributions."""
    x = np.linspace(-5, 5, 201)
    y_even = x**2
    y_odd = x
    d = Data(x, y_even + y_odd, setas="xy", column_headers=["X", "Y"])
    d.decompose()
    np.testing.assert_allclose(d // "Symmetric Data", y_even, atol=0.1)
    np.testing.assert_allclose(d // "Asymmetric Data", y_odd, atol=0.1)


def test_decompose_with_explicit_output_columns():
    """decompose writes results into specified existing columns when sym/asym are given."""
    x = np.linspace(-5, 5, 201)
    y_even = x**2
    zeros = np.zeros_like(x)
    d = Data(
        np.column_stack([x, y_even, zeros, zeros]),
        setas="xy..",
        column_headers=["X", "Y", "Sym", "Asym"],
    )
    d.decompose(sym="Sym", asym="Asym", replace=True)
    assert d.column_headers[2] == "Symmetric Data"
    assert d.column_headers[3] == "Asymmetric Data"
    np.testing.assert_allclose(d // "Symmetric Data", y_even, atol=0.1)
    np.testing.assert_allclose(d // "Asymmetric Data", 0.0, atol=0.1)


if __name__ == "__main__":
    pytest.main(["--pdb", __file__])
