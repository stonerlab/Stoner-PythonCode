"""Numerical regression coverage for symmetric and hysteretic decomposition."""

import numpy as np
from numpy.testing import assert_allclose

from Stoner import Data


def test_hysteretic_decomposition():
    """Separate an even offset from an odd loop on rising and falling branches."""
    x = np.concatenate([np.linspace(-2, 2, 201), np.linspace(2, -2, 201)[1:]])
    y = np.concatenate([3 + x[:201] + 0.5, 3 + x[201:] - 0.5])
    data = Data(np.column_stack([x, y]), setas='xy')
    result = data.decompose(hysteretic=True)
    assert result is data
    # Exclude the turning points affected by branch detection and interpolation.
    interior = np.abs(x) < 1.5
    assert_allclose(data.data[interior, -2], 3, atol=1e-10)
    assert_allclose(data.data[interior, -1], x[interior] + 0.5, atol=1e-10)
