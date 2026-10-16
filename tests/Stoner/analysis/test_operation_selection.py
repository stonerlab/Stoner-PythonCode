"""Protect selected fits, full-axis evaluation and atomic publication."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import curve_fit

from Stoner import Data


def line(x, slope, offset):
    """Evaluate a linear background."""
    return slope * x + offset


def test_weighted_background_selection_matches_explicit_fit():
    x = np.linspace(-10, 10, 41)
    y = line(x, 2, 3) + 0.1 * np.sin(x)
    y[np.abs(x) <= 3] += 20
    sigma = np.linspace(0.1, 0.8, len(x))
    y[2] = np.nan
    data = Data(x, y, sigma, np.full(len(x), np.nan), setas="xye.")
    data.mask[1, 1] = True
    data.mask[:, 3] = True
    before = data.export_storage()
    where = pd.Series(np.abs(x) > 3)
    eligible = where.to_numpy() & np.isfinite(y)
    eligible[1] = False
    expected = curve_fit(line, x[eligible], y[eligible], sigma=sigma[eligible], absolute_sigma=True)
    result = data.curve_fit(line, where=where, output="report")
    np.testing.assert_allclose(result.popt, expected[0])
    np.testing.assert_allclose(result.pcov, expected[1])
    assert result.N == eligible.sum()
    np.testing.assert_allclose(result.residual_vals, y[eligible] - line(x[eligible], *result.popt))
    np.testing.assert_allclose(data.to_numpy(masked=False), before.values.to_numpy())
    np.testing.assert_array_equal(data.mask, before.excluded)
    assert data.metadata == before.metadata
    np.testing.assert_allclose(result.fit_values, line(x, *result.popt))


def test_channels_resolve_exclusions_independently_and_publish_full_results():
    x = np.arange(12.)
    first, second = line(x, 2, 3), line(x, -1, 4)
    first[1], second[2] = 1000, -1000
    data = Data(x, first, second, setas="xyy")
    data.mask[1, 1] = True
    data.mask[2, 2] = True
    results = data.curve_fit(line, where=x < 8, result=True, residuals=True)
    np.testing.assert_allclose(results[0][0], [2, 3])
    np.testing.assert_allclose(results[1][0], [-1, 4])
    assert data.shape == (12, 7)
    np.testing.assert_allclose(data.to_numpy(masked=False)[:, 3], line(x, 2, 3))
    np.testing.assert_allclose(data.to_numpy(masked=False)[:, 5], line(x, -1, 4))
    assert not np.any(np.asarray(data.mask)[:, [3, 5]])
    assert data.mask[1, 4] and data.mask[2, 6]


def test_later_fit_failure_does_not_publish_earlier_results():
    x = np.arange(10.)
    data = Data(x, line(x, 2, 3), line(x, 4, 5), setas="xyy")
    data.mask[2:, 2] = True
    data["existing"] = "keep"
    before = data.export_storage()
    with pytest.raises(ValueError, match="more eligible"):
        data.curve_fit(line, result=True)
    np.testing.assert_array_equal(data.to_numpy(masked=False), before.values.to_numpy())
    np.testing.assert_array_equal(data.mask, before.excluded)
    assert data.metadata == before.metadata
    assert data.column_ids == tuple(column.id for column in before.schema)


def test_external_uncertainties_and_parameter_bounds_intersect_legacy_region():
    x = np.arange(20.)
    data = Data(x, line(x, 2, 3), setas="xy")
    sigma = np.linspace(0.1, 1, len(x))
    where = x % 2 == 0
    eligible = where & (x > 2) & (x <= 16)
    expected = curve_fit(line, x[eligible], data.to_numpy(masked=False)[eligible, 1],
                         sigma=sigma[eligible], bounds=([0, 0], [1.5, 10]), absolute_sigma=True)
    actual = data.curve_fit(line, sigma=sigma, where=where, bounds=(2, 16),
                            parameter_bounds=([0, 0], [1.5, 10]))
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])


@pytest.mark.parametrize("where", [True, np.ones(10), np.ones((10, 1), dtype=bool),
                                   pd.Series([True] * 10, index=range(9, -1, -1)),
                                   pd.Series([True] * 9 + [pd.NA], dtype="boolean")])
def test_bad_selection_does_not_mutate(where):
    data = Data(np.arange(10.), np.arange(10.), setas="xy")
    before = data.export_storage()
    with pytest.raises(ValueError, match="where"):
        data.curve_fit(line, where=where, result=True)
    np.testing.assert_array_equal(data.to_numpy(masked=False), before.values.to_numpy())
    assert data.metadata == before.metadata


def test_invalid_uncertainties_are_checked_only_in_eligible_region():
    x = np.arange(10.)
    data = Data(x, line(x, 2, 3), setas="xy")
    sigma = np.ones(10)
    sigma[0] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        data.curve_fit(line, sigma=sigma, result=True)
    actual = data.curve_fit(line, sigma=sigma, where=x > 0)
    np.testing.assert_allclose(actual[0], [2, 3])
    with pytest.raises(ValueError, match="complete row count"):
        data.curve_fit(line, sigma=sigma[1:], where=x > 0)


def test_selected_fit_does_not_evaluate_model_outside_its_domain():
    x = np.arange(-5., 6.)
    data = Data(x, np.abs(x), setas="xy")

    def positive_line(values, slope, offset):
        if np.any(values <= 0):
            raise ValueError("Model requires positive x")
        return line(values, slope, offset)

    parameters, _ = data.curve_fit(positive_line, where=x > 0, p0=[1, 0])
    np.testing.assert_allclose(parameters, [1, 0], atol=1e-8)
    with pytest.raises(ValueError, match="more eligible"):
        data.curve_fit(line, where=np.zeros(len(x), dtype=bool))


def test_result_publication_failure_leaves_owner_unchanged(monkeypatch):
    from Stoner.analysis.fitting import functions

    data = Data(np.arange(10.), np.arange(10.), setas="xy")
    before = data.export_storage()

    def failing_record(draft, *args):
        draft["partial"] = "must not escape"
        draft.add_column(np.zeros(len(draft)))
        raise RuntimeError("publication failed")

    monkeypatch.setattr(functions, "_record_curve_fit_result", failing_record)
    with pytest.raises(RuntimeError, match="publication failed"):
        data.curve_fit(line, result=True)
    np.testing.assert_array_equal(data.to_numpy(masked=False), before.values.to_numpy())
    assert data.metadata == before.metadata
