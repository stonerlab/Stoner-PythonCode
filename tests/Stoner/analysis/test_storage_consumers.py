"""Check scientific consumers preserve the frame owner's exclusions and schema."""

import numpy as np
import pytest

from Stoner import Data


def linear(x, slope, intercept):
    """Evaluate a straight line."""
    return slope * x + intercept


@pytest.mark.parametrize("result", [None, True, 0])
def test_fit_preserves_source_masks_and_columns(result):
    x = np.arange(20, dtype=float)
    data = Data(x, linear(x, 2, 3), setas="xy", column_headers=["x", "y"])
    data.mask[3, 0] = True
    data.mask[7, 1] = True
    before = data.export_storage()
    ids = data.column_ids
    data.curve_fit(linear, p0=[1, 1], result=result, residuals=True, replace=False)
    expected_width = 2 if result is None else 4
    assert data.shape == (20, expected_width)
    for old_index, column_id in enumerate(ids):
        index = data.column_ids.index(column_id)
        np.testing.assert_array_equal(data.to_numpy(masked=False)[:, index], before.values.iloc[:, old_index])
        np.testing.assert_array_equal(np.asarray(data.mask)[:, index], before.excluded[:, old_index])
    if result is not None:
        fit_index = 2 if result is True else 0
        np.testing.assert_allclose(data.to_numpy(masked=False)[:, fit_index], linear(x, 2, 3), atol=1e-8)
        np.testing.assert_array_equal(np.asarray(data.mask)[:, fit_index], np.any(before.excluded, axis=1))
        np.testing.assert_allclose(data.to_numpy()[:, fit_index + 1].compressed(), 0, atol=1e-8)


def test_outlier_callback_rolls_back_numeric_edits():
    data = Data(np.arange(20, dtype=float), np.ones(20), setas="xy")
    before = data.to_numpy()

    def detector(row, window, **kwargs):
        return row[0] == 10

    def action(index, column, draft):
        draft[index, column] = 99
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        data.outlier_detection(func=detector, action=action)
    np.testing.assert_array_equal(data.to_numpy(), before)


def test_deduplicate_multiple_keys_and_masked_average():
    data = Data(np.array([[1., 2., 4.], [1., 2., 8.], [1., 3., 10.], [1., 3., 20.]]))
    data.mask[3, 2] = True
    result = data.deduplicate([0, 1], action="average")
    np.testing.assert_array_equal(result.to_numpy(), [[1, 2, 6], [1, 3, 10]])
    assert result.column_ids == data.column_ids
    assert data.shape == (4, 3)


def test_bin_retains_counts_and_full_headers():
    x = np.linspace(1, 2, 20)
    data = Data(x, 2 * x, setas="xy", column_headers=["Position", "Signal"])
    result = data.bin(bins=np.array([0.99, 1.25, 1.5, 1.75, 2.01]))
    assert result.shape == (4, 4)
    assert result.column_headers == ["Position", "Signal", "dSignal", "#/bin Signal"]
    assert str(result.setas) == "xye."
    assert np.sum(result.column(3)) == len(data)


def test_row_selection_with_column_slice_keeps_rectangular_shape():
    data = Data(np.arange(30).reshape(10, 3))
    data.mask[7, 1] = True
    selected = data[[2, 7], :]
    np.testing.assert_array_equal(selected.data, [[6, 7, 8], [21, 22, 23]])
    np.testing.assert_array_equal(selected.mask, [[False, False, False], [False, True, False]])
    data[[2, 7], :] = 5
    np.testing.assert_array_equal(data[[2, 7], :], 5)


def test_append_dictionary_with_latex_header():
    data = Data()
    header = r"Field $\mu_0 H$"
    data += {header: 1.0}
    data += {header: 2.0}
    assert data.column_headers == [header]
    np.testing.assert_array_equal(data.column(header), [1, 2])


def test_tolerant_duplicate_search_does_not_change_source():
    data = Data(np.array([[1., 4.], [1.000001, 8.], [2., 10.]]), setas="xy")
    before = data.to_numpy()
    assert list(data.find_duplicates(delta=1e-4).values()) == [[0, 1], [2]]
    np.testing.assert_array_equal(data.to_numpy(), before)
    ids = data.column_ids
    data.remove_duplicates(delta=1e-4, strategy="average")
    np.testing.assert_allclose(data.column(1), [6, 10])
    assert data.column_ids == ids
