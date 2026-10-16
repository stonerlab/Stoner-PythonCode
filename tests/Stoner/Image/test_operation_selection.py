"""Check calibrated regional reductions without modifying measurements."""

import numpy as np
import pytest
import xarray as xr

from Stoner.Image import ImageFile, ImageStack


def test_calibrated_region_preserves_raw_integer_pixels():
    image = ImageFile(np.array([[1, 10000, 3], [4, 5, 6]], dtype=np.int16))
    package = image.export_storage()
    package.dataset = package.dataset.assign_coords(x=[0., 2., 4.], y=[0., 3.])
    image = ImageFile.from_storage(package)
    image.mask[0, 1] = True
    before = image.export_storage()
    region = (before.dataset.x < 3).broadcast_like(before.dataset.intensity)
    assert image.mean(where=region) == pytest.approx(10 / 3)
    xr.testing.assert_identical(image.export_storage().dataset, before.dataset)
    image.mask[0, 1] = False
    assert image[0, 1] == 10000
    assert image.dtype == np.dtype("int16")
    assert np.isnan(image.mean(where=np.zeros(image.shape, dtype=bool)))
    assert np.isnan(image.mean(axis=0, where=np.zeros(image.shape, dtype=bool))).all()


def test_stack_selection_excludes_padding_and_nonfinite_values():
    stack = ImageStack([np.array([[1., np.nan], [3., 4.]]), np.array([[5.]])])
    before = stack.export_storage()
    region = xr.ones_like(before.dataset.intensity, dtype=bool)
    region.values[:, 1, 1] = False
    average = stack.average(where=region).to_numpy()
    assert average[0, 0] == 3
    assert average[1, 0] == 3
    assert average.mask[0, 1] and average.mask[1, 1]
    assert np.isnan(average.data[0, 1]) and np.isnan(average.data[1, 1])
    xr.testing.assert_identical(stack.export_storage().dataset, before.dataset)
    with pytest.raises(ValueError, match="shape"):
        stack.average(where=np.ones((2, 2), dtype=bool))


@pytest.mark.parametrize("change", ["order", "coordinate", "dtype", "missing"])
def test_labelled_image_selector_rejects_misalignment(change):
    image = ImageFile(np.arange(6.).reshape(2, 3))
    dataset = image.export_storage().dataset
    where = xr.ones_like(dataset.intensity, dtype=bool)
    if change == "order":
        where = where.transpose()
    elif change == "coordinate":
        where = where.assign_coords(x=[10, 11, 12])
    elif change == "dtype":
        where = where.astype(int)
    else:
        where = where.drop_vars("x")
    with pytest.raises(ValueError, match="where"):
        image.mean(where=where)
    xr.testing.assert_identical(image.export_storage().dataset, dataset)
