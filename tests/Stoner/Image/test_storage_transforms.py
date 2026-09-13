"""Protect exact calibrated transforms and image ownership boundaries."""

import numpy as np
import pytest
import xarray as xr

from Stoner.Image import ImageFile, ImageStack
from Stoner.Image.storage_transforms import permute


@pytest.fixture
def image():
    source = ImageFile(np.ma.array(np.arange(6).reshape(2, 3), mask=[[0, 1, 0], [0, 0, 0]], fill_value=-9))
    package = source.export_storage()
    package.dataset = package.dataset.assign_coords(y=[10., 12.], x=[1., 4., 8.],
                                                    temperature=("y", [100., 101.]), time=42.)
    package.dataset.y.attrs["units"] = "um"
    package.dataset.x.attrs["units"] = "mm"
    package.dataset.temperature.attrs["units"] = "K"
    package.dataset.intensity.attrs["units"] = "counts"
    package.metadata["run{String}"] = "001"
    return ImageFile.from_storage(package)


@pytest.mark.parametrize("name, transform", [("flip_h", np.fliplr), ("flip_v", np.flipud),
                                            ("CW", lambda a: np.rot90(a, -1)),
                                            ("CCW", lambda a: np.rot90(a, 1))])
def test_public_permutations_preserve_calibration_and_masks(image, name, transform):
    original = image.export_storage()
    result = getattr(image, name)
    values = result.to_numpy()
    expected = transform(image.to_numpy())
    np.testing.assert_array_equal(values.data, expected.data)
    np.testing.assert_array_equal(values.mask, expected.mask)
    assert values.dtype == expected.dtype
    assert values.fill_value == -9
    assert result.metadata["run"] == "001"
    dataset = result.export_storage().dataset
    if name in ("CW", "CCW"):
        assert dataset.x.attrs["units"] == "um"
        assert dataset.y.attrs["units"] == "mm"
        assert dataset.temperature.dims == ("x",)
    else:
        assert dataset.x.attrs["units"] == "mm"
        assert dataset.y.attrs["units"] == "um"
    inverse = {"CW": "CCW", "CCW": "CW", "flip_h": "flip_h", "flip_v": "flip_v"}[name]
    xr.testing.assert_identical(getattr(result, inverse).export_storage().dataset, original.dataset)
    xr.testing.assert_identical(image.export_storage().dataset, original.dataset)


def test_transpose_arguments_and_in_place_invalidation(image):
    source = image.export_storage()
    transposed = image.T
    assert isinstance(transposed, ImageFile)
    xr.testing.assert_identical(transposed.T.export_storage().dataset, source.dataset)
    result = image.transpose(1, 0)
    assert result.shape == (3, 2)
    xr.testing.assert_identical(result.transpose(1, 0).export_storage().dataset, source.dataset)
    same = image.transpose(0, 1, _=None)
    xr.testing.assert_identical(same.export_storage().dataset, source.dataset)
    crop = image.crop(0, 2, 0, 2, _=None)
    image.transpose(1, 0, _=True)
    assert image.shape == (3, 2)
    with pytest.raises(ReferenceError):
        crop[0, 0]


def test_invalid_permutation_is_atomic(image):
    before = image.export_storage()
    for axes in ((0, 0), (0, 2), (1,)):
        with pytest.raises((ValueError, TypeError)):
            image.transpose(axes)
    with pytest.raises(TypeError):
        image.transpose(1, 0, axes=(1, 0))
    xr.testing.assert_identical(image.export_storage().dataset, before.dataset)
    result = image.rotate(0.3, _=None)
    assert result.export_storage().dataset.physical_x.dims == ("y", "x")


def test_frame_and_region_permutations_detach_or_reject():
    stack = ImageStack(np.arange(18).reshape(2, 3, 3))
    item = stack[0]
    before = stack.export_storage()
    turned = item.CW
    turned[0, 0] = 99
    xr.testing.assert_identical(stack.export_storage().dataset, before.dataset)
    with pytest.raises(ValueError, match="Clone"):
        item.transpose(1, 0, _=True)
    with stack.edit_numpy():
        with pytest.raises(RuntimeError):
            item.transpose(1, 0, _=True)


def test_swapaxes_and_auxiliary_coordinate_values(image):
    package = permute(image.export_storage(), "swapaxes", (0, 1))
    np.testing.assert_array_equal(package.dataset.x, [10, 12])
    np.testing.assert_array_equal(package.dataset.y, [1, 4, 8])
    np.testing.assert_array_equal(image.CW.export_storage().dataset.temperature, [101, 100])
