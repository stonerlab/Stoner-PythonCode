"""Exercise calibrated ragged packing, extraction and atomic insertion."""

import numpy as np
import pytest
import xarray as xr

from Stoner.Image.storage import ImageStorage
from Stoner.Image.stack_owner import StackOwner


def calibrated(height, width, offset):
    """Make unequal calibrated frames with masks and typed metadata."""
    image = ImageStorage.from_numpy(np.ma.array(np.arange(height * width).reshape(height, width),
                                                mask=False, fill_value=-9))
    image.dataset.excluded.values[0, 0] = True
    image.dataset = image.dataset.assign_coords(y=offset + np.arange(height) * -0.5,
                                                x=offset + np.arange(width) * 0.25,
                                                time=np.datetime64("2026-09-13") + np.timedelta64(offset, "D"))
    image.dataset.y.attrs["units"] = "um"
    image.dataset.x.attrs["units"] = "um"
    image.dataset.time.attrs["description"] = "acquired"
    image.dataset.intensity.attrs["units"] = "counts"
    image.dataset.excluded.attrs["meaning"] = "excluded"
    image.dataset.attrs["instrument"] = ["camera"]
    image.metadata["run{String}"] = "001"
    return image


def test_ragged_calibration_roundtrip_and_detachment():
    images = [calibrated(2, 4, 1), calibrated(3, 2, 2)]
    owner = StackOwner(ImageStorage.from_images(images, names=["same", "same"]))
    package = owner.export_storage()
    assert np.isnan(package.dataset.physical_y.values[0, 2])
    assert np.isnan(package.dataset.physical_x.values[1, 2:]).all()
    for i, source in enumerate(images):
        extracted = owner.frame(i).export_storage()
        xr.testing.assert_identical(extracted.dataset, source.dataset)
        assert extracted.metadata["run"] == "001"
        assert extracted.fill_value == -9
    package.dataset.attrs["instrument"].append("changed")
    assert images[0].dataset.attrs["instrument"] == ["camera"]
    assert owner.export_storage().dataset.attrs["instrument"] == ["camera"]


def test_calibrated_insert_preserves_handles_crops_and_structural_attributes():
    package = ImageStorage.from_images([calibrated(2, 4, 1)])
    package.dataset.valid_height.attrs["meaning"] = "valid rows"
    package.dataset.frame.attrs["meaning"] = "identity"
    owner = StackOwner(package)
    frame = owner.frame(0)
    crop = frame.region(slice(0, 1), slice(0, 2))
    added = owner.insert(0, calibrated(3, 2, 2))
    owner.reorder([frame.id, added.id])
    crop[0, 1] = 123
    assert owner.frame(0)[0, 1] == 123
    xr.testing.assert_identical(added.export_storage().dataset, calibrated(3, 2, 2).dataset)
    assert owner.export_storage().dataset.valid_height.attrs == {"meaning": "valid rows"}
    assert owner.export_storage().dataset.frame.attrs == {"meaning": "identity"}
    owner.delete(added.id)
    assert owner.shape == (1, 2, 4)
    assert crop[0, 1] == 123


@pytest.mark.parametrize("change", ["units", "dataset", "variable", "missing", "spatial", "reserved"])
def test_incompatible_calibration_rejected_atomically(change):
    image = calibrated(2, 4, 1)
    owner = StackOwner(ImageStorage.from_images([image]))
    before = owner.export_storage()
    if change == "units":
        image.dataset.x.attrs["units"] = "nm"
    elif change == "dataset":
        image.dataset.attrs["instrument"] = ["other"]
    elif change == "variable":
        image.dataset.intensity.attrs["units"] = "volts"
    elif change == "missing":
        image.dataset = image.dataset.drop_vars("time")
    elif change == "spatial":
        image.dataset = image.dataset.assign_coords(time=("x", np.arange(4)))
    else:
        image.dataset = image.dataset.rename(time="physical_y")
    with pytest.raises(ValueError):
        owner.insert(0, image)
    xr.testing.assert_identical(owner.export_storage().dataset, before.dataset)
    assert owner.frame(0).id == before.frames[0].id


def test_physical_integer_precision_is_not_silently_lost():
    image = ImageStorage.from_numpy(np.ones((1, 2)))
    image.dataset = image.dataset.assign_coords(x=np.array([2**63, 2**63 + 1], dtype=np.uint64))
    with pytest.raises(ValueError, match="precision"):
        ImageStorage.from_images([image])


def test_empty_calibrated_stack_reinsertion_preserves_attributes():
    image = calibrated(2, 4, 1)
    image.dataset.attrs["array"] = np.array([1, 2])
    owner = StackOwner(ImageStorage.from_images([image]))
    owner.delete(0)
    added = owner.insert(0, image)
    xr.testing.assert_identical(added.export_storage().dataset, image.dataset)
    owner.delete(0)
    before = owner.export_storage()
    image.dataset.intensity.attrs["units"] = "volts"
    with pytest.raises(ValueError, match="Incompatible intensity"):
        owner.insert(0, image)
    xr.testing.assert_identical(owner.export_storage().dataset, before.dataset)
