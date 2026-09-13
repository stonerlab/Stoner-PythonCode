"""Protect shared rectangular ownership before switching public image adapters."""

import numpy as np
import pytest
import xarray as xr

from Stoner.Image.storage import ImageStorage
from Stoner.Image.storage_owner import ImageOwner


@pytest.fixture
def owner():
    values = np.ma.array(np.arange(20, dtype=np.int16).reshape(4, 5), mask=False, fill_value=-7)
    values.mask[1, 2] = True
    package = ImageStorage.from_numpy(values)
    package.metadata["run{String}"] = "001"
    package.metadata["nested"] = {"values": [1]}
    package.dataset = package.dataset.assign_coords(y=[10., 12., 14., 16.], x=[2., 3., 4., 5., 6.])
    package.dataset.y.attrs["units"] = "um"
    return ImageOwner(package)


def test_region_pixels_masks_and_nested_bounds(owner):
    region = owner.region(slice(1, 4), slice(1, 4))
    nested = region.region(slice(-2, None), slice(1, 20))
    assert nested.shape == (2, 2)
    nested[:] = 80
    np.testing.assert_array_equal(owner.to_numpy(masked=False)[2:4, 2:4], 80)
    assert owner[0, 0] == 0
    assert owner[3, 4] == 19
    assert region[0, 1] is np.ma.masked
    region.mask[0, 1] = False
    assert owner[1, 2] == 7
    owner[2, 2] = 99
    assert nested[0, 0] == 99
    nested.mask[:, 0] = True
    np.testing.assert_array_equal(np.asarray(owner.mask)[2:4, 2], True)


def test_region_exports_and_clone_detach_with_calibration(owner):
    region = owner.region(slice(1, 3), slice(1, 4))
    package = region.export_storage()
    np.testing.assert_array_equal(package.dataset.y, [12, 14])
    np.testing.assert_array_equal(package.dataset.x, [3, 4, 5])
    assert package.dataset.y.attrs == {"units": "um"}
    assert package.fill_value == -7
    assert package.metadata.type("run") == "String"
    clone = region.clone()
    clone[:] = -1
    clone.metadata["nested"]["values"].append(2)
    package.dataset.intensity.values[:] = -2
    exported = region.to_numpy()
    exported[:] = -3
    snapshot = region[:]
    with pytest.raises(ValueError):
        snapshot[0, 0] = -4
    assert owner[1, 1] == 6
    assert owner.metadata["nested"]["values"] == [1]
    xr.testing.assert_identical(clone.export_storage().dataset.coords.to_dataset(),
                                region.export_storage().dataset.coords.to_dataset())


@pytest.mark.parametrize("context", ["edit_numpy", "edit_xarray"])
def test_region_transaction_commits_only_rectangle_and_detaches_draft(owner, context):
    region = owner.region(slice(1, 3), slice(1, 4))
    before = owner.to_numpy(masked=False)
    with getattr(region, context)() as draft:
        if context == "edit_numpy":
            draft[:] = 40
        else:
            draft.intensity.values[:] = 40
            draft.excluded.values[:] = False
        for target in (owner, region, owner.region(slice(None), slice(None))):
            with pytest.raises(RuntimeError):
                target[0, 0] = 0
            with pytest.raises(RuntimeError):
                target.metadata["blocked"] = 1
        region.metadata["nested"]["values"].append(3)
    if context == "edit_numpy":
        draft[:] = 50
    else:
        draft.intensity.values[:] = 50
    before[1:3, 1:4] = 40
    np.testing.assert_array_equal(owner.to_numpy(masked=False), before)
    assert not np.any(region.mask)
    assert owner.metadata["nested"]["values"] == [1, 3]


@pytest.mark.parametrize("context", ["edit_numpy", "edit_xarray"])
def test_region_rollback_and_parent_lock(owner, context):
    region = owner.region(slice(1, 3), slice(1, 4))
    before = owner.export_storage()
    with pytest.raises(RuntimeError, match="cancel"):
        with getattr(region, context)() as draft:
            if context == "edit_numpy":
                draft[:] = 100
            else:
                draft.intensity.values[:] = 100
            raise RuntimeError("cancel")
    xr.testing.assert_identical(owner.export_storage().dataset, before.dataset)
    with owner.edit_numpy():
        with pytest.raises(RuntimeError):
            with getattr(region, context)():
                pass
    region[0, 0] = 2
    assert owner[1, 1] == 2


def test_region_rejects_structural_changes(owner):
    region = owner.region(slice(1, 3), slice(1, 4))
    with pytest.raises(ValueError, match="parent image"):
        region.replace(region.export_storage())
    with pytest.raises(ValueError, match="coordinates"):
        with region.edit_xarray() as draft:
            draft.x.attrs["units"] = "nm"
    with pytest.raises(ValueError, match="shape or dtype"):
        with region.edit_numpy() as draft:
            draft.dtype = np.uint16
    region[0, 0] = 3
    assert owner[1, 1] == 3


def test_parent_replacement_invalidates_handles_and_saved_proxies(owner):
    region = owner.region(slice(1, 3), slice(1, 4))
    nested = region.region(slice(None), slice(None))
    metadata = region.metadata
    mask = region.mask
    invalid = owner.export_storage()
    invalid.dataset["excluded"] = invalid.dataset.excluded.astype(int)
    with pytest.raises(TypeError):
        owner.replace(invalid)
    assert region[0, 0] == 6
    owner.replace(owner.export_storage())
    for handle in (region, nested):
        with pytest.raises(ReferenceError):
            handle.to_numpy()
        with pytest.raises(ReferenceError):
            handle[0, 0] = 5
    with pytest.raises(ReferenceError):
        metadata["run"]
    with pytest.raises(ReferenceError):
        mask[0, 0] = True


def test_region_metadata_updates_reach_parent(owner):
    region = owner.region(slice(None), slice(None))
    region.metadata["test{String}"] = "002"
    region.metadata.import_key("extra{String}=003")
    assert owner.metadata["test"] == "002"
    assert owner.metadata["extra"] == "003"
    assert owner.metadata.type("extra") == "String"


@pytest.mark.parametrize("rows, columns", [
    (1, slice(None)), (slice(None), [0, 1]),
    (slice(2, 2), slice(None)), (slice(None, None, -1), slice(None)),
    (slice(None), slice(None, None, 2)), (slice(10, 20), slice(None)),
])
def test_invalid_region_bounds(owner, rows, columns):
    with pytest.raises((TypeError, ValueError)):
        owner.region(rows, columns)


def test_stack_requires_frame_extraction_before_region():
    owner = ImageOwner(ImageStorage.from_images([np.ones((3, 4))]))
    with pytest.raises(TypeError, match="two-dimensional"):
        owner.region(slice(None), slice(None))
