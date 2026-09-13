"""Protect frame identity, atomic structural edits and public frame writes."""

import numpy as np
import pytest
import xarray as xr

from Stoner.Image.storage import ImageStorage
from Stoner.Image.stack_owner import StackOwner


@pytest.fixture
def owner():
    small = np.ma.array([[1, 2, 99], [4, 5, 6]], dtype=np.int16,
                        mask=[[False, False, True], [False, False, False]], fill_value=-7)
    image = ImageStorage.from_numpy(small)
    image.metadata["run{String}"] = "001"
    image.metadata["nested"] = [1]
    package = ImageStorage.from_images([image, np.ones((3, 4), dtype=np.int16)], names=["same", "same"])
    package.metadata["global"] = "stack"
    return StackOwner(package)


def test_frame_writeback_raw_masked_values_and_extents(owner):
    frame = owner.frame(0)
    assert frame.shape == (2, 3)
    assert frame[0, 2] is np.ma.masked
    frame.mask[0, 2] = False
    assert frame[0, 2] == 99
    frame[1, 2] = 12
    assert owner.to_numpy()[0, 1, 2] == 12
    assert frame.to_numpy().fill_value == -7
    frame.metadata["local"] = 3
    assert "local" not in owner.metadata
    assert owner.frame(0).metadata["local"] == 3
    assert owner.mask[0, 2, 0]
    assert owner.to_numpy(masked=False)[0, 2, 0] == 0


def test_saved_handles_survive_insert_reorder_and_retire_on_delete(owner):
    first, second = owner.frame(0), owner.frame(1)
    mask, metadata = first.mask, first.metadata
    added = owner.insert(0, ImageStorage.from_numpy(np.full((4, 2), 20, dtype=np.int16)), name="same")
    assert len({first.id, second.id, added.id}) == 3
    assert owner.frame(1).id == first.id
    first[0, 0] = 10
    owner.reorder([second.id, added.id, first.id])
    assert owner.frame(2)[0, 0] == 10
    mask[0, 2] = False
    metadata["shared"] = 4
    assert owner.frame(2)[0, 2] == 99
    assert owner.frame(2).metadata["shared"] == 4
    saved = owner.export_storage()
    owner.delete(first.id)
    for read in (lambda: first.shape, lambda: first.metadata, lambda: mask[0, 0], lambda: metadata["run"]):
        with pytest.raises(ReferenceError):
            read()
    with pytest.raises(ValueError, match="cannot be reused"):
        owner.replace(saved)
    assert second[0, 0] == 1


@pytest.mark.parametrize("context", ["edit_numpy", "edit_xarray"])
def test_frame_transactions_share_parent_and_sibling_lock(owner, context):
    first, second = owner.frame(0), owner.frame(1)
    before = owner.export_storage()
    with pytest.raises(RuntimeError, match="cancel"):
        with getattr(first, context)() as draft:
            if context == "edit_numpy":
                draft[:] = 30
            else:
                draft.intensity.values[:] = 30
            for change in (lambda: owner.delete(1), lambda: second.__setitem__((0, 0), 2),
                           lambda: first.metadata.import_all(["new{String}=text"]),
                           lambda: owner.metadata.__setitem__("blocked", 1)):
                with pytest.raises(RuntimeError):
                    change()
            first.metadata["nested"].append(2)
            raise RuntimeError("cancel")
    xr.testing.assert_identical(owner.export_storage().dataset, before.dataset)
    assert first.metadata["nested"] == [1, 2]
    with getattr(first, context)() as draft:
        if context == "edit_numpy":
            draft[1, 1] = 50
        else:
            draft.intensity.values[1, 1] = 50
    if context == "edit_numpy":
        draft[:] = 0
    else:
        draft.intensity.values[:] = 0
    assert first[1, 1] == 50
    assert second[1, 1] == 1


def test_public_frame_crop_drawing_metadata_and_clone(owner):
    image = owner.image(0)
    assert "_image" not in image.__dict__
    assert image.filename == "same"
    crop = image.crop(0, 2, 0, 2, _=None)
    crop.draw.rectangle(0, 0, 1, 1, value=42)
    assert owner.frame(0)[0, 0] == 42
    crop.metadata = {"replacement": 4}
    assert image.metadata["replacement"] == 4
    assert owner.metadata["global"] == "stack"
    image.metadata.import_all(["run{String}=002"])
    assert owner.frame(0).metadata["run"] == "002"
    copied = image.clone
    copied[0, 0] = 20
    assert image[0, 0] == 42
    image.image = np.ma.ones(image.shape, dtype=image.dtype)
    assert owner.frame(0)[0, 0] == 1
    with pytest.raises(ValueError, match="shape or dtype"):
        image.image = np.zeros((4, 4), dtype=image.dtype)
    with pytest.raises(ValueError, match="shape or dtype"):
        image.asfloat()
    owner.delete(0)
    with pytest.raises(ReferenceError):
        crop.mask[0, 0] = True


def test_replacement_keeps_frame_id_but_invalidates_old_crop(owner):
    frame = owner.frame(0)
    crop = frame.region(slice(None), slice(0, 2))
    replacement = owner.export_storage()
    replacement.dataset.intensity.values[0, 0, 0] = 18
    owner.replace(replacement)
    assert frame[0, 0] == 18
    with pytest.raises(ReferenceError):
        crop[0, 0]


def test_frame_clone_and_stack_clone_detach(owner):
    frame = owner.frame(0)
    detached = frame.clone()
    stack_copy = owner.clone()
    detached.metadata["nested"].append(2)
    stack_copy.frame(0)[0, 0] = 10
    assert frame.metadata["nested"] == [1]
    assert frame[0, 0] == 1
    snapshot = frame[:]
    with pytest.raises(ValueError):
        snapshot[0, 0] = 0
    assert not np.shares_memory(frame.to_numpy(), owner.to_numpy())


def test_calibrated_frame_extraction_reorders_coordinates(owner):
    package = owner.export_storage()
    package.dataset = package.dataset.assign_coords(
        physical_y=(("frame", "y"), [[10., 12., np.nan], [20., 23., 26.]]),
        time=("frame", [1., 2.]))
    package.dataset.physical_y.attrs["units"] = "um"
    owner = StackOwner(package)
    first = owner.frame(0)
    owner.reorder([owner.frame(1).id, first.id])
    exported = first.export_storage()
    exported.validate()
    np.testing.assert_array_equal(exported.dataset.y, [10, 12])
    assert exported.dataset.y.attrs == {"units": "um"}
    assert float(exported.dataset.time) == 1.
    with first.edit_xarray() as draft:
        draft.intensity.values[0, 0] = 5
    assert first[0, 0] == 5
    assert owner.export_storage().dataset.physical_y.attrs == {"units": "um"}
    with pytest.raises(ValueError, match="same scalar"):
        owner.insert(0, ImageStorage.from_numpy(np.zeros((2, 2))))


def test_invalid_structural_operations_are_atomic(owner):
    before = owner.export_storage()
    identity = owner.frame(0).id
    for order in ([identity, identity], [identity], [identity, "unknown"]):
        with pytest.raises(ValueError):
            owner.reorder(order)
    broken = ImageStorage.from_numpy(np.ones((2, 2), dtype=np.int16))
    broken.dataset["excluded"] = broken.dataset.excluded.astype(int)
    with pytest.raises(TypeError):
        owner.insert(1, broken)
    with pytest.raises(IndexError):
        owner.delete(20)
    xr.testing.assert_identical(owner.export_storage().dataset, before.dataset)
    assert owner.frame(0).id == identity


def test_insertion_rejects_native_attributes_without_discarding_them(owner):
    package = ImageStorage.from_numpy(np.ones((2, 2), dtype=np.int16))
    package.dataset.intensity.attrs["units"] = "counts"
    before = owner.export_storage()
    with pytest.raises(ValueError, match="Incompatible intensity"):
        owner.insert(0, package)
    xr.testing.assert_identical(owner.export_storage().dataset, before.dataset)
    assert package.dataset.intensity.attrs == {"units": "counts"}


def test_empty_delete_reinsert_and_integer_promotion():
    owner = StackOwner(ImageStorage.from_images([]))
    frame = owner.insert(0, ImageStorage.from_numpy(np.ones((2, 3), dtype=np.int16)))
    owner.delete(frame.id)
    assert owner.shape == (0, 0, 0)
    new = owner.insert(0, ImageStorage.from_numpy(np.ones((3, 2), dtype=np.int16)))
    with pytest.raises(ReferenceError):
        frame[0, 0]
    before = owner.export_storage()
    huge = ImageStorage.from_numpy(np.array([[2**63 + 1]], dtype=np.uint64))
    with pytest.raises(ValueError, match="precision"):
        owner.insert(1, huge)
    xr.testing.assert_identical(owner.export_storage().dataset, before.dataset)
    assert new[0, 0] == 1
