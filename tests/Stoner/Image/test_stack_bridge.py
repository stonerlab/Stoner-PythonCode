"""Protect public stack ownership, frame lifetimes and masked reductions."""

import numpy as np
import pytest
import xarray as xr

from Stoner.Image import ImageFile, ImageStack
from Stoner.Image.storage import ImageStorage


def test_specialised_stacks_remain_in_the_public_stack_family():
    from Stoner.Image.kerr import KerrStack, MaskStack
    from Stoner.formats.attocube import AttocubeScan
    from Stoner.formats.maximus import MaximusStack

    for specialised in (KerrStack, MaskStack, AttocubeScan, MaximusStack):
        assert issubclass(specialised, ImageStack)


def test_public_stack_owns_one_package_and_saved_items_follow_ids():
    stack = ImageStack([np.ones((2, 3)), np.full((3, 2), 2.)])
    assert not {"_stack", "_sizes", "_metadata", "_names"}.intersection(stack.__dict__)
    first = stack[0]
    first["rank"] = 3
    stack[1]["rank"] = 1
    first_id = stack.frame_ids[0]
    crop = first.crop(0, 2, 0, 1, _=None)
    mask, metadata = first.mask, first.metadata
    stack.insert(0, np.full((4, 4), 8.))
    stack[0]["rank"] = 2
    stack.sort("rank")
    assert stack.frame_ids[-1] == first_id
    crop[0, 1] = 10
    mask[1, 1] = True
    metadata["shared{String}"] = "yes"
    assert stack[-1][0, 1] == 10
    assert stack[-1].mask[1, 1]
    assert stack[-1]["shared"] == "yes"
    del stack[-1]
    with pytest.raises(ReferenceError):
        first[0, 0]
    with pytest.raises(ReferenceError):
        crop[0, 0]


def test_transactions_lock_items_and_rollback():
    stack = ImageStack(np.ones((2, 3, 4)))
    item = stack[0]
    before = stack.export_storage()
    with pytest.raises(RuntimeError, match="cancel"):
        with stack.edit_numpy() as draft:
            draft[:] = 5
            cloned = stack.clone
            cloned[0][0, 0] = 7
            assert item[0, 0] == 1
            for edit in (lambda: item.__setitem__((0, 0), 2), lambda: stack.insert(0, item),
                         lambda: stack.sort(), lambda: stack.__delitem__(0)):
                with pytest.raises(RuntimeError):
                    edit()
            raise RuntimeError("cancel")
    xr.testing.assert_identical(stack.export_storage().dataset, before.dataset)
    with stack.edit_xarray() as draft:
        draft.intensity.values[0, 0, 0] = 9
    draft.intensity.values[:] = 0
    assert item[0, 0] == 9


def test_reductions_ignore_raw_masked_values_and_padding():
    first = np.ma.array([[2., 999.]], mask=[[False, True]])
    second = np.array([[4., 6.], [8., 10.]])
    stack = ImageStack([first, second])
    np.testing.assert_array_equal(stack.mean().to_numpy(), [[3, 6], [8, 10]])
    np.testing.assert_array_equal(stack.stddev().to_numpy(), [[1, 0], [0, 0]])
    assert stack.stderr()[0, 0] == pytest.approx(1 / np.sqrt(2))
    assert stack.imarray.mask[0, 1].all()
    assert (stack.to_numpy(masked=False)[0, 1] == 0).all()


def test_clone_slice_bulk_crop_conversion_and_last_deletion():
    stack = ImageStack(np.ones((2, 3, 4)))
    stack[0]["nested"] = [1]
    copied = stack.clone
    copied[0]["nested"].append(2)
    assert stack[0]["nested"] == [1]
    copied[0][0, 0] = 9
    assert stack[0][0, 0] == 1
    assert stack[::-1].frame_ids == stack.frame_ids[::-1]
    original_ids = stack.frame_ids
    stack.each.crop(0, 2, 0, 2)
    assert stack.shape == (2, 2, 2)
    assert stack.frame_ids == original_ids
    stack.convert(np.float32)
    assert stack.imarray.dtype == np.float32
    del stack[-len(stack)]
    del stack[0]
    assert stack.shape == (0, 0, 0)
    stack.append(ImageFile(np.ones((2, 2))))
    assert stack.shape == (1, 2, 2)


def test_public_calibrated_interchange_and_atomic_incompatible_insert():
    image = ImageStorage.from_numpy(np.ones((2, 3)))
    image.dataset = image.dataset.assign_coords(x=[1., 2., 4.])
    image.dataset.x.attrs["units"] = "um"
    package = ImageStorage.from_images([image])
    stack = ImageStack.from_storage(package)
    xr.testing.assert_identical(stack[0].export_storage().dataset, image.dataset)
    before = stack.export_storage()
    with pytest.raises(ValueError, match="Incompatible x"):
        stack.append(np.zeros((2, 3)))
    xr.testing.assert_identical(stack.export_storage().dataset, before.dataset)
    with pytest.raises(ValueError):
        stack.imarray[0, 0, 0] = 8
    stack[0, 0, 0] = 8
    assert stack[0][0, 0] == 8


def test_fixed_shape_bulk_edit_keeps_saved_crop_and_replacement_invalidates_it():
    stack = ImageStack(np.arange(24., dtype=float).reshape(2, 3, 4))
    image = stack[0]
    crop = image.crop(0, 2, 0, 2, _=None)
    stack.each.normalise()
    crop[0, 0] = 0.25
    assert image[0, 0] == 0.25
    stack[0] = ImageFile(np.ones((2, 2)))
    assert image.shape == (2, 2)
    with pytest.raises(ReferenceError):
        crop[0, 0]


def test_calibrated_reduction_preserves_coordinates_or_rejects_conflicts():
    image = ImageStorage.from_numpy(np.ones((2, 3)))
    image.dataset = image.dataset.assign_coords(x=[1., 2., 4.])
    image.dataset.x.attrs["units"] = "um"
    stack = ImageStack.from_storage(ImageStorage.from_images([image, image]))
    result = stack.mean().export_storage()
    xr.testing.assert_identical(result.dataset.x, image.dataset.x)
    image.dataset = image.dataset.assign_coords(x=[1., 2., 5.])
    image.dataset.x.attrs["units"] = "um"
    stack.append(ImageFile.from_storage(image))
    with pytest.raises(ValueError, match="reconcile frame calibration"):
        stack.mean()


def test_native_attributes_survive_reduction_and_guard_structural_assignment():
    image = ImageStorage.from_numpy(np.ones((2, 3)))
    image.dataset.intensity.attrs["units"] = "counts"
    stack = ImageStack.from_storage(ImageStorage.from_images([image]))
    assert stack.mean().export_storage().dataset.intensity.attrs == {"units": "counts"}
    with pytest.raises(ValueError, match="calibrated storage package"):
        stack.imarray = np.ones((2, 3, 4))
    assert stack.shape == (1, 2, 3)
