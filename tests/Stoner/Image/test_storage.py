"""Exercise production image interchange independently of the legacy image owner."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from Stoner.Image import ImageFile
from Stoner.Image.storage import ImageStorage
from Stoner.Image.storage_owner import ImageOwner
from Stoner.core.base import TypeHintedDict


@pytest.fixture
def image():
    values = np.ma.array([[1, 2, 3], [4, 5, 99]], dtype=np.uint16,
                         mask=[[False, False, False], [False, False, True]], fill_value=42)
    metadata = TypeHintedDict()
    metadata["run{String}"] = "001"
    metadata["nested"] = {"values": [1]}
    return ImageStorage.from_numpy(values, metadata=metadata)


@pytest.fixture
def stack(image):
    return ImageStorage.from_images([image, np.ones((3, 4), dtype=np.uint16)], names=["small", "large"])


def test_image_round_trip_and_typed_metadata(image):
    restored = image.copy()
    assert restored.metadata["run"] == "001"
    assert restored.metadata.type("run") == "String"
    restored.metadata["nested"]["values"].append(2)
    assert image.metadata["nested"]["values"] == [1]
    array = restored.to_numpy()
    assert array[1, 2] is np.ma.masked
    array.mask[1, 2] = False
    assert array[1, 2] == 99
    assert array.fill_value == 42
    assert array.dtype == np.uint16
    assert restored.dataset.excluded.values[1, 2]
    array[0, 0] = 100
    assert restored.dataset.intensity.values[0, 0] == 1


def test_stack_extents_names_padding_and_reduction(stack):
    assert stack.kind == "stack"
    assert stack.dataset.intensity.dims == ("frame", "y", "x")
    assert [f.name for f in stack.frames] == ["small", "large"]
    assert stack.frames[0].metadata["run"] == "001"
    np.testing.assert_array_equal(stack.dataset.valid_height, [2, 3])
    np.testing.assert_array_equal(stack.dataset.valid_width, [3, 4])
    assert stack.to_numpy().count() == 17
    assert stack.to_numpy().sum() == 27
    assert stack.to_numpy(masked=False)[0, 1, 2] == 99
    copied = stack.copy()
    copied.frames[0].metadata["nested"]["values"].append(7)
    assert stack.frames[0].metadata["nested"]["values"] == [1]


@pytest.mark.parametrize("shape", [(0, 2), (2, 0), (2,), (1, 2, 3)])
def test_reject_non_image_shapes(shape):
    with pytest.raises(ValueError):
        ImageStorage.from_numpy(np.empty(shape))


@pytest.mark.parametrize("dtype", [complex, object, "U2"])
def test_reject_unsupported_intensities(dtype):
    with pytest.raises((TypeError, ValueError)):
        ImageStorage.from_numpy(np.zeros((2, 2), dtype=dtype))


@pytest.mark.parametrize("damage", ["version", "mask", "order", "padding", "raw_padding", "extent", "id", "fill"])
def test_reject_malformed_stack(stack, damage):
    match damage:
        case "version":
            stack.version = 2
        case "mask":
            stack.dataset["excluded"] = stack.dataset.excluded.astype(int)
        case "order":
            stack.frames.reverse()
        case "padding":
            stack.dataset.excluded.values[0, 2, 0] = False
        case "raw_padding":
            stack.dataset.intensity.values[0, 2, 0] = 3
        case "extent":
            stack.dataset.valid_height.values[0] = 4
        case "id":
            stack.frames[0].id = "not-a-uuid"
        case "fill":
            stack.fill_value = -1
    with pytest.raises((ValueError, TypeError, OverflowError)):
        stack.validate()


def test_empty_stack_and_large_integer_promotion():
    empty = ImageStorage.from_images([])
    assert empty.to_numpy().shape == (0, 0, 0)
    with pytest.raises(ValueError, match="precision"):
        ImageStorage.from_images([np.array([[2**63 + 1]], dtype=np.uint64), np.ones((1, 1), dtype=float)])


def test_native_exports_are_detached_and_explicitly_lossy(stack):
    with pytest.warns(UserWarning, match="typed metadata"):
        ds = stack.to_xarray()
    native = ImageStorage.from_xarray(ds)
    assert [f.id for f in native.frames] == [f.id for f in stack.frames]
    assert not native.metadata
    assert not native.frames[0].metadata
    ds.intensity.values[0, 0, 0] = 100
    assert native.dataset.intensity.values[0, 0, 0] == 1
    assert stack.dataset.intensity.values[0, 0, 0] == 1


def test_calibration_survives_copy_and_packing(image):
    image.dataset = image.dataset.assign_coords(x=[0.1, 0.3, 0.5], y=[2., 1.])
    image.dataset.x.attrs["units"] = "um"
    copied = image.copy()
    xr.testing.assert_identical(copied.dataset, image.dataset)
    copied.dataset.x.attrs["units"] = "nm"
    assert image.dataset.x.attrs["units"] == "um"
    packed = ImageStorage.from_images([image])
    np.testing.assert_array_equal(packed.dataset.physical_x[0], image.dataset.x)
    assert packed.dataset.physical_x.attrs == {"units": "um"}


def test_stack_physical_coordinate_validation(stack):
    stack.dataset = stack.dataset.assign_coords(physical_y=(("frame", "y"), [[1., 2., np.nan], [2., 3., 4.]]))
    stack.dataset.physical_y.attrs["units"] = "um"
    stack.validate()
    stack.dataset.physical_y.values[0, 2] = 0
    with pytest.raises(ValueError, match="NaN padding"):
        stack.validate()


def test_owner_edits_rollback_and_retained_drafts(image):
    owner = ImageOwner(image)
    owner[1, 2] = 7
    assert owner[1, 2] == 7
    assert not owner.mask[1, 2]
    with owner.edit_numpy() as draft:
        draft[0, 0] = 10
        with pytest.raises(RuntimeError):
            owner[0, 0] = 2
        with pytest.raises(RuntimeError):
            owner.metadata["new"] = 1
        owner.metadata["nested"]["values"].append(5)
    draft[0, 0] = 20
    assert owner[0, 0] == 10
    assert owner.metadata["nested"]["values"] == [1, 5]
    with pytest.raises(RuntimeError, match="cancel"):
        with owner.edit_numpy() as draft:
            draft[0, 0] = 30
            raise RuntimeError("cancel")
    assert owner[0, 0] == 10
    assert image.dataset.intensity.values[0, 0] == 1


@pytest.mark.parametrize("context", ["edit_numpy", "edit_xarray"])
def test_padding_cannot_be_unmasked_or_written(stack, context):
    owner = ImageOwner(stack)
    with pytest.raises(ValueError, match="Padding"):
        with getattr(owner, context)() as draft:
            if context == "edit_numpy":
                draft[0, 2, 0] = 1
            else:
                draft.excluded.values[0, 2, 0] = False
    assert owner.mask[0, 2, 0]
    assert owner.to_numpy(masked=False)[0, 2, 0] == 0


@pytest.mark.parametrize("damage", ["coordinate", "dtype", "variable", "extent"])
def test_xarray_transaction_rejects_structural_changes(stack, damage):
    owner = ImageOwner(stack)
    before = owner.export_storage()
    with pytest.raises((ValueError, TypeError)):
        with owner.edit_xarray() as draft:
            match damage:
                case "coordinate":
                    draft.x.attrs["units"] = "um"
                case "dtype":
                    draft["intensity"] = draft.intensity.astype(float)
                case "variable":
                    draft["extra"] = 1
                case "extent":
                    draft.valid_height.values[0] = 1
    xr.testing.assert_identical(owner.export_storage().dataset, before.dataset)


def test_xarray_cell_edit_and_snapshot_independence(image):
    owner = ImageOwner(image)
    with owner.edit_xarray() as draft:
        draft.intensity.values[1, 2] = 55
        draft.excluded.values[1, 2] = False
    draft.intensity.values[1, 2] = 11
    assert owner[1, 2] == 55
    snapshot = owner[:]
    with pytest.raises(ValueError):
        snapshot[0, 0] = 0
    snapshot.setflags(write=True)
    snapshot.data[0, 0] = 9
    assert owner[0, 0] == 1


def test_numpy_shape_change_rolls_back_and_releases_lock(image):
    owner = ImageOwner(image)
    with pytest.raises(ValueError, match="shape or dtype"):
        with owner.edit_numpy() as draft:
            draft.shape = (3, 2)
    assert owner.shape == (2, 3)
    owner.mask[0, 0] = True
    assert owner[0, 0] is np.ma.masked
    owner.mask[0, 0] = False
    assert owner[0, 0] == 1


def test_invalid_replacement_is_atomic(stack):
    owner = ImageOwner(stack)
    before = owner.export_storage()
    candidate = owner.export_storage()
    candidate.dataset.valid_width.values[0] = 0
    with pytest.raises(ValueError, match="extent"):
        owner.replace(candidate)
    xr.testing.assert_identical(owner.export_storage().dataset, before.dataset)


def test_image_preserves_explicit_nonseparable_calibration(image):
    image.dataset = image.dataset.assign_coords(physical_x=(("y", "x"), np.ones((2, 3))))
    image.validate()
    restored = ImageStorage.from_xarray(image.dataset, metadata=image.metadata)
    xr.testing.assert_identical(restored.dataset, image.dataset)
    restored.dataset.physical_x.values[0, 0] = 2
    assert image.dataset.physical_x.values[0, 0] == 1


def test_frame_metadata_supplied_to_native_import(stack):
    ds = stack.dataset.copy(deep=True)
    frame_id = stack.frames[0].id
    restored = ImageStorage.from_xarray(ds, frame_metadata={frame_id: stack.frames[0].metadata})
    assert restored.frames[0].metadata["run"] == "001"
    restored.frames[0].metadata["nested"]["values"].append(10)
    assert stack.frames[0].metadata["nested"]["values"] == [1]
    with pytest.raises(ValueError, match="unknown frame"):
        ImageStorage.from_xarray(ds, frame_metadata={"unknown": TypeHintedDict()})


def test_real_hdf5_image_package():
    source = Path(__file__).resolve().parents[3] / "sample-data" / "Sample_Image_2017-10-15_100.hdf5"
    image = ImageFile(source)
    package = ImageStorage.from_numpy(image.image, metadata=image.metadata.copy())
    restored = package.copy()
    np.testing.assert_array_equal(restored.to_numpy(), image.image)
    assert restored.metadata == image.metadata
