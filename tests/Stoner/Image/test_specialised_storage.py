"""Protect specialised owner state, scan metadata and channel-grid exports."""

import numpy as np
import pytest
import xarray as xr

from Stoner.Image import ImageFile
from Stoner.Image.kerr import KerrImageFile, KerrStack, MaskStack
from Stoner.Image.stack_owner import FrameOwner
from Stoner.Image.storage import ImageStorage
from Stoner.formats.attocube import AttocubeScan
from Stoner.formats.maximus import MaximusStack


def calibrated_scan(kind):
    """Construct calibrated ragged frames with typed local/common metadata."""
    images = []
    for index, (height, width) in enumerate(((3, 4), (2, 3))):
        package = ImageStorage.from_numpy(np.ma.array(
            np.arange(height*width, dtype=np.int16).reshape(height, width), mask=False, fill_value=-9))
        package.dataset.excluded.values[0, 1] = True
        package.dataset = package.dataset.assign_coords(
            y=1.5-np.arange(height)*0.5, x=np.arange(width)*0.25,
            temperature=280.+index, position=(("y", "x"), np.arange(height*width).reshape(height, width)))
        for coordinate in ("x", "y", "position"):
            package.dataset[coordinate].attrs["units"] = "um"
        package.dataset.temperature.attrs["units"] = "K"
        package.dataset.intensity.attrs["units"] = "V"
        package.dataset.attrs["instrument"] = "scan"
        package.metadata["channel{String}"] = f"00{index}"
        package.metadata["display"] = ("first", "second")[index]
        images.append(package)
    scan = kind.from_storage(ImageStorage.from_images(images, names=["first", "second"]))
    scan._common_metadata["run{String}"] = "001"
    return scan


@pytest.mark.parametrize("kind", [AttocubeScan, MaximusStack])
def test_scan_frame_lookup_does_not_export_numerical_pixels(kind, monkeypatch):
    """Keep frame selection and metadata lookup independent of a NumPy export."""
    scan = calibrated_scan(kind)
    before = scan.export_storage()

    def unexpected_export(*args, **kwargs):
        pytest.fail("Metadata-only scan access exported frame pixels/masks")

    with monkeypatch.context() as patch:
        patch.setattr(FrameOwner, "to_numpy", unexpected_export)
        for selector in (0, "first", scan.frame_ids[0]):
            image = scan[selector]
            assert image.metadata["run"] == "001" and image.metadata["channel"] == "000"
            assert image.metadata.type("run") == image.metadata.type("channel") == "String"
        np.testing.assert_array_equal(scan.metadata.slice("run", values_only=True), ["001", "001"])
    xr.testing.assert_identical(scan.export_storage().dataset, before.dataset)


@pytest.mark.parametrize("kind", [AttocubeScan, MaximusStack])
def test_scan_handles_preserve_identity_calibration_and_live_state(kind):
    """Retain saved handles, display state, common defaults and local writes."""
    scan = calibrated_scan(kind)
    identity = scan.frame_ids[0]
    scan._public_attrs_store[identity] = {"title": "Signal", "_mask_color": "blue", "_mask_alpha": 0.25}
    item = scan[0]
    metadata = item.metadata
    before = item.export_storage()
    assert item.title == "Signal" and item._mask_color == "blue" and item._mask_alpha == 0.25
    assert item.filename == "first" and item.to_numpy().fill_value == -9
    scan.reorder(reversed(scan.frame_ids))
    assert scan.frame_ids[1] == identity
    xr.testing.assert_identical(item.export_storage().dataset, before.dataset)
    scan._common_metadata["run{String}"] = "002"
    assert metadata["run"] == scan[1].metadata["run"] == "002"
    metadata["run{String}"] = "003"
    assert scan._common_metadata["run"] == "002" and scan[1].metadata["run"] == "003"
    item[0, 0] = 99
    item.mask[0, 1] = False
    assert scan[1][0, 0] == 99 and not scan[1].mask[0, 1]
    current = item.export_storage()
    xr.testing.assert_identical(current.dataset.coords.to_dataset(), before.dataset.coords.to_dataset())
    assert current.dataset.intensity.attrs == before.dataset.intensity.attrs
    assert current.dataset.attrs == before.dataset.attrs and current.fill_value == before.fill_value
    del scan[1]
    with pytest.raises(ReferenceError):
        metadata["run"]


@pytest.mark.parametrize("kind", [AttocubeScan, MaximusStack])
def test_scan_owners_and_live_header_overrides(kind, tmp_path):
    scan = kind(scan=7)
    image = ImageFile(np.ma.array(np.ones((3, 4)), mask=False, fill_value=-9))
    image.filename = "channel"
    image.metadata["display"] = "channel"
    scan.append(image)
    scan._common_metadata["run{String}"] = "001"
    item = scan[0]
    assert item.metadata["run"] == "001"
    scan._common_metadata["run{String}"] = "002"
    assert item.metadata["run"] == "002"
    item.metadata["run{String}"] = "003"
    assert scan._common_metadata["run"] == "002"
    item.mask[0, 0] = True
    item[1, 1] = 9
    assert not {"_stack", "_sizes", "_metadata", "_common_metadata"}.intersection(scan.__dict__)
    with scan.edit_numpy():
        assert scan[0].metadata["run"] == "003"
        with pytest.raises(RuntimeError):
            item.metadata["run"] = "blocked"
    filename = tmp_path / "scan.hdf5"
    scan.to_hdf5(filename)
    item[1, 1] = 8
    scan.to_hdf5(filename)
    restored = kind.read_hdf5(filename)
    assert restored[0].metadata["run"] == "003"
    assert restored[0].mask[0, 0]
    assert restored[0][1, 1] == 8
    assert restored[0].to_numpy().fill_value == -9
    del scan[0]
    with pytest.raises(ReferenceError):
        item.metadata["run"]


def test_attocube_named_channels_and_irregular_position_maps():
    scan = AttocubeScan()
    arrays = {"PosX": [[0., 1.], [0.1, 1.1]], "PosY": [[0., 0.2], [1., 1.2]], "Signal": [[3., 4.], [5., 6.]]}
    for name, values in arrays.items():
        image = ImageFile(np.array(values))
        image.filename = name
        image.metadata["display"] = name
        image.metadata["z-unit"] = "m" if name.startswith("Pos") else "V"
        scan.append(image)
    scan["Signal"].mask[1, 1] = True
    with pytest.warns(UserWarning):
        dataset = scan.to_xarray(format="channels")
    assert dataset.Signal.dims == ("y", "x")
    assert dataset.Signal.attrs["units"] == "V"
    assert dataset.physical_x.dims == ("y", "x")
    np.testing.assert_array_equal(dataset.physical_x, arrays["PosX"])
    assert dataset.Signal__excluded.values[1, 1]
    dataset.Signal.values[:] = 0
    assert scan["Signal"][0, 0] == 3


def test_kerr_descriptor_and_frame_handles_use_owner():
    image = KerrImageFile(np.arange(12.).reshape(3, 4))
    assert "_image" not in image.__dict__
    assert type(image.imarray) is np.ma.MaskedArray
    with pytest.raises(ValueError):
        image.imarray[0, 0] = 1
    stack = KerrStack([image, image])
    item = stack[0]
    assert isinstance(item, KerrImageFile)
    saved_id = stack.frame_ids[0]
    item.metadata["field"] = 2
    stack[1].metadata["field"] = 1
    np.testing.assert_array_equal(stack.fields, [2, 1])
    stack.sort("field")
    np.testing.assert_array_equal(stack.fields, [1, 2])
    assert stack.frame_ids[1] == saved_id
    item[0, 0] = 20
    assert stack[1][0, 0] == 20


def test_mask_switch_analysis_is_pixelwise_and_does_not_change_source():
    values = np.array([[[False, False]], [[False, True]], [[True, True]]])
    stack = MaskStack(values)
    before = stack.export_storage()
    index, progression = stack.switch_index()
    np.testing.assert_array_equal(index, [[1, 0]])
    assert progression.shape == (2, 1, 2)
    stack.switch_index(saturation_value=False, saturation_end=False)
    xr.testing.assert_identical(stack.export_storage().dataset, before.dataset)


def test_mask_constructor_retains_nonzero_boolean_conversion_and_padding():
    stack = MaskStack([np.array([[0., 0.1, -0.1]]), np.ones((2, 3)), np.ones((2, 3))])
    np.testing.assert_array_equal(stack[0].to_numpy(), [[False, True, True]])
    before = stack.export_storage()
    _, progression = stack.switch_index()
    assert progression.imarray.mask[0, 1].all()
    assert not progression.to_numpy(masked=False)[0, 1].any()
    xr.testing.assert_identical(stack.export_storage().dataset, before.dataset)


def test_denoise_inversion_preserves_ragged_padding():
    stack = KerrStack([np.arange(20.).reshape(4, 5), np.arange(25.).reshape(5, 5)])
    normal = stack.denoise_thresh()
    inverted = stack.denoise_thresh(invert=True)
    for index in range(len(stack)):
        np.testing.assert_array_equal(inverted[index].to_numpy(), ~normal[index].to_numpy())
    assert inverted.imarray.mask[0, 4].all()
    assert not inverted.to_numpy(masked=False)[0, 4].any()
