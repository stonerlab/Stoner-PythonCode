"""Exercise the public ImageFile boundary against its authoritative xarray owner."""

import numpy as np
import pytest
import xarray as xr

from Stoner.Image import ImageFile
from Stoner.Image.storage import ImageStorage


@pytest.fixture
def image():
    values = np.ma.array(np.arange(20, dtype=np.int16).reshape(4, 5), mask=False, fill_value=-7)
    values.mask[1, 2] = True
    package = ImageStorage.from_numpy(values)
    package.metadata["run{String}"] = "001"
    package.metadata["nested"] = [1]
    return ImageFile.from_storage(package)


def test_single_owner_and_detached_public_arrays(image):
    assert "_image_owner" in image.__dict__
    assert "_image" not in image.__dict__
    assert "_metadata" not in image.__dict__
    for snapshot in (image.image, image.data):
        with pytest.raises(ValueError):
            snapshot[0, 0] = 90
        snapshot.metadata["run"] = "changed"
        snapshot.setflags(write=True)
        snapshot.data[0, 0] = 80
    assert image[0, 0] == 0
    assert image.metadata["run"] == "001"
    assert image.metadata.type("run") == "String"
    image[0, 0] = 50
    image.mask[1, 2] = False
    assert image[1, 2] == 7
    assert image.to_numpy().fill_value == -7
    raw = np.asarray(image)
    raw[:] = 100
    assert image[0, 0] == 50
    with pytest.raises(ValueError):
        image.mask.data[0, 0] = True
    image.mask.fill(True)
    assert np.all(image.mask)


def test_constructor_clone_and_whole_assignment_preserve_masks(image):
    duplicate = ImageFile(image)
    duplicate[0, 0] = 12
    duplicate.metadata["nested"].append(2)
    assert image[0, 0] == 0
    assert image.metadata["nested"] == [1]
    assert duplicate[1, 2] is np.ma.masked
    clone = image.clone
    clone.image = image.to_numpy()
    assert clone[1, 2] is np.ma.masked
    assert clone.to_numpy(masked=False)[1, 2] == 7
    assert clone.metadata["run"] == "001"
    clone.image = clone.image
    assert clone.metadata["run"] == "001"
    assert clone.metadata.type("run") == "String"


@pytest.mark.parametrize("context", ["edit_numpy", "edit_xarray"])
def test_public_transactions_rollback_and_lock(image, context):
    before = image.export_storage()
    with pytest.raises(RuntimeError, match="cancel"):
        with getattr(image, context)() as draft:
            if context == "edit_numpy":
                draft[:] = 30
            else:
                draft.intensity.values[:] = 30
            for write in (lambda: image.__setitem__((0, 0), 1),
                          lambda: image.mask.__setitem__((0, 0), True),
                          lambda: image.draw.rectangle(1, 1, 2, 2),
                          lambda: setattr(image, "image", np.ones(image.shape))):
                with pytest.raises(RuntimeError):
                    write()
            raise RuntimeError("cancel")
    xr.testing.assert_identical(image.export_storage().dataset, before.dataset)
    image[0, 0] = 2
    assert image[0, 0] == 2


def test_shared_crop_writes_draws_and_invalidates(image):
    crop = image.crop(1, 4, 1, 4, copy=False, _=None)
    assert image.shape == (4, 5)
    assert crop.shape == (3, 3)
    crop[0, 0] = 30
    assert image[1, 1] == 30
    crop.mask[0, 1] = False
    assert image[1, 2] == 7
    crop.draw.rectangle(0, 0, 1, 1, value=40)
    assert image[1, 1] == 40
    detached = crop.clone
    detached[:] = 5
    assert image[1, 1] == 40
    image.image = image.to_numpy()
    with pytest.raises(ReferenceError):
        crop[0, 0] = 10


def test_calibrated_crop_and_native_roundtrip(image):
    package = image.export_storage()
    package.dataset = package.dataset.assign_coords(y=[10., 12., 14., 16.], x=[2., 3., 4., 5., 6.])
    package.dataset.x.attrs["units"] = "um"
    image = ImageFile.from_storage(package)
    crop = image.crop(1, 4, 1, 3, copy=True, _=None)
    np.testing.assert_array_equal(crop.export_storage().dataset.y, [12, 14])
    np.testing.assert_array_equal(crop.export_storage().dataset.x, [3, 4, 5])
    assert crop.export_storage().dataset.x.attrs == {"units": "um"}
    with pytest.warns(UserWarning):
        native = image.to_xarray()
    restored = ImageFile.from_xarray(native, metadata=image.metadata.copy())
    xr.testing.assert_identical(restored.export_storage().dataset, native)
    rotated = image.rotate(0.2, _=None)
    assert rotated.export_storage().dataset.physical_x.attrs == {"units": "um"}
    xr.testing.assert_identical(image.export_storage().dataset, native)


def test_method_clone_control_and_empty_placeholder(image):
    before = image.to_numpy(masked=False)
    result = image.asfloat(_=None)
    assert result is not image
    assert result.dtype.kind == "f"
    np.testing.assert_array_equal(image.to_numpy(masked=False), before)
    placeholder = ImageFile()
    placeholder.image = np.ones((2, 3))
    assert "_image_owner" in placeholder.__dict__
    with pytest.raises(ValueError):
        ImageFile(np.empty((0, 3)))


def test_dimensions_do_not_export_arrays(image, monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("Unexpected array export")
    monkeypatch.setattr(image.__dict__["_image_owner"], "to_numpy", reject)
    assert image.shape == (4, 5)
    assert image.dtype == np.dtype("int16")
    assert image.ndim == 2
    assert image.size == 20


def test_fixed_shape_method_keeps_shared_crop_live(image):
    image.image = image.to_numpy(dtype=float)
    crop = image.crop(1, 4, 1, 3, _=None)
    image.normalise()
    np.testing.assert_array_equal(crop.to_numpy(), image.to_numpy()[1:3, 1:4])


def test_direct_loader_installs_owner():
    from pathlib import Path
    source = Path(__file__).resolve().parents[3] / "sample-data" / "kermit.png"
    image = ImageFile.load(source)
    assert "_image_owner" in image.__dict__
    assert "_image" not in image.__dict__
    assert Path(image.filename) == source.resolve()


def test_mask_selection_targets_owner(image, monkeypatch):
    from Stoner.Image import attrs

    def select(target):
        assert target is image
        return np.ones(target.shape, dtype=bool)

    monkeypatch.setattr(attrs, "ShapeSelect", lambda: select)
    image.mask.select()
    assert np.all(image.mask)


def test_imshow_retains_actual_axes(image):
    from matplotlib import pyplot as plt

    try:
        figure = image.imshow()
        assert image.metadata["fig"] is figure
        assert image.metadata["ax"] is figure.axes[0]
    finally:
        plt.close("all")
