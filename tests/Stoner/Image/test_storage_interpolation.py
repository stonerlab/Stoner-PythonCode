"""Protect calibrated interpolation, map ownership and explicit operand alignment."""

import numpy as np
import pytest
import xarray as xr

from Stoner.Image import ImageFile, ImageStack


@pytest.fixture
def image():
    source = ImageFile(np.arange(30.).reshape(5, 6))
    package = source.export_storage()
    package.dataset = package.dataset.assign_coords(y=np.arange(5.) * 2, x=np.arange(6.) * 3, time=42.)
    package.dataset.y.attrs["units"] = "um"
    package.dataset.x.attrs["units"] = "um"
    package.metadata["run{String}"] = "001"
    return ImageFile.from_storage(package)


@pytest.mark.parametrize("name,args", [("rotate", (0.2,)), ("resize", ((7, 8),)),
                                      ("rescale", (1.2,)), ("translate", ((1, 0),)),
                                      ("shift", ((1, 0),)), ("zoom", (1.2,))])
def test_geometry_maps_and_stack_round_trip(image, name, args):
    before = image.export_storage()
    result = getattr(image, name)(*args, _=None)
    package = result.export_storage()
    assert package.dataset.physical_x.dims == ("y", "x")
    assert package.dataset.physical_x.attrs["units"] == "um"
    assert package.metadata["run"] == "001"
    assert package.dataset.time == 42
    stack = ImageStack([result, result])
    xr.testing.assert_identical(stack[0].export_storage().dataset, package.dataset)
    xr.testing.assert_identical(image.export_storage().dataset, before.dataset)


def test_shift_moves_values_maps_and_exclusions_together(image):
    image.mask[2, 2] = True
    result = image.shift((1, 0), order=0, _=None)
    dataset = result.export_storage().dataset
    assert dataset.excluded.values[0].all()
    assert dataset.excluded.values[3, 2]
    assert dataset.physical_y.values[3, 2] == 4
    assert dataset.physical_x.values[3, 2] == 6
    assert result[2, 2] == image[1, 2]
    assert np.isnan(dataset.physical_y.values[0]).all()


def test_operand_mismatch_rejects_before_mutation(image):
    other = image.clone
    package = other.export_storage()
    package.dataset.x.attrs["units"] = "mm"
    other = ImageFile.from_storage(package)
    before = image.export_storage()
    for operation in ("__add__", "__iadd__", "__sub__", "__isub__", "__truediv__", "__floordiv__"):
        with pytest.raises(ValueError, match="identical coordinates"):
            getattr(image, operation)(other)
    xr.testing.assert_identical(image.export_storage().dataset, before.dataset)
    assert (image + image.clone)[1, 1] == 14
    assert (image + np.ones(image.shape))[1, 1] == 8


def test_interpolation_invalidates_crops_and_rejects_shared_geometry(image):
    crop = image.crop(0, 3, 0, 3, _=None)
    image.rotate(0.2)
    with pytest.raises(ReferenceError):
        crop[0, 0]
    stack = ImageStack([image])
    with pytest.raises(ValueError, match="Clone"):
        stack[0].rotate(0.2)


def test_affine_and_warp_share_the_recorded_pixel_mapping(image):
    from skimage.transform import AffineTransform

    shifted = image.affine_transform(np.eye(2), offset=(1, 0), order=0, _=None)
    assert shifted[1, 2] == image[2, 2]
    assert shifted.export_storage().dataset.physical_y.values[1, 2] == 4
    warped = image.warp(AffineTransform(translation=(1, 0)), order=0, _=None)
    assert warped[1, 2] == image[1, 3]
    assert warped.export_storage().dataset.physical_x.values[1, 2] == 9


def test_nonseparable_maps_crop_permute_and_ragged_pack(image):
    result = image.rotate(0.2, _=None)
    cropped = result[1:4, 1:5]
    stack = ImageStack([cropped, result])
    xr.testing.assert_identical(stack[0].export_storage().dataset, cropped.export_storage().dataset)
    assert np.isnan(stack.export_storage().dataset.physical_x.values[0, 3:]).all()
    xr.testing.assert_identical(result.T.T.export_storage().dataset, result.export_storage().dataset)
    from Stoner.Image.storage_transforms import permute
    rotated = ImageFile.from_storage(permute(result.export_storage(), "rot90", (2,)))
    np.testing.assert_array_equal(rotated.to_numpy().data, np.rot90(result.to_numpy().data, 2))


def test_gridimage_uses_explicit_sample_positions(image):
    y, x = np.mgrid[:5, :6]
    points = np.column_stack((x.ravel(), y.ravel()))
    result = image.gridimage(points, (x.astype(float), y.astype(float)), _=None)
    np.testing.assert_array_equal(result.to_numpy(), image.to_numpy())
    np.testing.assert_allclose(result.export_storage().dataset.physical_x, x * 3)


def test_registration_uses_reference_grid_and_measured_pixel_shift(image, monkeypatch):
    from Stoner.Image import imagefuncs

    monkeypatch.setattr(imagefuncs, "_align_scharr", lambda *args, **kwargs: (np.array([1., 0.]), {}))
    result = image.align(image.clone, _=None)
    assert result.metadata["tvec"] == (1., 0.)
    xr.testing.assert_identical(result.export_storage().dataset.coords.to_dataset(),
                                image.export_storage().dataset.coords.to_dataset())
    assert result[2, 2] == pytest.approx(image[1, 2])


def test_external_output_rejected_without_writing(image):
    output = np.full(image.shape, 99.)
    with pytest.raises(ValueError, match="output buffer"):
        image.shift((1, 0), output=output)
    assert (output == 99).all()


def test_folder_alignment_reports_original_failure(image, monkeypatch):
    from Stoner.Image import ImageFolder, imagefuncs

    def fail(*args, **kwargs):
        raise RuntimeError("registration backend failed")

    monkeypatch.setattr(imagefuncs, "_align_scharr", fail)
    folder = ImageFolder(type=ImageFile)
    folder += image
    with pytest.raises(RuntimeError, match="registration backend failed") as caught:
        folder.align(_serial=True)
    assert isinstance(caught.value.__cause__, RuntimeError)
