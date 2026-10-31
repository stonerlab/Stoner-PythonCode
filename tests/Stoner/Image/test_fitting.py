"""Verify surface estimates, physical covariance, replay and whole-grid correction."""

import pickle
from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import xarray as xr
from lmfit.model import ModelResult
from matplotlib import pyplot as plt

from Stoner import ROI, Data, ImageFile, ImageStack
from Stoner.analysis.fitting.models import make_model, model_registry
from Stoner.analysis.fitting.models.surfaces import Plane, conditioning, design_matrix
from Stoner.enums import BackgroundReference, CoordinateMode, FitOutput
from Stoner.Image import ImageFitter


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    plt.close("all")
    monkeypatch.setattr(model_registry, "_entries", model_registry._entries.copy())
    yield
    plt.close("all")


def measurement(quadratic=False, noise=0.):
    x, y = np.linspace(-3, 4, 19), np.linspace(-2, 5, 17)
    xx, yy = np.meshgrid(x, y)
    coefficients = np.array([5., .4, -.2, .08, -.06, .03] if quadratic else [5., .4, -.2])
    background = design_matrix(xx, yy, quadratic) @ coefficients
    values = background + np.random.default_rng(123).normal(0, noise, xx.shape)
    values[6:10, 8:12] += 20
    package = ImageFile(values).export_storage()
    package.dataset = package.dataset.assign_coords(x=x, y=y)
    for axis in ("x", "y"):
        package.dataset[axis].attrs["units"] = "um"
    package.dataset.intensity.attrs["units"] = "mV"
    image = ImageFile.from_storage(package)
    included = np.ones(xx.shape, bool)
    included[5:11, 7:13] = False
    image.roi.from_mask(included)
    return image, coefficients, background


def test_lookup_and_contextual_discovery_are_inert(monkeypatch):
    image, _, _ = measurement()
    monkeypatch.setattr("Stoner.Image.fitting.construct_model", lambda *a: pytest.fail("Constructed on lookup"))
    assert isinstance(image.fit, ImageFitter)
    assert set(image.fit.models) == {"plane", "quadratic_surface"}
    assert "plane" in dir(image.fit) and "linear" not in dir(image.fit)
    assert image.fit.plane.__name__ == "plane" and "ImageFitter" in image.fit.plane.__doc__
    assert "fit" not in image.__dict__ and not plt.get_fignums()
    for name in ("linear", "odr", "curve_fit", "unknown"):
        with pytest.raises(AttributeError):
            getattr(image.fit, name)
    with pytest.raises(AttributeError, match="spatial"):
        _ = Data().fit.plane


@pytest.mark.parametrize("quadratic", [False, True])
def test_recovery_feature_preservation_and_independent_ownership(quadratic):
    image, expected, background = measurement(quadratic)
    before = image.export_storage()
    report = image.fit("quadratic_surface" if quadratic else "plane", output="report")
    assert isinstance(report, ModelResult) and report.success
    np.testing.assert_allclose([report.params[p].value for p in report.model.param_names], expected, atol=1e-10)
    assert report.var_names[0].startswith("image_normal_")
    np.testing.assert_allclose(image.fit.evaluate(report).to_numpy(), background, atol=1e-10)
    image.roi.clear()
    corrected = image.fit.flatten(report)
    np.testing.assert_allclose(corrected.to_numpy(), before.dataset.intensity.values-background, atol=1e-10)
    assert corrected.dtype == np.float64 and corrected[7, 9] == pytest.approx(20.)
    corrected[0, 0] = -100
    xr.testing.assert_identical(image.image, before.dataset)


def test_combined_flatten_preserves_source_metadata_and_roi():
    image, _, _ = measurement(True)
    image.metadata["nested"] = {"values": [1]}
    before, roi = image.export_storage(), image.roi.current
    corrected = image.fit.flatten(model="quadratic_surface")
    xr.testing.assert_identical(image.image, before.dataset)
    assert dict(image.metadata) == dict(before.metadata) and image.roi.current == roi
    corrected.metadata["nested"]["values"].append(2)
    assert image.metadata["nested"]["values"] == [1]


@pytest.mark.parametrize("quadratic", [False, True])
@pytest.mark.parametrize("absolute", [False, True])
def test_covariance_against_independent_weighted_design(quadratic, absolute):
    image, _, _ = measurement(quadratic, noise=.02)
    sigma = np.linspace(.01, .05, image.size).reshape(image.shape)
    report = image.fit("quadratic_surface" if quadratic else "plane", sigma=sigma,
                       absolute_sigma=absolute, output="report")
    selected = image.roi.resolve().values
    xx, yy = np.meshgrid(image.image.x, image.image.y)
    weighted = design_matrix(xx[selected], yy[selected], quadratic)/sigma[selected, None]
    expected = np.linalg.inv(weighted.T @ weighted)*(1 if absolute else report.redchi)
    np.testing.assert_allclose(report.image_fit.to_dict()["covariance"], expected, rtol=5e-4, atol=1e-10)
    np.testing.assert_allclose([report.params[p].stderr for p in report.model.param_names],
                               np.sqrt(np.diag(expected)), rtol=5e-4, atol=1e-10)


def test_large_offsets_small_spacing_stable_evaluation_and_covariance():
    x, y = 1e9+np.arange(13)*.001, -2e8+np.arange(11)*.003
    xx, yy = np.meshgrid(x, y)
    u, v = (xx-x.mean())/.006, (yy-y.mean())/.015
    values = 4+.2*u-.3*v+.4*u*u+.1*u*v-.05*v*v
    package = ImageFile(values).export_storage()
    package.dataset = package.dataset.assign_coords(x=x, y=y)
    image = ImageFile.from_storage(package)
    report = image.fit.quadratic_surface(output="report", sigma=.01, absolute_sigma=True)
    np.testing.assert_allclose(image.fit.evaluate(report).to_numpy(), values, atol=1e-8)
    (cx, cy, sx, sy), transform = conditioning(xx.ravel(), yy.ravel(), True)
    matrix = design_matrix((xx.ravel()-cx)/sx, (yy.ravel()-cy)/sy, True)/.01
    expected = transform @ np.linalg.inv(matrix.T @ matrix) @ transform.T
    scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
    np.testing.assert_allclose(np.asarray(report.image_fit.to_dict()["covariance"])/scale,
                               expected/scale, rtol=2e-4, atol=1e-9)


@pytest.mark.parametrize("reference", [BackgroundReference.ZERO, "region_mean", 7.5])
def test_reference_levels_excluded_values_and_nonfinite_pixels(reference):
    image, _, background = measurement()
    image.mask[0, 0] = True
    image[0, 1] = np.nan
    sigma = np.linspace(.1, 1, image.size).reshape(image.shape)
    report = image.fit.plane(sigma=sigma, output="report")
    result = image.fit.flatten(report, reference=reference)
    record = report.image_fit.to_dict()
    level = 0. if reference == "zero" else record["reference_mean"] if reference == "region_mean" else reference
    assert result.to_numpy(masked=False)[0, 0] == pytest.approx(level, abs=1e-10)
    assert np.asarray(result.mask)[0, 0] and np.isnan(result[0, 1])
    mask = ROI.from_dict(record["eligibility"]).resolve(image.image.intensity).values
    assert record["reference_mean"] == pytest.approx(np.average(background[mask], weights=1/sigma[mask]**2))


@pytest.mark.parametrize("output", list(FitOutput))
def test_output_matrix_enum_string_parity(output):
    image, _, _ = measurement(noise=.01)
    first = image.fit.plane(output=output, prefix="background")
    second = image.fit.lmfit(Plane, output=output.value, prefix="background")
    assert "background:image_fit" in image.metadata
    if output is FitOutput.FIT:
        np.testing.assert_allclose(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        assert first[1].shape == (3, 3)
    elif output is FitOutput.REPORT:
        assert isinstance(first, ModelResult)
    elif output is FitOutput.FULL:
        assert isinstance(first[0], ModelResult) and first[1].shape == (7,)
    elif output is FitOutput.ROW:
        np.testing.assert_allclose(first, second)
        assert first.shape == (7,)
    elif output is FitOutput.DATA:
        assert first is image and second is image
    else:
        assert first["slope_x"] == pytest.approx(second["slope_x"])


def test_fixed_coefficients_allow_an_identifiable_line_of_pixels():
    xx, yy = np.meshgrid(np.arange(8.), np.arange(3.))
    image = ImageFile(2+3*xx+4*yy)
    selected = np.zeros(image.shape, bool)
    selected[1] = True
    params = Plane().make_params(offset=1, slope_x=1, slope_y=4)
    params["slope_y"].vary = False
    values, covariance = image.fit(Plane, where=selected, p0=params)
    np.testing.assert_allclose(values, [2, 3, 4], atol=1e-8)
    assert covariance[2, 2] == 0
    with pytest.raises(ValueError, match="rank|identify"):
        image.fit.plane(where=selected)


def test_decorated_model_guesser_receives_both_coordinates():
    calls = []

    @make_model(name="test_image_surface", independent_vars=("x", "y"))
    def surface(x, y, offset, slope_x, slope_y):
        return offset+slope_x*x+slope_y*y

    @surface.guesser
    def guess(data, x=None, y=None):
        calls.append((len(data), x.copy(), y.copy()))
        return [4, .3, -.1]

    image, expected, _ = measurement()
    np.testing.assert_allclose(image.fit.test_image_surface()[0], expected, atol=1e-8)
    assert calls[0][0] == image.roi.resolve().values.sum()
    assert model_registry.registration("test_image_surface").independent_vars == ("x", "y")


@pytest.mark.parametrize("options", [{"where": np.zeros((17, 19), bool)}, {"where": np.ones((17, 1), bool)},
                                      {"sigma": -1}, {"sigma": np.zeros((17, 19))},
                                      {"absolute_sigma": True, "scale_covar": True},
                                      {"output": CoordinateMode.NATIVE}, {"output": "bad"},
                                      {"xcol": 0}, {"nan_policy": "omit"}, {"method": "typo"}])
def test_invalid_inputs_leave_source_unmodified(options):
    image, _, _ = measurement()
    before = image.export_storage()
    with pytest.raises((TypeError, ValueError)):
        image.fit.plane(**options)
    xr.testing.assert_identical(image.image, before.dataset)
    assert dict(image.metadata) == dict(before.metadata)


def test_sigma_alignment_and_irrelevant_invalid_values():
    image, expected, _ = measurement()
    sigma = image.image.intensity.copy(data=np.full(image.shape, .1))
    sigma.values[~image.roi.resolve().values] = np.nan
    np.testing.assert_allclose(image.fit.plane(sigma=sigma)[0], expected, atol=1e-10)
    with pytest.raises(ValueError, match="coordinates"):
        image.fit.plane(sigma=sigma.isel(x=slice(None, None, -1)))


def test_frozen_report_replay_after_copy_pickle_registry_and_source_changes():
    image, _, background = measurement()
    report = image.fit.plane(output="report")
    with pytest.raises(FrozenInstanceError):
        report.image_fit.record = "changed"
    report.params["image_normal_0"].value = 500
    model_registry._entries.pop("plane")
    image.roi.set(~ROI.all())
    image += 10
    for replay in (report, deepcopy(report), pickle.loads(pickle.dumps(report))):
        np.testing.assert_allclose(image.fit.evaluate(replay).to_numpy(), background, atol=1e-9)
        assert image.fit.flatten(replay)[0, 0] == pytest.approx(10, abs=1e-8)
    with pytest.raises(ValueError, match="grid"):
        image[1:5, 2:7].fit.evaluate(report)


def test_box_intersection_offsets_and_native_pixel_modes():
    image, expected, _ = measurement()
    np.testing.assert_allclose(image.fit.plane(box=(0, 7, 0, 8))[0], expected, atol=1e-8)
    child = image[0:5, 1:7]
    child.roi.clear()
    np.testing.assert_allclose(child.fit.plane()[0], expected, atol=1e-8)
    pixel, _ = child.fit.plane(coordinates="pixel")
    assert pixel[1] == pytest.approx(expected[1]*np.diff(image.image.x)[0])
    assert pixel[0] != pytest.approx(expected[0])


@pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.bool_, np.float32])
def test_dtype_raw_units_and_frame_independence(dtype):
    xx, yy = np.meshgrid(np.arange(5), np.arange(4))
    image = ImageFile((10+2*xx+3*yy).astype(dtype))
    before = image.to_numpy()
    result = image.fit.flatten()
    assert result.dtype == np.float64
    np.testing.assert_array_equal(image.to_numpy(), before)
    np.testing.assert_allclose(result.to_numpy(), 0, atol=1e-6)
    stack = ImageStack([image])
    corrected = stack[0].fit.flatten()
    corrected[0, 0] = -100
    np.testing.assert_array_equal(stack[0].to_numpy(), before)


def test_report_rejects_fit_options_and_inplace_is_explicitly_deferred():
    image, _, _ = measurement()
    report = image.fit.plane(output="report")
    with pytest.raises(TypeError, match="fitting-only"):
        image.fit.flatten(report, where=ROI.all())
    with pytest.raises(NotImplementedError):
        image.fit.flatten(report, inplace=True)


@pytest.mark.parametrize("selection", [np.eye(5, dtype=bool), np.ones((1, 5), dtype=bool)])
def test_rank_deficient_and_one_pixel_axis_fail_without_publication(selection):
    image = ImageFile(np.ones(selection.shape))
    with pytest.raises(ValueError, match="rank|identify"):
        image.fit.plane(where=selection)
    assert not any("image_fit" in key for key in image.metadata)


def test_nonuniform_decreasing_axes_alternate_coordinate_names_and_constraints():
    x, y = np.array([10., 7, 5, 4, 0]), np.array([9., 6, 3, -1])
    xx, yy = np.meshgrid(x, y)
    package = ImageFile(2+3*xx-4*yy).export_storage()
    package.dataset = package.dataset.assign_coords(x=x, y=y, distance=("x", 2*x), height=("y", 3*y))
    image = ImageFile.from_storage(package)
    report = image.fit.plane(x="distance", y="height", output="report")
    np.testing.assert_allclose([report.params[p].value for p in report.model.param_names], [2, 1.5, -4/3])
    np.testing.assert_allclose(image.fit.flatten(report).to_numpy(), 0, atol=1e-10)
    params = Plane(prefix="bg_").make_params(offset=0, slope_x=1, slope_y=-3)
    params["bg_slope_x"].set(min=0, max=5)
    constrained = image.fit(Plane(prefix="bg_"), p0=params, output="report")
    assert constrained.params["bg_slope_x"].value == pytest.approx(3)
    assert constrained.image_fit.to_dict()["parameter_names"][0] == "bg_offset"


def test_transaction_conflicts_do_not_publish_metadata_or_corrections():
    image, _, _ = measurement()
    before = dict(image.metadata)
    with image.edit_numpy(), pytest.raises(RuntimeError):
        image.fit.plane()
    assert dict(image.metadata) == before


@pytest.mark.parametrize("variables", ["x", (), ("x", "x"), ("missing", "y")])
def test_decorator_rejects_invalid_independent_variable_declarations(variables):
    with pytest.raises((TypeError, ValueError)):
        make_model(lambda x, y, p: p*x+y, name="bad_surface", independent_vars=variables)


def test_native_report_eval_overrides_and_reusing_a_conditioned_model_with_constraints():
    image, _, _ = measurement()
    report = image.fit.plane(output="report")
    x, y = report.userkws["x"], report.userkws["y"]
    np.testing.assert_allclose(report.eval(slope_x=2), 5+2*x-.2*y, atol=1e-9)
    assert report.image_fit.to_dict()["parameter_units"] == ["mV", "mV / um", "mV / um"]
    params = report.model.make_params(offset=0, slope_x=.4, slope_y=0)
    params["slope_x"].vary = False
    np.testing.assert_allclose(image.fit(report.model, p0=params)[0], [5, .4, -.2], atol=1e-9)


def test_parameter_expressions_and_failed_fit_publication():
    xx, yy = np.meshgrid(np.arange(6.), np.arange(5.))
    image = ImageFile(2+3*xx+6*yy)
    params = Plane().make_params(offset=0, slope_x=1, slope_y=2)
    params["slope_y"].set(expr="2*slope_x")
    values, covariance = image.fit(Plane, p0=params)
    np.testing.assert_allclose(values, [2, 3, 6], atol=1e-8)
    assert covariance[2, 2] == pytest.approx(4*covariance[1, 1])
    metadata = dict(image.metadata)
    with pytest.raises(RuntimeError):
        image.fit.plane(iter_cb=lambda *args, **kwargs: True)
    assert dict(image.metadata) == metadata
    with pytest.raises(ValueError, match="unrecognised"):
        image.fit.plane(p0={"slop_x": 2})


def test_current_explicit_all_and_box_precedence_capture_actual_membership():
    image, _, _ = measurement()
    image.roi.set(~ROI.all())
    with pytest.raises(ValueError, match="eligible"):
        image.fit.plane()
    report = image.fit.plane(where=ROI.all(), box=(0, 7, 0, 8), output="report")
    assert report.image_fit.to_dict()["sample_count"] == 56
    selected = ROI.from_dict(report.image_fit.to_dict()["selection"]).resolve(image.image.intensity)
    assert int(selected.sum()) == 56
    assert not image.roi.resolve().values.any()
    image.roi.clear()
    report = image.fit.plane(output="report")
    assert report.image_fit.to_dict()["sample_count"] == image.size


@pytest.mark.parametrize("where", [np.ones((17, 19)), np.full((17, 19), None, dtype=object)])
def test_non_boolean_selection_rejected(where):
    image, _, _ = measurement()
    with pytest.raises(ValueError):
        image.fit.plane(where=where)


def test_labelled_selection_and_unsupported_coordinate_map():
    image, _, _ = measurement()
    selected = image.roi.resolve()
    with pytest.raises(ValueError, match="coordinates"):
        image.fit.plane(where=selected.isel(x=slice(None, None, -1)))
    package = image.export_storage()
    package.dataset = package.dataset.assign_coords(position_map=(("y", "x"), np.ones(image.shape)))
    mapped = ImageFile.from_storage(package)
    with pytest.raises(ValueError, match="one-dimensional"):
        mapped.fit.plane(x="position_map")


@pytest.mark.parametrize("reference", [CoordinateMode.PIXEL, "bad", np.inf, [1, 2], True])
def test_invalid_correction_reference_rejected_before_publication(reference):
    image, _, _ = measurement()
    before = dict(image.metadata)
    with pytest.raises((TypeError, ValueError)):
        image.fit.flatten(reference=reference)
    assert dict(image.metadata) == before


def test_nonfinite_raw_pixels_negative_correction_and_intensity_unit_mismatch():
    xx, yy = np.meshgrid(np.arange(6.), np.arange(5.))
    image = ImageFile(10+xx+yy)
    image[0, 0], image[0, 1] = np.inf, -np.inf
    report = image.fit.plane(output="report")
    result = image.fit.flatten(report, reference=-7)
    values = result.to_numpy(masked=False)
    assert np.isposinf(values[0, 0]) and np.isneginf(values[0, 1])
    np.testing.assert_allclose(values[1:], -7, atol=1e-9)
    package = image.export_storage()
    package.dataset.intensity.attrs["units"] = "different"
    with pytest.raises(ValueError, match="intensity units"):
        ImageFile.from_storage(package).fit.evaluate(report)


def test_unknown_factory_is_lazy_and_validated_on_invocation():
    from Stoner.analysis.fitting.models import register_model

    calls = []

    @register_model(name="surface_factory")
    def factory():
        calls.append(True)
        return Plane()

    image, expected, _ = measurement()
    shortcut = image.fit.surface_factory
    assert "surface_factory" in dir(image.fit) and not calls
    np.testing.assert_allclose(shortcut()[0], expected, atol=1e-9)
    np.testing.assert_allclose(shortcut()[0], expected, atol=1e-9)
    assert len(calls) == 2


def test_ragged_frame_limits_and_too_few_reference_samples():
    xx, yy = np.meshgrid(np.arange(5.), np.arange(4.))
    stack = ImageStack([ImageFile(3+xx+yy), ImageFile((3+xx+yy)[:2, :3])])
    frame = stack[1]
    report = frame.fit.plane(output="report")
    assert report.image_fit.to_dict()["sample_count"] == 6
    assert frame.fit.flatten(report).shape == (2, 3)
    with pytest.raises(ValueError, match="more eligible"):
        frame.fit.plane(box=(0, 3, 0, 1))


def test_explicit_native_optimiser_is_honoured():
    image, expected, _ = measurement()
    report = image.fit.plane(method="least_squares", output="report")
    assert report.method == "least_squares"
    np.testing.assert_allclose([report.params[p].value for p in report.model.param_names], expected, atol=1e-8)


def test_near_singular_geometry_rejected():
    x = np.arange(7.)
    y = x.copy()
    y[-1] += 1e-11
    package = ImageFile(np.ones((7, 7))).export_storage()
    package.dataset = package.dataset.assign_coords(x=x, y=y)
    image = ImageFile.from_storage(package)
    with pytest.raises(ValueError, match="ill-conditioned"):
        image.fit.plane(where=np.eye(7, dtype=bool))


def test_fitting_and_correction_do_not_clone_legacy_figures():
    image, _, _ = measurement()
    figure = image.imshow()
    before = set(plt.get_fignums())
    report = image.fit.plane(output="report")
    assert image.metadata["fig"] is figure
    background = image.fit.evaluate(report)
    corrected = image.fit.flatten(report)
    assert set(plt.get_fignums()) == before
    assert not {"ax", "fig"}.intersection(background.metadata)
    assert not {"ax", "fig"}.intersection(corrected.metadata)


@pytest.mark.parametrize("bounds", [(), (None,), None, "bad_box"])
def test_invalid_metadata_boxes_never_open_selectors(monkeypatch, bounds):
    from Stoner.Image.widgets import RegionSelect

    image, _, _ = measurement()
    image.metadata["bad_box"] = bounds
    monkeypatch.setattr(RegionSelect, "__call__", lambda *a, **k: pytest.fail("Implicit GUI selection"))
    with pytest.raises(ValueError, match="four explicit"):
        image.fit.plane(box="bad_box")
