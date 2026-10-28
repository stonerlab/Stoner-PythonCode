"""Protect shared matching and cardinality contracts across scientific consumers."""

import inspect
import re

import numpy as np
import pytest

from Stoner import Data
from Stoner.analysis.fitting.selection import curve_fit_inputs
from Stoner.core.setas import Setas
from Stoner.core.storage import resolve_columns
from Stoner.plot.selection import _uncertainty, resolve_columns as plot_columns
from Stoner.plot.statistics import _row_values, prepare_series


@pytest.fixture
def channels():
    """Return duplicate channels with distinct values, uncertainties and identities."""
    x = np.arange(6, dtype=float)
    return Data(np.column_stack((x, x**2, 2*x + 3, 4*x + 7, np.full(6, 0.2), np.full(6, 0.4))),
                column_headers=["Field A", "Field B", "Signal", "Signal", "Sigma A", "Sigma B"],
                setas="x.yyee")


@pytest.fixture(params=["schema", "data", "bound", "clone", "numerical", "detached"])
def resolver(channels, request):
    """Exercise each resolver entry point, including standalone numerical roles."""
    if request.param == "schema":
        return lambda selector, **kwargs: resolve_columns(channels.export_storage().schema, selector, **kwargs)
    if request.param == "data":
        return channels.find_col
    if request.param == "bound":
        return channels.setas.find_col
    if request.param == "clone":
        return channels.setas.clone.find_col
    if request.param == "numerical":
        return channels[:, :].setas.find_col
    detached = Setas()
    detached.shape = channels.shape
    detached.column_headers = list(channels.column_headers)
    detached(channels.setas.to_list())
    return detached.find_col


@pytest.mark.parametrize("selector, expected", [(0, 0), (-1, 5), ("Field A", 0),
                                               (slice(0, 2), [0, 1]), ([1, 0, 1], [1, 0, 1]),
                                               (re.compile("field a", re.I), [0])])
def test_unchanged_selector_shapes(resolver, selector, expected):
    """Retain positions, explicit repeats, exact singletons and compiled flags."""
    assert resolver(selector) == expected
    assert resolver("Field A", force_list=True) == [0]


@pytest.mark.parametrize("selector, expected", [("Signal", [2, 3]), ("Sig", [2, 3, 4, 5]),
                                               ("Sig*", [2, 3, 4, 5])])
def test_cs2_strings_return_every_match(resolver, selector, expected):
    """Select duplicate exact names and all substring/glob matches consistently."""
    assert resolver(selector) == expected


@pytest.mark.parametrize("first, force_list, expected", [(True, False, 1), (True, True, [1]),
                                                         (False, False, [1, 0, 1])])
def test_cs2_first_is_explicit_and_preserves_force_list(resolver, first, force_list, expected):
    """Truncate the resolved selection only when the caller requests first."""
    # Assert the planned signature before invoking it: an unrelated TypeError
    # must not count as an expected CS2 failure.
    if "first" not in inspect.signature(resolver).parameters:
        assert "kwargs" in inspect.signature(resolver).parameters
        if resolver.__name__ == "<lambda>":
            assert "first" in inspect.signature(resolve_columns).parameters
    assert resolver([1, 0, 1], first=first, force_list=force_list) == expected


@pytest.mark.parametrize("headers, selector, expected", [
    (["sig*", "signal"], "sig*", 0),
    (["signal", "signal error"], "signal", 0),
    (["sssg suffix"], "s*g", 0),  # Whole-header glob fails; regex fallback succeeds.
    (["field", "Field"], re.compile("^field$", re.I), [0, 1]),
])
def test_precedence_and_explicit_regex_that_already_work(headers, selector, expected):
    """Protect exact literals and explicit regex semantics while adding stages."""
    data = Data(np.ones((2, len(headers))), column_headers=headers)
    assert data.find_col(selector) == expected


@pytest.mark.parametrize("headers, selector, expected", [
    (["Field", "a.b"], ".", 1),
    (["H (Oe)", "H (T)"], "H (", [0, 1]),
    (["sig", "sag", "signal"], "s?g", [0, 1]),
    (["sigA", "sigB"], "sig[AB]", [0, 1]),
])
def test_cs2_substring_and_glob_precedence(headers, selector, expected):
    """Defer regex when literal substring or glob matching succeeds."""
    data = Data(np.ones((2, len(headers))), column_headers=headers)
    assert data.find_col(selector) == expected


@pytest.mark.parametrize("selector, error", [([], KeyError), (slice(0, 0), KeyError),
                                          ([0, "absent"], KeyError), ([0, 99], IndexError)])
def test_cs2_first_validates_the_whole_selection(channels, selector, error):
    """Reject empty results and invalid later members before first truncation."""
    assert "first" in inspect.signature(channels.find_col).parameters
    with pytest.raises(error):
        channels.find_col(selector, first=True)


def test_plot_accepts_many_y_but_one_coordinate(channels):
    """Keep dependent channels together and reject ambiguous spatial coordinates."""
    selected = plot_columns(channels, x=0, y=re.compile("^Signal$"))
    assert selected.x == 0 and selected.y == (2, 3) and selected.yerr == (4, 5)
    for coordinate in ("x", "z", "u", "v", "w"):
        with pytest.raises(ValueError, match="exactly one"):
            plot_columns(channels, **{coordinate: re.compile("^Field")})


@pytest.mark.parametrize("consumer", ["uncertainty", "weights"])
def test_per_row_plot_inputs_require_one_column(channels, consumer):
    """Reject several uncertainty/weight columns instead of flattening observations."""
    with pytest.raises(ValueError, match="exactly one"):
        if consumer == "uncertainty":
            _uncertainty(channels, re.compile("^Sigma"))
        else:
            _row_values(channels, re.compile("^Sigma"), "weights")


def test_curve_fit_keeps_multivariate_x_and_independent_channel_masks(channels):
    """Preserve predictor matrices and filter each y/sigma pair on source rows."""
    channels.mask[0, 0] = True
    channels.mask[1, 2] = True
    channels.mask[2, 5] = True
    before = channels.export_storage()
    inputs = curve_fit_inputs(channels, [0, 1], [2, 3], [4, 5], None, None)
    assert len(inputs) == 2
    for fit, rows, yc, sigma in zip(inputs, ([2, 3, 4, 5], [1, 3, 4, 5]), (2, 3), (0.2, 0.4)):
        x, y, errors, columns, full_x, full_y, model_mask, residual_mask = fit
        assert x.shape == (4, 2) and full_x.shape == (6, 2)
        np.testing.assert_array_equal(x, before.to_numpy(masked=False)[rows, :2])
        np.testing.assert_array_equal(y, before.to_numpy(masked=False)[rows, yc])
        np.testing.assert_array_equal(errors, sigma)
        assert columns.xcol == [0, 1] and columns.ycol == [yc]
        assert model_mask[0] and residual_mask[0] and full_y.shape == (6,)
    after = channels.export_storage()
    np.testing.assert_array_equal(after.excluded, before.excluded)
    np.testing.assert_array_equal(after.to_numpy(masked=False), before.to_numpy(masked=False))
    assert after.schema == before.schema


def test_statistics_and_structural_selection_keep_all_compiled_matches(channels):
    """Keep duplicate channels distinct across distribution and structural consumers."""
    assert [series.column for series in prepare_series(channels, re.compile("^Signal$"))] == [2, 3]
    ids = channels.column_ids
    channels.mask[1, 3] = True
    selected = channels._storage_owner.select(columns=[3, 2, 3])
    assert selected.column_ids[:2] == (ids[3], ids[2])
    assert selected.column_ids[2] not in ids
    assert list(selected.roles) == ["y", "y", "y"]
    np.testing.assert_array_equal(selected.mask[1], [True, False, True])
    assert channels.del_column(re.compile("^Signal$")) is channels
    assert channels.column_ids == ids[:2] + ids[4:]


def test_role_assignment_is_multi_column_and_atomic(channels):
    """Apply role-to-selector syntax to every compiled match without changing IDs."""
    ids = channels.column_ids
    channels.setas(y=re.compile("^Field"), reset=False)
    assert channels.setas.to_string() == "yyyyee"
    before = channels.export_storage()
    with pytest.raises(KeyError):
        channels.setas({"y": [2, "absent"]}, reset=False)
    assert channels.export_storage().schema == before.schema and channels.column_ids == ids


def test_insertion_end_and_named_position_are_distinct(channels):
    """Permit the one-past-end append position without treating it as a column."""
    ids = channels.column_ids
    assert channels.add_column(np.ones(6), index=channels.shape[1], header="end") is channels
    assert channels.column_ids[:6] == ids
    assert channels.add_column(np.zeros(6), index="Field B", header="before B") is channels
    assert list(channels.column_headers)[:3] == ["Field A", "before B", "Field B"]
    assert channels.column_ids[2] == ids[1]


def test_scalar_col_args_is_an_observed_first_policy(channels):
    """Record existing scalar conversion separately from exactly-one validation."""
    assert channels._col_args(xcol=[1, 0], ycol=[3, 2], scalar=True).xcol == 1
    assert channels._col_args(xcol=[1, 0], ycol=[3, 2], scalar=True).ycol == 3
    many = channels._col_args(xcol=[1, 0], ycol=[3, 2], scalar=False)
    assert many.xcol == [1, 0] and many.ycol == [3, 2]


def test_cs2_standalone_compiled_duplicates_match_bound(channels):
    """Require standalone Setas to retain each distinct duplicate-header position."""
    standalone = Setas()
    standalone.shape = channels.shape
    standalone.column_headers = list(channels.column_headers)
    assert standalone.find_col(re.compile("^Signal$")) == [2, 3]


def test_cs2_header_to_role_mapping_handles_multiple_matches(channels):
    """Give both mapping directions the same multi-column assignment semantics."""
    channels.setas({re.compile("^Signal$"): "z"}, reset=False)
    assert channels.setas.to_string() == "x.zzee"


def test_cs2_search_requires_one_explicit_predicate_column(channels):
    """Require a deliberate column choice for a scalar row predicate."""
    with pytest.raises(ValueError):
        channels.search_index(re.compile("^Field"), 2.0)


def test_cs2_rename_updates_all_matching_headers_and_preserves_identity(channels):
    """Rename duplicate selections together without changing numerical state."""
    before = channels.export_storage()
    assert channels.rename("Signal", "Moment") is channels
    assert list(channels.column_headers)[2:4] == ["Moment", "Moment"]
    assert channels.column_ids == tuple(column.id for column in before.schema)
    assert channels.setas.to_string() == "x.yyee"
    np.testing.assert_array_equal(channels.to_numpy(), before.to_numpy())


def test_cs2_mapping_append_broadcasts_to_duplicate_matches(channels):
    """Populate every selected channel while keeping unspecified row entries masked."""
    ids = channels.column_ids
    channels += {"Signal": 9.0}
    np.testing.assert_array_equal(channels.to_numpy(masked=False)[-1, 2:4], [9, 9])
    np.testing.assert_array_equal(channels.mask[-1], [True, True, False, False, True, True])
    assert channels.column_ids == ids


def test_cs2_duplicate_detection_uses_every_explicit_key():
    """Distinguish rows sharing one key but differing in another key."""
    data = Data(np.array([[1., 2., 10.], [1., 3., 20.], [1., 2., 30.]]),
                column_headers=["key A", "key B", "value"], setas="x.y")
    assert list(data.find_duplicates(xcol=[0, 1]).values()) == [[0, 2], [1]]
