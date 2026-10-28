"""Executable ownership, cardinality and native pandas compatibility gates."""

from copy import copy, deepcopy
from dataclasses import replace
import re

import numpy as np
import pandas as pd
import pytest

from Stoner.core.column_schema import ColumnSchema, _assign_roles
from Stoner.core.storage_owner import DataOwner
from .model import ColumnIndex, IndexOwner, combine, fixture, pickle_roundtrip, real_fixture


def assert_package(left, right, *, same_ids=True):
    np.testing.assert_array_equal(left.values.to_numpy(), right.values.to_numpy())
    np.testing.assert_array_equal(left.excluded, right.excluded)
    assert [(c.header, c.role) for c in left.schema] == [(c.header, c.role) for c in right.schema]
    if same_ids:
        assert [c.id for c in left.schema] == [c.id for c in right.schema]
    assert left.metadata == right.metadata
    assert left.metadata.types == right.metadata.types
    assert left.fill_value == right.fill_value
    assert left.dtype == right.dtype


@pytest.mark.parametrize("width", [0, 1, 8, 128])
@pytest.mark.parametrize("rows", [0, 4])
def test_owner_construction(width, rows):
    package = fixture(width, rows)
    composed, indexed = DataOwner(package), IndexOwner(package)
    assert_package(composed.export_storage(), indexed.export_storage())
    assert isinstance(indexed.frame.columns, ColumnIndex)
    assert "schema" not in vars(indexed._state)
    assert indexed._state.schema is indexed.frame.columns.schema


@pytest.mark.parametrize("selector", [0, -1, "Signal", "Sig", "S*", "literal*", "Temperature [K]",
                                     re.compile("signal", re.I), "^Sign.*$", ["Signal", 0, "Signal"],
                                     slice(None, None, -1), [], "missing", "[", True, [0, 999]])
@pytest.mark.parametrize("first,force_list", [(False, False), (False, True), (True, False), (True, True)])
def test_cs2_matching(selector, first, force_list):
    schema = ColumnSchema(fixture().schema)
    index = ColumnIndex(schema)
    try:
        expected = schema.resolve(selector, first=first, force_list=force_list)
    except Exception as error:
        with pytest.raises(type(error)):
            index.find_col(selector, first=first, force_list=force_list)
    else:
        assert index.find_col(selector, first=first, force_list=force_list) == expected


def test_pandas_exact_duplicates_equality_alignment_and_hashes():
    owner = IndexOwner(fixture())
    index = owner.frame.columns
    plain = pd.Index(index, dtype=object)
    np.testing.assert_array_equal(index.get_loc("Signal"), plain.get_loc("Signal"))
    with pytest.raises(KeyError):
        index.get_loc("Sig")
    changed = ColumnIndex(ColumnSchema(replace(c, id=fixture(1).schema[0].id if i == 0 else c.id,
                                               role=".") for i, c in enumerate(index.schema)))
    assert index.equals(changed) and index.equals(plain)
    np.testing.assert_array_equal(pd.util.hash_pandas_object(index), pd.util.hash_pandas_object(changed))
    frame = owner.frame
    other = frame.copy()
    other.columns = changed
    np.testing.assert_array_equal((frame + other).to_numpy(), frame.to_numpy() * 2)
    assert frame.loc[:, "Signal"].shape[1] == 2


@pytest.mark.parametrize("owner_type", [DataOwner, IndexOwner])
def test_saved_views_atomicity_no_numeric_copy(owner_type):
    owner = owner_type(fixture())
    headers, roles = owner.headers, owner.roles
    frame, mask = owner.frame, owner._state.excluded
    raw = frame.to_numpy(copy=False)
    old_schema = owner._state.schema
    old_axis = frame.columns
    owner.roles[:] = _assign_roles(tuple(headers), tuple(roles), "Signal", "y")
    owner.headers[1] = "renamed"
    assert owner.frame is frame and owner._state.excluded is mask
    assert np.shares_memory(raw, owner.frame.to_numpy())
    assert roles[1] == "y" and headers[1] == "renamed"
    assert old_schema.headers[1] == "Signal"
    if owner_type is IndexOwner:
        assert old_axis.schema is old_schema
        assert owner.frame.columns is not old_axis
    before = owner.export_storage()
    with pytest.raises(ValueError):
        roles[:] = ["y"]
    assert_package(before, owner.export_storage())
    owner.take(columns=[3, 1, 1, 0])
    assert list(headers) == ["Error", "renamed", "renamed", "Field (T)"]
    assert list(roles) == ["y", "y", "y", "x"]
    assert len(set(owner.column_ids)) == 4
    with owner.edit_numpy():
        with pytest.raises(RuntimeError):
            roles[0] = "z"


@pytest.mark.parametrize("transform", [copy, deepcopy, pickle_roundtrip])
def test_index_copy_and_pickle(transform):
    index = ColumnIndex(fixture().schema, name="channels")
    result = transform(index)
    assert isinstance(result, ColumnIndex)
    assert result.schema == index.schema
    assert result.name == index.name


@pytest.mark.parametrize("transform", [deepcopy, pickle_roundtrip])
def test_owner_and_frame_copy(transform):
    owner = IndexOwner(fixture())
    frame = transform(owner.frame)
    assert isinstance(frame.columns, ColumnIndex)
    assert frame.columns.schema == owner._state.schema
    result = transform(owner)
    assert_package(owner.export_storage(), result.export_storage())
    result.metadata["run"]["nested"].append(3)
    result.roles[0] = "z"
    result.frame.iloc[0, 0] = -99
    result.mask[0, 0] = True
    assert owner.roles[0] == "x"
    assert owner.frame.iloc[0, 0] == 0
    assert owner.metadata["run"]["nested"] == [1, 2]
    assert not owner.mask[0, 0]


@pytest.mark.parametrize("positions", [[2, 0, 1], [1, 1, 1], [], [7, 0]])
def test_paired_owner_selection(positions):
    package = fixture()
    left = DataOwner(package).select(columns=positions).export_storage()
    right = IndexOwner(package).select(columns=positions).export_storage()
    assert_package(left, right, same_ids=len(set(positions)) == len(positions))
    assert len({c.id for c in right.schema}) == len(positions)
    for i, position in enumerate(positions):
        if position not in positions[:i]:
            assert right.schema[i].id == package.schema[position].id
        else:
            assert right.schema[i].id not in {c.id for c in package.schema}


def test_real_fixture_pair():
    package = real_fixture()
    assert_package(DataOwner(package).export_storage(), IndexOwner(package).export_storage())


def test_native_reentry_requires_axis_proof_and_explicit_side_state():
    owner = IndexOwner(fixture())
    frame = owner.frame.copy()
    result = IndexOwner.reenter(frame, expected_schema=owner._state.schema,
                                mask=owner._state.excluded, metadata=owner._state.metadata,
                                fill_value=owner._state.fill_value, dtype=owner._state.dtype)
    assert_package(owner.export_storage(), result.export_storage())
    frame.columns = pd.Index(list(frame.columns))
    with pytest.raises(ValueError, match="Ordinary Index"):
        IndexOwner.reenter(frame, expected_schema=owner._state.schema,
                           mask=owner._state.excluded, metadata=owner._state.metadata,
                           fill_value=owner._state.fill_value, dtype=owner._state.dtype)
    owner.frame.columns = pd.Index(list(owner.frame.columns))
    with pytest.raises(ValueError, match="Ordinary Index"):
        owner.export_storage()


def test_native_side_metadata_conflict_never_inferred_from_labels():
    owner = IndexOwner(fixture())
    other = owner.frame.copy()
    wrong = owner._state.schema.with_field("role", 0, "z")
    other.columns = ColumnIndex(wrong)
    with pytest.raises(ValueError, match="transformation schema"):
        IndexOwner.reenter(other, expected_schema=owner._state.schema,
                           mask=owner._state.excluded, metadata=owner._state.metadata,
                           fill_value=owner._state.fill_value, dtype=owner._state.dtype)


@pytest.mark.parametrize("owner_type", [DataOwner, IndexOwner])
@pytest.mark.parametrize("axis", [0, 1])
def test_explicit_combination_policy(owner_type, axis):
    package = fixture()
    owners = [owner_type(package), owner_type(package)]
    result = combine(owners, axis=axis, owner_type=owner_type).export_storage()
    np.testing.assert_array_equal(result.to_numpy().data,
                                  np.concatenate([package.to_numpy().data] * 2, axis=axis))
    np.testing.assert_array_equal(result.excluded, np.concatenate([package.excluded] * 2, axis=axis))
    assert result.metadata == package.metadata
    assert len({c.id for c in result.schema}) == len(result.schema)
    assert result.schema[:8] == package.schema
    if axis == 1:
        assert [c.role for c in result.schema[8:]] == [c.role for c in package.schema]
    else:
        owners[1].roles[0] = "z"
        with pytest.raises(ValueError, match="identical scientific"):
            combine(owners, axis=axis, owner_type=owner_type)
    owners[1].metadata["different"] = 1
    with pytest.raises(ValueError, match="typed metadata"):
        combine(owners, axis=axis, owner_type=owner_type)
    with pytest.raises(ValueError, match="explicit owners"):
        combine([owners[0], owners[1].frame], axis=axis, owner_type=owner_type)


@pytest.mark.parametrize("owner_type", [DataOwner, IndexOwner])
def test_inherited_structural_and_transaction_machinery(owner_type):
    owner = owner_type(fixture())
    original = owner.export_storage()
    with pytest.raises(ValueError):
        with owner.edit_pandas() as draft:
            draft.columns = list(reversed(draft.columns))
    assert_package(original, owner.export_storage())
    with owner.edit_pandas() as draft:
        draft.iloc[0, 0] = 123
    assert owner.frame.iloc[0, 0] == 123
    assert owner.column_ids == tuple(c.id for c in original.schema)
    with owner.edit_numpy() as draft:
        draft[0, 0] = 124
    assert owner.frame.iloc[0, 0] == 124
    owner.insert_rows(1, np.ma.array([[7] * 8], mask=[[True] + [False] * 7]))
    assert owner.mask[1, 0]
    owner.insert_columns(2, np.ma.array(np.ones((65, 1)), mask=True))
    assert len(owner.column_ids) == 9
    assert owner._state.excluded[:, 2].all()
    assert tuple(owner.headers) == owner._state.schema.headers
    assert tuple(owner.frame.columns) == tuple(owner.headers)


@pytest.mark.parametrize("selector", [slice(None, None, -1), [2, 0], [1, 1], np.array([True, False] * 4)])
def test_native_axis_selection_records(selector):
    owner = IndexOwner(fixture())
    index = owner.frame.columns
    result = index[selector]
    positions = np.arange(len(index))[selector]
    assert result.schema.headers == tuple(index[i] for i in positions)
    assert result.schema.roles == tuple(index.schema[i].role for i in positions)
    assert len(set(result.schema.ids)) == len(result)
    frame = owner.frame.iloc[:, selector]
    assert isinstance(frame.columns, ColumnIndex)
    assert frame.columns.schema.headers == result.schema.headers
    np.testing.assert_array_equal(frame.to_numpy(), owner.frame.to_numpy()[:, selector])


def test_native_matrix_records_every_boundary():
    from .assess import native_matrix

    rows = native_matrix()
    assert len(rows) == 52
    index = {row["operation"]: row for row in rows if row["backend"] == "index"}
    for operation in ("copy_shallow", "copy_deep", "iloc_reorder", "iloc_repeat", "iloc_slice",
                      "loc_duplicates", "take", "drop", "pickle"):
        assert index[operation]["schema"] == "valid"
    for operation in ("rename", "insert", "concat_columns_same", "reindex_unique", "replace_ordinary"):
        assert index[operation]["schema"] == "lost"
    assert index["reindex_duplicate_source"]["error"] == "ValueError"
    assert index["concat_rows_conflict"]["boundary"].startswith("unsupported native combination")
    for row in rows:
        assert row["boundary"]


def test_paired_owner_matrix():
    from .assess import owner_matrix

    rows = owner_matrix()
    assert len(rows) == 24
    left, right = rows[:12], rows[12:]
    for composed, indexed in zip(left, right):
        for key in ("operation", "labels", "roles", "shape", "dtype", "values", "excluded",
                    "metadata", "metadata_types", "identity_origin", "shares_values", "shares_mask",
                    "same_owner", "same_frame",
                    "saved_headers", "saved_roles"):
            assert composed[key] == indexed[key], (composed["operation"], key)
        assert len(set(indexed["ids"])) == len(indexed["ids"])
        if composed["operation"] not in ("repeat", "concat_columns", "insert"):
            assert composed["ids"] == indexed["ids"]


@pytest.mark.parametrize("owner_type", [DataOwner, IndexOwner])
def test_actual_bound_setas_uses_experimental_owner(owner_type):
    from Stoner import Data

    package = fixture()
    data = Data.from_storage(package)
    # Test-only injection: production never imports or selects IndexOwner.
    data._storage_owner = owner_type(package)
    roles = data.setas
    roles["Signal"] = "y"
    data.column_headers[1] = "renamed"
    assert roles.find_col("Signal") == 2
    assert roles.to_list()[1:3] == ["y", "y"]
    data._storage_owner.take(columns=[3, 1, 1, 0])
    assert roles.to_list() == ["y", "y", "y", "x"]
    assert roles.find_col("renamed") == [1, 2]
    detached = roles.clone
    roles[0] = "z"
    assert detached.to_list()[0] == "y"
    assert roles.to_list()[0] == "z"


def test_pandas_draft_cannot_replace_roles_just_because_labels_compare_equal():
    owner = IndexOwner(fixture())
    before = owner.export_storage()
    with pytest.raises(ValueError, match="scientific columns"):
        with owner.edit_pandas() as draft:
            draft.columns = draft.columns.with_field("role", 0, "z")
    assert_package(before, owner.export_storage())
