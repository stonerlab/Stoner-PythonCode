"""Protect composed schema ownership and shared bound/detached role semantics."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import pickle
from uuid import uuid4

import numpy as np
import pytest

from Stoner import Data
from Stoner.core.column_schema import ColumnSchema
from Stoner.core.setas import Setas
from Stoner.core.storage import Column
from Stoner.plot.selection import resolve_columns as plot_columns


@pytest.fixture
def data():
    """Return two x groups, duplicate names, every uncertainty and vector component."""
    values = np.arange(36, dtype=float).reshape(3, 12)
    result = Data(values,
                  column_headers=["x", "dx", "signal", "error", "x2", "dx2", "signal", "error", "z", "dz", "u", "v"],
                  setas="xdyexdyezfuv")
    result.mask[1, 6] = True
    result.metadata["run{String}"] = "001"
    return result


@pytest.fixture(params=["bound", "detached"])
def roles(data, request):
    """Run the same role language against an owner view and a detached facade."""
    return data.setas if request.param == "bound" else data.setas.clone


def test_owner_schema_is_immutable_and_interchange_stays_an_editable_list(data):
    """Keep one immutable authority while retaining version-one export contracts."""
    schema = data._storage_owner._state.schema
    assert isinstance(schema, ColumnSchema) and schema.ids == data.column_ids
    assert schema.headers == tuple(data.column_headers) and schema.roles == tuple(data.setas)
    assert deepcopy(schema) is schema
    with pytest.raises(FrozenInstanceError):
        schema._columns = ()
    with pytest.raises(FrozenInstanceError):
        schema[0].role = "y"
    package = data.export_storage()
    assert isinstance(package.schema, list)
    package.schema[0] = Column(str(uuid4()), "other", "z")
    assert data._storage_owner._state.schema is schema


def test_schema_selection_and_insertion_share_stable_identity_rules(data):
    """Retain original identities once and allocate new identities for copies."""
    schema = data._storage_owner._state.schema
    selected = schema.take([6, 0, 6])
    assert selected.ids[:2] == (schema[6].id, schema[0].id)
    assert selected.ids[2] not in schema.ids
    assert selected.headers == ("signal", "x", "signal") and selected.roles == ("y", "x", "y")
    inserted = schema.insert(1, [schema[0]])
    assert inserted[0].id == schema[0].id and inserted[1].id not in schema.ids
    assert inserted.ids[2:] == schema.ids[1:]
    assert schema.resized(14).ids[:12] == schema.ids
    assert schema.resized(14).roles[-2:] == (".", ".")
    assert schema.resized(2).ids == schema.ids[:2]


@pytest.mark.parametrize("field,key,value,error", [("role", 0, "bad", ValueError),
                                                   ("header", 0, 12, TypeError),
                                                   ("role", slice(None), ["x"], ValueError)])
def test_invalid_field_replacement_cannot_change_owner(data, field, key, value, error):
    """Validate all records before replacing either frame labels or schema."""
    schema = data._storage_owner._state.schema
    frame = data.data
    with pytest.raises(error):
        schema.with_field(field, key, value)
    assert data._storage_owner._state.schema is schema and data.data is frame
    assert tuple(frame.columns) == schema.headers


def test_saved_views_follow_rename_reorder_repeated_selection_and_role_edits(data):
    """Reacquire owner state through saved facade and schema-field views."""
    roles = data.setas
    headers = roles.column_headers
    explicit = roles.setas
    ids = data.column_ids
    data.rename("signal", "moment")
    assert headers[2] == headers[6] == "moment"
    data.reorder_columns([6, 4, 5, 6])
    assert roles.to_list() == ["y", "x", "d", "y"]
    assert list(explicit) == roles.to_list() and list(headers) == ["moment", "x2", "dx2", "moment"]
    assert data.column_ids[:3] == (ids[6], ids[4], ids[5]) and data.column_ids[3] not in ids
    roles[[0, 3]] = ["z", "y"]
    assert roles.to_string() == "zxdy" and data.setas.to_string() == "zxdy"
    assert data.mask[1, 0] and data.mask[1, 3]


def test_bound_reads_and_edits_do_not_construct_setas_or_export_values(data, monkeypatch):
    """Read and mutate schema directly without reconstructing detached facades."""
    roles = data.setas
    frame = data.data
    metadata = data._storage_owner._state.metadata

    def forbidden(*args, **kwargs):
        pytest.fail("Schema-only operation constructed Setas or exported numerical values")

    monkeypatch.setattr(Setas, "__init__", forbidden)
    monkeypatch.setattr(data._storage_owner, "to_numpy", forbidden)
    assert roles.find_col("signal") == [2, 6]
    assert roles.x == 0 and roles.y == [2]
    assert roles.to_dict()["y"] == ["signal", 6]
    assert len(list(roles.items())) == 12
    assert len(roles.set) == 12 and len(roles.not_set) == 12
    assert roles.to_string() == "xdyexdyezfuv"
    roles(y="signal", reset=False)
    roles.update({"z": 8})
    roles -= {"z": 8}
    roles += {"z": 8}
    assert roles._get_cols(startx=4).xerr == 5
    assert data.data is frame and data._storage_owner._state.metadata is metadata


def test_role_cloning_does_not_copy_the_data_owner(data, monkeypatch):
    """Detach role/header state while retaining callable return-to-owner semantics."""
    def forbidden(*args, **kwargs):
        pytest.fail("Cloning role descriptions attempted to clone Data")

    monkeypatch.setattr(Data, "__deepcopy__", forbidden)
    cloned = data.setas.clone
    assert cloned(y=0, reset=False) is data
    assert cloned.to_list() != data.setas.to_list()
    cloned.column_headers[0] = "changed"
    assert data.column_headers[0] == "x"
    nested = cloned.clone
    assert nested.to_list() == cloned.to_list() and nested._object is data


@pytest.mark.parametrize("start,x,y,dx,dy", [(0, 0, [2], 1, [3]), (4, 4, [6], 5, [7])])
def test_group_positions_are_absolute_and_common_to_both_facades(roles, start, x, y, dx, dy):
    """Derive all groups without bridge/plot uncertainty corrections."""
    columns = roles._get_cols(startx=start)
    assert (columns.xcol, columns.ycol, columns.xerr, columns.yerr) == (x, y, dx, dy)
    assert roles._get_cols("xerr", startx=start) == dx
    assert roles._get_cols("ycols", startx=start) == y
    assert roles["#x"] == [0, 4] and roles["#y"] == [2, 6]
    assert roles["signal"] == ["y", "y"]
    assert roles[["signal", 0]] == ["y", "y", "x"]


def test_plot_and_numerical_results_use_the_same_second_group(data):
    """Keep owner, detached numerical annotation and plot uncertainty positions equal."""
    owner = data.setas._get_cols(startx=4)
    detached = data.asarray()._setas._get_cols(startx=4)
    plotted = plot_columns(data, startx=4)
    assert owner.xerr == detached.xerr == plotted.xerr == 5
    assert owner.yerr == detached.yerr == list(plotted.yerr) == [7]


@pytest.mark.parametrize("width,roles_string", [(2, "xy"), (3, "xye"), (4, "xdye"),
                                              (5, "xyuvw"), (6, "xyzuvw")])
def test_defaults_are_fresh_and_never_written_into_explicit_roles(width, roles_string):
    """Infer intelligent count defaults without retaining mutable group dictionaries."""
    data = Data(np.ones((3, width)))
    roles = data.setas
    inferred = roles.cols
    inferred.ycol.append(99)
    expected = [2] if width == 4 else [1]
    assert roles.cols.ycol == expected
    assert roles.to_list() == ["."] * width and roles.clone.to_list() == ["."] * width
    assert roles._get_cols(no_guess=True).xcol is None
    explicit = roles.clone
    explicit(roles_string)
    assert explicit.x == roles.x


@pytest.mark.parametrize("operation", ["call", "assignment", "update", "add", "subtract"])
def test_late_invalid_role_edits_are_atomic_in_both_facades(roles, operation):
    """Prepare the entire role result before committing a bound or detached edit."""
    before = roles.to_list()
    with pytest.raises((KeyError, ValueError)):
        if operation == "call":
            roles({"y": [0, "absent"]}, reset=True)
        elif operation == "assignment":
            roles[[0, 1]] = ["z", "invalid"]
        elif operation == "update":
            roles.update({"z": 0, "y": "absent"})
        elif operation == "add":
            roles += {"z": 0, "y": "absent"}
        else:
            roles -= [0, "absent"]
    assert roles.to_list() == before


def test_mapping_update_keeps_all_targets_of_each_replaced_group(roles):
    """Clear an updated role group once, then process every mapping entry in order."""
    roles.update({"x": 0, "z": "signal"})
    assert roles["#z"] == [2, 6]
    roles.update({"dx": "z", "dx2": "z"})
    assert roles["#z"] == [1, 5]


def test_edit_guard_precedes_even_invalid_role_parsing(data):
    """Reject attempted schema writes before parsing while a numerical draft is active."""
    roles = data.setas
    before = roles.to_list()
    with data.edit_numpy() as draft:
        draft[0, 0] = 99
        for write in (lambda: roles.__setitem__(None, "invalid"),
                      lambda: roles.update({"y": "absent"}),
                      lambda: roles("bad"), lambda: roles.clear()):
            with pytest.raises(RuntimeError, match="editing"):
                write()
        assert roles.to_list() == before
    assert data[0, 0] == 99 and roles.to_list() == before


def test_call_return_conventions_and_reset_hyphens_are_shared(roles):
    """Keep _self/reset semantics while using the same validated role parser."""
    assert roles(y=0, reset=False, _self=True) is roles
    assert roles(x=4, reset=False, _self=None) is None
    assert roles(_self=True) == roles.to_list()
    before = roles.to_list()
    roles("-z", reset=False)
    assert roles.to_list()[0] == before[0] and roles.to_list()[1] == "z"
    roles("-z", reset=True)
    assert roles.to_list() == [".", "z"] + ["."] * 10


def test_previous_owner_pickle_schema_lists_are_upgraded(data):
    """Upgrade old schema state without changing the existing pickle boundary."""
    from Stoner.core.storage_owner import _TableState

    old = _TableState.__new__(_TableState)
    old.__dict__.update(data._storage_owner._state.__dict__)
    old.schema = list(old.schema)
    # Compare metadata with its existing independent pickle behaviour: the schema
    # upgrade must not silently claim a new lossless TypeHintedDict pickle format.
    expected_metadata = pickle.loads(pickle.dumps(old.metadata))
    restored = pickle.loads(pickle.dumps(old))
    assert isinstance(restored.schema, ColumnSchema) and restored.schema.ids == data.column_ids
    np.testing.assert_array_equal(restored.excluded, data.mask)
    np.testing.assert_array_equal(restored.values, data.data)
    assert restored.metadata == expected_metadata
    assert restored.metadata.type("run") == expected_metadata.type("run")


def test_owner_clone_retains_typed_metadata_and_detaches_mutable_state(data):
    """Share only immutable descriptions while copying values, masks and metadata."""
    data.metadata["history"] = [1]
    cloned = data.clone
    assert cloned._storage_owner._state.schema is data._storage_owner._state.schema
    assert cloned.metadata["run"] == "001" and cloned.metadata.type("run") == "String"
    cloned.metadata["history"].append(2)
    cloned.mask[1, 6] = False
    cloned[0, 0] = -1
    cloned.setas[0] = "z"
    assert data.metadata["history"] == [1] and data.mask[1, 6]
    assert data[0, 0] == 0 and data.setas.to_string() == "xdyexdyezfuv"


def test_legacy_detached_state_drops_only_redundant_caches(data):
    """Keep the former private pickle name as an alias to the common facade."""
    from Stoner.core.storage_bridge import _ResolvedSetas

    assert _ResolvedSetas is Setas
    roles = Setas.__new__(Setas)
    state = dict(data.setas.clone.__dict__, _schema=list(data._storage_owner._state.schema),
                 _cols={"xerr": -99}, _col_defaults={})
    roles.__setstate__(state)
    assert not {"_schema", "_cols", "_col_defaults"}.intersection(roles.__dict__)
    assert roles._get_cols(startx=4).xerr == 5


def test_detached_subclass_clone_retains_type_and_independent_roles():
    """Preserve direct Setas subclass compatibility while detaching column descriptions."""
    class SpecialSetas(Setas):
        pass

    roles = SpecialSetas()
    roles.shape = (3, 2)
    roles("xy")
    cloned = roles.clone
    assert isinstance(cloned, SpecialSetas)
    cloned[0] = "z"
    assert roles.to_string() == "xy" and cloned.to_string() == "zy"


@pytest.mark.parametrize("result", [True, 0, False])
def test_fitting_publication_reenters_the_native_schema_boundary(result):
    """Publish fitting packages with header-labelled values and composed owner state."""
    def line(x, slope, offset):
        return slope * x + offset

    x = np.arange(8, dtype=float)
    data = Data(x, line(x, 2, 3), column_headers=["field", "signal"], setas="xy")
    data.metadata["run{String}"] = "001"
    ids = data.column_ids
    roles = data.setas
    data.curve_fit(line, result=result)
    assert isinstance(data._storage_owner._state.schema, ColumnSchema)
    assert tuple(data.data.columns) == data._storage_owner._state.schema.headers
    assert roles.to_list() == data.setas.to_list()
    assert data.metadata["run"] == "001" and data.metadata.type("run") == "String"
    if result is True:
        assert data.column_ids[:2] == ids
    elif result is False:
        assert data.column_ids == ids
    else:
        assert data.column_ids[1:] == ids
