"""Verify the authoritative frame owner before connecting legacy Data descriptors."""

import numpy as np
import pytest

from Stoner.core.storage import DataStorage
from Stoner.core.storage_owner import DataOwner


@pytest.fixture
def owner():
    """Provide duplicate columns, stable identities and one excluded integer."""
    values = np.ma.array([[0, 10, 11], [1, 100, 101]], dtype=np.int16,
                         mask=[[False, False, False], [False, True, False]], fill_value=-9)
    return DataOwner(DataStorage.from_numpy(values, headers=["x", "y", "y"], roles="xyy"))


def test_schema_masks_and_values_share_one_owner(owner):
    """Keep IDs fixed through header, role, mask and promoted value edits."""
    ids = owner.column_ids
    headers, roles, mask = owner.headers, owner.roles, owner.mask
    headers[1] = "signal"
    roles[:] = "xye"
    mask[1, 1] = False
    assert owner[1, "signal"] == 100
    owner[0, "signal"] = 2.5
    assert owner[0, 1] == 2.5
    assert owner.to_numpy().dtype.kind == "f"
    assert owner.column_ids == ids
    assert list(roles) == list("xye")
    assert headers[1] == "signal"


def test_owner_boundary_detaches_packages_and_snapshots(owner):
    """Never share mutable arrays with imports, exports, saved slices or clones."""
    package = owner.export_storage()
    cloned = DataOwner(package)
    package.values.iloc[0, 0] = 77
    assert cloned[0, 0] == owner[0, 0] == 0
    snapshot = owner[:, "y"]
    with pytest.raises(ValueError):
        snapshot[0] = 99
    with pytest.raises(ValueError):
        snapshot.mask[0] = True
    raw = owner.to_numpy(masked=False)
    raw[:] = 90
    assert owner[0, 0] == 0
    cloned[0, 0] = 80
    assert owner[0, 0] == 0


def test_masked_assignment_and_nan_are_distinct(owner):
    """Recover excluded integers and preserve NaN's existing unmasked semantics."""
    owner[0, 1] = np.ma.masked
    assert owner[0, 1] is np.ma.masked
    assert owner.to_numpy(masked=False)[0, 1] == 10
    owner[0, 1] = 20
    assert not owner.mask[0, 1]
    owner[0, 2] = np.nan
    assert np.isnan(owner[0, 2]) and not owner.mask[0, 2]


@pytest.mark.parametrize("field, value", [("roles", ["x", "y", "invalid"]),
                                         ("headers", ["x", 5, "z"]), ("roles", ["x"])])
def test_schema_edits_fail_atomically(owner, field, value):
    """Reject malformed partial schema edits before publishing any records."""
    previous = owner.export_storage()
    with pytest.raises((TypeError, ValueError)):
        getattr(owner, field)[:] = value
    assert owner.export_storage().schema == previous.schema


@pytest.mark.parametrize("context", ["edit_numpy", "edit_pandas"])
def test_transactions_commit_once_and_rollback(owner, context):
    """Retained drafts and aborted edits cannot affect committed values."""
    with getattr(owner, context)() as draft:
        if context == "edit_numpy":
            draft[0, 1] = 50
        else:
            draft.iloc[0, 1] = 50
    assert owner[0, 1] == 50
    if context == "edit_numpy":
        draft[0, 1] = 70
    else:
        draft.iloc[0, 1] = 70
    assert owner[0, 1] == 50
    with pytest.raises(RuntimeError, match="cancel"):
        with getattr(owner, context)() as draft:
            raise RuntimeError("cancel")
    assert owner[0, 1] == 50


def test_saved_proxy_locks_and_nested_metadata(owner):
    """Block stale mapping/proxy writes while preserving nested metadata edits."""
    metadata, roles, headers, mask = owner.metadata, owner.roles, owner.headers, owner.mask
    metadata["run{String}"] = "001"
    metadata["history"] = [1, 2]
    history = metadata["history"]
    with owner.edit_numpy() as draft:
        for proxy, key, value in [(metadata, "run", 9), (roles, 1, "e"),
                                  (headers, 1, "bad"), (mask, (0, 0), True), (owner, (0, 0), 7)]:
            with pytest.raises(RuntimeError):
                proxy[key] = value
        with pytest.raises(RuntimeError):
            with owner.edit_pandas():
                pass
        history.append(3)
        draft[0, 1] = 12
    assert metadata["history"] == [1, 2, 3]
    assert metadata["run"] == "001" and metadata.type("run") == "String"


@pytest.mark.filterwarnings("ignore:Setting the dtype on a MaskedArray has been deprecated:DeprecationWarning")
def test_failed_draft_validation_unlocks_owner(owner):
    """Reject shape/dtype changes and ensure the next edit can proceed."""
    with pytest.raises(ValueError):
        with owner.edit_numpy() as draft:
            # Deliberately corrupt the draft; this old setter warns on NumPy 2.5.
            draft.dtype = np.uint16
    with pytest.raises(ValueError):
        with owner.edit_pandas() as draft:
            draft.columns = ["a", "b", "c"]
    owner[0, 0] = 9
    assert owner[0, 0] == 9


def test_reject_implicit_large_integer_precision_loss():
    """Do not round uint64 values when a signed or floating assignment promotes."""
    owner = DataOwner(DataStorage.from_numpy(np.array([[2**64 - 1, 2]], dtype=np.uint64)))
    with pytest.raises(ValueError, match="precision"):
        owner[0, 1] = -1
    assert owner.to_numpy(masked=False)[0, 0] == 2**64 - 1
    with pytest.raises(TypeError):
        owner[0, 1] = "bad"
    assert owner[0, 1] == 2
