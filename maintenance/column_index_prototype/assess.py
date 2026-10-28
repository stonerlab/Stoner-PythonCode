"""Record native pandas propagation, including rejected and lossy boundaries."""

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Stoner.core.column_schema import ColumnSchema
from Stoner.core.storage_owner import DataOwner
from .model import ColumnIndex, IndexOwner, combine, fixture, pickle_roundtrip


def describe(frame, original):
    """Describe the actual result rather than reconstructing state from labels."""
    axis = frame.columns if isinstance(frame, pd.DataFrame) else frame.index
    result = dict(type=type(frame).__name__, axis_type=type(axis).__name__,
                  axis_dtype=str(axis.dtype), labels=list(axis), shape=list(frame.shape),
                  shares_values=bool(np.shares_memory(frame.to_numpy(), original.to_numpy())),
                  external_mask="not carried by pandas", typed_metadata="not carried by pandas")
    try:
        schema = axis.schema
        result.update(ids=list(schema.ids), roles=list(schema.roles), schema="valid")
    except (AttributeError, ValueError) as error:
        result.update(schema="lost", detail=str(error))
    return result


def _native_matrix(original, backend):
    """Keep native transforms honest, even where they discard scientific state."""
    ordinary = original.copy()
    ordinary.columns = pd.Index(list(ordinary.columns))
    conflict = original.copy()
    if isinstance(original.columns, ColumnIndex):
        conflict.columns = ColumnIndex(ColumnSchema(replace(c, role=".") for c in original.columns.schema))
    unique = original.iloc[:, [0, 3, 4]]
    operations = {
        "copy_shallow": lambda f: f.copy(deep=False),
        "copy_deep": lambda f: f.copy(deep=True),
        "deepcopy": deepcopy,
        "rename": lambda f: f.rename(columns={"Signal": "renamed"}),
        "rename_axis": lambda f: f.rename_axis(columns="channels"),
        "iloc_reorder": lambda f: f.iloc[:, [3, 0, 1]],
        "iloc_repeat": lambda f: f.iloc[:, [1, 1, 0]],
        "iloc_slice": lambda f: f.iloc[:, ::2],
        "loc_duplicates": lambda f: f.loc[:, ["Signal"]],
        "take": lambda f: f.take([3, 0, 1], axis=1),
        "drop": lambda f: f.drop(columns="Signal"),
        "insert": lambda f: insert(f),
        "concat_rows_same": lambda f: pd.concat([f, f], ignore_index=True),
        "concat_rows_conflict": lambda f: pd.concat([f, conflict], ignore_index=True),
        "concat_rows_ordinary": lambda f: pd.concat([f, ordinary], ignore_index=True),
        "concat_columns_same": lambda f: pd.concat([f, f], axis=1),
        "concat_columns_ordinary": lambda f: pd.concat([f, ordinary], axis=1),
        "reindex_duplicate_source": lambda f: f.reindex(columns=["Signal", "Field (T)"]),
        "reindex_unique": lambda f: unique.reindex(columns=["Error", "Field (T)"]),
        "reindex_add": lambda f: unique.reindex(columns=["Field (T)", "new"]),
        "align_unique": lambda f: unique.align(pd.DataFrame({"new": [1]}), axis=1)[0],
        "series": lambda f: f.iloc[:, 0],
        "series_to_frame": lambda f: f.iloc[:, 0].to_frame(),
        "plain_export": lambda f: pd.DataFrame(f.to_numpy(copy=True), columns=list(f.columns)),
        "pickle": pickle_roundtrip,
        "replace_ordinary": lambda f: set_axis(f),
    }
    rows = []
    for operation, transform in operations.items():
        try:
            result = transform(original.copy())
            row = dict(backend=backend, operation=operation, **describe(result, original))
            row["boundary"] = ("lossy export" if operation in {"series", "series_to_frame", "plain_export"}
                               else "requires explicit transform proof and side state" if row["schema"] == "valid"
                               else "reject re-entry")
            if operation in {"concat_rows_conflict", "concat_rows_ordinary"} and row["schema"] == "valid":
                row["boundary"] = "unsupported native combination: valid-looking left schema is insufficient"
        except Exception as error:
            row = dict(backend=backend, operation=operation, boundary="native operation rejected",
                       error=type(error).__name__, detail=str(error))
        rows.append(row)
    return rows


def native_matrix():
    package = fixture()
    return (_native_matrix(DataOwner(package).frame, "composed") +
            _native_matrix(IndexOwner(package).frame, "index"))


def owner_matrix():
    """Record complete side state and detachment for paired supported operations."""
    package = fixture()
    result = []
    for backend, owner_type in [("composed", DataOwner), ("index", IndexOwner)]:
        operations = {
            "construct": lambda owner: owner,
            "rename": lambda owner: edit(owner, "header"),
            "roles": lambda owner: edit(owner, "role"),
            "reorder": lambda owner: owner.select(columns=[3, 0, 1]),
            "repeat": lambda owner: owner.select(columns=[1, 1, 0]),
            "drop": lambda owner: owner.select(columns=[0, 3, 4, 5, 6, 7]),
            "insert": lambda owner: insert_owner(owner),
            "clone": deepcopy,
            "pickle": pickle_roundtrip,
            "package_import": lambda owner: owner_type(owner.export_storage()),
            "concat_rows": lambda owner: combine([owner, owner], axis=0, owner_type=owner_type),
            "concat_columns": lambda owner: combine([owner, owner], axis=1, owner_type=owner_type),
        }
        for operation, transform in operations.items():
            original = owner_type(package)
            saved_headers, saved_roles = original.headers, original.roles
            old_frame = original.frame
            before = old_frame.to_numpy(copy=False)
            changed = transform(original)
            state = changed._state
            frame = changed.frame
            result.append(dict(backend=backend, operation=operation, type=type(changed).__name__,
                               frame_type=type(frame).__name__, axis_type=type(frame.columns).__name__,
                               labels=list(state.schema.headers), roles=list(state.schema.roles),
                               ids=list(state.schema.ids), shape=list(frame.shape), dtype=str(state.dtype),
                               identity_origin=[next((i for i, c in enumerate(package.schema) if c.id == column.id),
                                                     "new") for column in state.schema],
                               values=frame.to_numpy().tolist(), excluded=np.argwhere(state.excluded).tolist(),
                               metadata=dict(state.metadata), metadata_types=dict(state.metadata.types),
                               shares_values=bool(np.shares_memory(frame.to_numpy(), before)),
                               shares_mask=bool(np.shares_memory(state.excluded, original._state.excluded)),
                               same_owner=changed is original, same_frame=frame is old_frame,
                               saved_headers=list(saved_headers), saved_roles=list(saved_roles)))
    return result


def edit(owner, field):
    if field == "header":
        owner.headers[1] = "renamed"
    else:
        owner.roles[1] = "z"
    return owner


def insert_owner(owner):
    owner.insert_columns(1, np.ma.array(np.ones((64, 1), dtype=int), mask=True),
                         headers=["new"], roles=["y"])
    return owner


def insert(frame):
    frame.insert(1, "new", 0)
    return frame


def set_axis(frame):
    frame.columns = pd.Index(list(frame.columns))
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from .benchmark import environment, source_hashes

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(environment=environment(), sources=source_hashes(),
                                           native=native_matrix(), owners=owner_matrix()), indent=2))


if __name__ == "__main__":
    main()
