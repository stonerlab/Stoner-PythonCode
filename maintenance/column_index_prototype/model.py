"""Compare index-owned schema with CS3 using the production owner machinery.

The deliberately small native propagation surface is tested separately from
owner operations. Unknown pandas constructors produce an invalid index, never
inventing column identities from displayed labels.
"""

from copy import deepcopy
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from Stoner.core.column_schema import ColumnSchema
from Stoner.core.storage import DataStorage, _copy_metadata
from Stoner.core.storage_owner import DataOwner, _SchemaField, _TableState


def _restore_index(schema, name):
    return ColumnIndex(schema, name=name)


class ColumnIndex(pd.Index):
    """Flat exact pandas labels with immutable scientific schema owned here.

    ``get_loc`` and equality remain pandas implementations. ``find_col`` is the
    explicit Stoner matching API. Supported copies and positional selection
    carry records; all other constructors fail closed at ``schema`` access.
    """

    def __new__(cls, schema=(), *, name=None):
        schema = schema if isinstance(schema, ColumnSchema) else ColumnSchema(schema)
        labels = pd.Index(schema.headers, dtype=object, name=name)
        result = cls._simple_new(labels._values, name=name)
        result._schema = schema
        return result

    @property
    def schema(self):
        schema = getattr(self, "_schema", None)
        if schema is None or tuple(self) != schema.headers:
            raise ValueError("Index schema lost: use an explicit owner operation or a new import")
        return schema

    def find_col(self, selector, *, force_list=False, first=False):
        return self.schema.resolve(selector, force_list=force_list, first=first)

    def with_field(self, field, key, value):
        return type(self)(self.schema.with_field(field, key, value), name=self.name)

    def copy(self, name=None, deep=False):
        # Records are immutable; pandas label buffers are also immutable.
        if getattr(self, "_schema", None) is None:
            return pd.Index(self._values, name=self.name if name is None else name, copy=deep)
        return type(self)(self.schema, name=self.name if name is None else name)

    def _view(self):
        return self.copy()

    def _getitem_slice(self, key):
        return self.take(np.arange(len(self))[key])

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, pd.Index):
            if getattr(self, "_schema", None) is None:
                return pd.Index(result._values, name=self.name)
            positions = np.arange(len(self))[key]
            return type(self)(self.schema.take(positions), name=self.name)
        return result

    def take(self, indices, axis=0, allow_fill=True, fill_value=None, **kwargs):
        # Let pandas validate axes, negative indices, fill rules and keywords.
        result = super().take(indices, axis=axis, allow_fill=allow_fill,
                              fill_value=fill_value, **kwargs)
        if getattr(self, "_schema", None) is None:
            return pd.Index(result._values, name=self.name)
        if allow_fill and fill_value is not None and np.any(np.asarray(indices) == -1):
            return pd.Index(result, dtype=object, name=self.name)
        return type(self)(self.schema.take(np.arange(len(self))[indices]), name=self.name)

    def __reduce__(self):
        if getattr(self, "_schema", None) is None:
            return pd.Index(self._values, name=self.name).__reduce__()
        return _restore_index, (self.schema, self.name)


class _IndexState(_TableState):
    """Reuse CS3 package validation/conversion, without a stored schema field."""

    def __init__(self, values, excluded, metadata, fill_value, dtype):
        self.values, self.excluded = values, excluded
        self.metadata, self.fill_value, self.dtype = metadata, fill_value, dtype

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.validate()

    @property
    def schema(self):
        if not isinstance(self.values.columns, ColumnIndex):
            raise ValueError("Ordinary Index replacement discarded the scientific schema")
        return self.values.columns.schema

    @classmethod
    def adopt(cls, package):
        package.validate()
        frame = package.values.copy(deep=False)
        frame.columns = ColumnIndex(package.schema)
        return cls(frame, package.excluded, package.metadata, package.fill_value, package.dtype)

    def __deepcopy__(self, memo):
        self.as_storage()
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        result.values = deepcopy(self.values, memo)
        result.excluded = deepcopy(self.excluded, memo)
        memo.pop(id(self.values), None)
        memo.pop(id(self.excluded), None)
        result.metadata = _copy_metadata(self.metadata, memo)
        result.fill_value = deepcopy(self.fill_value, memo)
        result.dtype = self.dtype
        return result


class _IndexField(_SchemaField):
    """Saved views resolve the current index, replacing it for every edit."""

    def __setitem__(self, key, value):
        self._owner._check_writable()
        state = self._owner._state
        state.values.columns = state.values.columns.with_field(self._field, key, value)


class IndexOwner(DataOwner):
    """Experimental owner: inherited transactions, masks and metadata, index schema.

    Native structural results are exports. Re-entry requires an explicit external
    mask and metadata plus a valid ColumnIndex; no inference from duplicate text.
    Same-shaped permutations of values alone remain undetectable, as in CS3.
    """

    def _publish_storage(self, package):
        self._state = _IndexState.adopt(package)
        self._native_frame_id = None

    def _commit_pandas(self, frame):
        """Replace CS3's dataclass replacement, which writes a schema field."""
        self.validate()
        original = self._state
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Expected a DataFrame")
        if (not frame.index.identical(original.values.index) or
                not frame.columns.identical(original.values.columns)):
            raise ValueError("Pandas replacement must preserve axes")
        if not frame.dtypes.equals(original.values.dtypes):
            raise ValueError("Pandas replacement must preserve dtypes")
        candidate = _IndexState(frame.copy(deep=True), original.excluded, original.metadata,
                                original.fill_value, original.dtype)
        if candidate.schema != original.schema:
            raise ValueError("Pandas replacement must preserve scientific columns")
        candidate.validate()
        self._state = candidate

    @property
    def headers(self):
        return _IndexField(self, "header")

    @property
    def roles(self):
        return _IndexField(self, "role")

    def select(self, rows=slice(None), columns=slice(None)):
        # Same algorithm as CS3 DataOwner.select; only constructor differs.
        array, schema = self._selection(rows, columns)
        package = DataStorage.from_numpy(array, headers=schema.headers, roles=schema.roles,
                                         metadata=self._state.metadata)
        package.schema = list(schema)
        package.values.columns = schema.ids
        return type(self)(package)

    @classmethod
    def reenter(cls, frame, *, expected_schema, mask, metadata, fill_value, dtype):
        state = _IndexState(frame, mask, metadata, fill_value, dtype)
        if state.schema != expected_schema:
            raise ValueError("Native result disagrees with the explicit transformation schema")
        return cls(state.as_storage())


def fixture(width=8, rows=64):
    """Identical deterministic masked integer package for both ownership models."""
    headers = ["Field (T)", "Signal", "Signal", "Error", "Temperature [K]",
               "literal*", "offset", "zero"]
    headers = (headers + [f"Channel {i:04d}" for i in range(8, width)])[:width]
    roles = (list("xyey....") + ["."] * width)[:width]
    values = np.ma.array(np.arange(rows * width).reshape(rows, width), mask=False)
    if width and rows:
        values.mask[0, -1] = True
    package = DataStorage.from_numpy(values, headers=headers, roles=roles)
    package.metadata["run"] = {"nested": [1, 2]}
    return package


def real_fixture():
    """Load real instrument data once; both owners receive the same package."""
    from Stoner import Data

    data = Data(Path(__file__).resolve().parents[2] / "tests/Stoner/CoreTest.dat")
    data.setas("xy")
    return data.export_storage()


def pickle_roundtrip(value):
    return pickle.loads(pickle.dumps(value))


def combine(owners, *, axis, owner_type):
    """Explicit prototype combination policy, shared by both ownership models.

    Require equal dtype/fill and typed metadata. Row concatenation additionally
    requires identical IDs, roles and headers; plain frames require a fresh
    explicit import. Column concatenation preserves roles and remaps repeated
    IDs in order. Values and external masks are prepared before publication.
    """
    if axis not in (0, 1) or not owners or any(type(owner) is not owner_type for owner in owners):
        raise ValueError("Combine explicit owners on axis zero or one")
    packages = [owner.export_storage() for owner in owners]
    first = packages[0]
    for package in packages[1:]:
        if (package.dtype != first.dtype or package.fill_value != first.fill_value or
                package.metadata != first.metadata or package.metadata.types != first.metadata.types):
            raise ValueError("Conflicting dtype, fill value or typed metadata")
        if axis == 0 and package.schema != first.schema:
            raise ValueError("Row combination requires identical scientific columns")
        if axis == 1 and len(package.values) != len(first.values):
            raise ValueError("Column combination requires equal row counts")
    schema = ColumnSchema(first.schema)
    if axis == 1:
        for package in packages[1:]:
            schema = schema.insert(len(schema), package.schema)
    array = np.ma.concatenate([p.to_numpy() for p in packages], axis=axis)
    result = DataStorage.from_numpy(array, headers=schema.headers, roles=schema.roles,
                                    metadata=first.metadata)
    result.schema = list(schema)
    result.values.columns = schema.ids
    return owner_type(result)
