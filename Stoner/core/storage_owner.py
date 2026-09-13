"""Own Data's validated pandas values, exclusions, schema and typed metadata."""

from collections.abc import MutableMapping, MutableSequence
from contextlib import contextmanager
from dataclasses import replace
from uuid import uuid4

import numpy as np

from .storage import Column, DataStorage, resolve_columns, _copy_metadata
from .base import TypeHintedDict


class _Metadata(MutableMapping):
    """Resolve typed metadata on each access and guard edits during transactions."""

    def __init__(self, owner):
        self._owner = owner

    def __getitem__(self, key):
        return self._owner._state.metadata[key]

    def __setitem__(self, key, value):
        self._owner._check_writable()
        prepared = TypeHintedDict()
        prepared[key] = value
        metadata = self._owner._state.metadata
        for name, item in prepared.items():
            super(TypeHintedDict, metadata).__setitem__(name, item)
            metadata.types[name] = prepared.type(name)

    def __delitem__(self, key):
        self._owner._check_writable()
        del self._owner._state.metadata[key]

    def __iter__(self):
        return iter(self._owner._state.metadata)

    def __len__(self):
        return len(self._owner._state.metadata)

    def type(self, key):
        """Return the existing hint without inferring it again."""
        return self._owner._state.metadata.type(key)

    def copy(self):
        """Copy typed values without repeating string inference."""
        return _copy_metadata(self._owner._state.metadata)

    def __getattr__(self, name):
        if name in {"export", "export_all", "string_to_type"}:
            return getattr(self._owner._state.metadata, name)
        if name in {"types", "_typehints"}:
            return getattr(self.copy(), name)
        if name in {"import_key", "import_all"}:
            def import_metadata(*args, **kwargs):
                self._owner._check_writable()
                draft = self.copy()
                result = getattr(draft, name)(*args, **kwargs)
                self._owner._state.metadata = draft
                return result
            return import_metadata
        raise AttributeError(name)

    def __xor__(self, other):
        return self._owner._state.metadata ^ (other.copy() if isinstance(other, _Metadata) else other)

    def __eq__(self, other):
        return self._owner._state.metadata == (other.copy() if isinstance(other, _Metadata) else other)


class _SchemaField(MutableSequence):
    """Expose fixed-width header or role updates without changing identities."""

    def __init__(self, owner, field):
        self._owner, self._field = owner, field

    def __len__(self):
        return len(self._owner._state.schema)

    def __getitem__(self, key):
        return [getattr(column, self._field) for column in self._owner._state.schema][key]

    def __eq__(self, other):
        return list(self) == list(other)

    def __add__(self, other):
        return list(self) + list(other)

    def __repr__(self):
        return repr(list(self))

    def __copy__(self):
        return list(self)

    def __deepcopy__(self, memo):
        return list(self)

    def copy(self):
        """Return an ordinary independent list."""
        return list(self)

    def __setitem__(self, key, value):
        self._owner._check_writable()
        current = list(self)
        current[key] = value
        if len(current) != len(self):
            raise ValueError("Schema field edits cannot change column count")
        # Construct all immutable records before committing any change.
        schema = [replace(column, **{self._field: item})
                  for column, item in zip(self._owner._state.schema, current)]
        self._owner._state.schema = schema

    def __delitem__(self, key):
        raise TypeError("Delete columns through structural owner operations")

    def insert(self, index, value):
        """Reject schema-only insertion, which would separate labels from values."""
        raise TypeError("Insert columns through structural owner operations")


class _Mask:
    """Expose indexed exclusion edits without handing out writable owner arrays."""

    def __init__(self, owner):
        self._owner = owner

    def __getitem__(self, key):
        return self._owner._state.excluded[key].copy()

    def __setitem__(self, key, value):
        self._owner._check_writable()
        mask = self._owner._state.excluded.copy()
        mask[key] = value
        self._owner._state.excluded = mask

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("A mask export always requires a copy")
        return np.array(self._owner._state.excluded, dtype=dtype, copy=True)

    def __getattr__(self, name):
        if name in {"shape", "dtype", "ndim", "size", "any", "all", "sum", "copy", "ravel"}:
            return getattr(np.asarray(self), name)
        raise AttributeError(name)

    def __invert__(self):
        return ~np.asarray(self)


class DataOwner:
    """Maintain one authoritative DataFrame and explicit exclusion/schema state.

    Args:
        package (DataStorage):
            Initial validated state, copied independently of the caller.

    Attributes:
        headers (mutable sequence):
            Fixed-width displayed header interface; duplicates are permitted.
        roles (mutable sequence):
            Fixed-width role interface. Full Setas syntax belongs to Data integration.
        metadata (mutable mapping):
            Owner-aware typed metadata interface.
        mask (object):
            Owner-aware indexed Boolean exclusion interface.
        column_ids (tuple of str):
            Stable identities in current order.

    Notes:
        This internal primitive powers the public Data class.
        Backend editing is cell-only, with fixed axes and dtypes. Nested mutable
        metadata values remain outside numerical editing transactions.
    """

    def __init__(self, package):
        if not isinstance(package, DataStorage):
            raise TypeError("DataOwner requires a DataStorage package")
        self._state = package.copy()
        self._editing = False

    def _check_writable(self):
        """Reject writes that could be overwritten by an active edit draft."""
        if self._editing:
            raise RuntimeError("The owner already has an active editing context")

    @property
    def headers(self):
        """Return a durable interface to displayed column headers."""
        return _SchemaField(self, "header")

    @property
    def roles(self):
        """Return a durable interface to explicit column roles."""
        return _SchemaField(self, "role")

    @property
    def metadata(self):
        """Return an interface that guards even saved metadata references."""
        return _Metadata(self)

    @property
    def mask(self):
        """Return an interface to per-cell exclusions."""
        return _Mask(self)

    @property
    def column_ids(self):
        """Return immutable column identities in positional order."""
        return tuple(column.id for column in self._state.schema)

    def export_storage(self):
        """Return independent values, exclusions, schema and typed metadata."""
        return self._state.copy()

    def to_numpy(self, *, masked=True, dtype=None):
        """Return a detached numerical conversion with optional masks and dtype."""
        return self._state.to_numpy(masked=masked, dtype=dtype)

    def _index(self, key):
        """Resolve the column part of a two-axis key before NumPy indexing."""
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Expected row and column selectors")
            columns = key[1] if isinstance(key[1], slice) else resolve_columns(self._state.schema, key[1])
            return key[0], columns
        return key

    def __getitem__(self, key):
        key = self._index(key)
        values = self._state.values.to_numpy(dtype=self._state.dtype, copy=False)[key]
        mask = self._state.excluded[key]
        if np.ndim(values) == 0:
            return np.ma.masked if mask else values
        result = np.ma.array(values, mask=mask, fill_value=self._state.fill_value, copy=True)
        if isinstance(result, np.ma.MaskedArray):
            result.setflags(write=False)
            result._mask.setflags(write=False)
        return result

    def __setitem__(self, key, value):
        self._check_writable()
        key = self._index(key)
        if value is np.ma.masked:
            self.mask[key] = True
            return
        incoming = np.ma.asarray(value)
        if incoming.dtype.kind not in "biufc":
            raise TypeError("Numeric assignment required")
        dtype = np.result_type(self._state.dtype, incoming.dtype)
        # Integer-to-float promotion can lose existing large integers, particularly
        # when mixing int64 and uint64. Reject such implicit loss before mutation.
        self._check_integer_precision(dtype, incoming)
        draft = self.to_numpy(dtype=dtype)
        draft[key] = incoming
        self._commit(draft)

    def _check_integer_precision(self, dtype, incoming):
        """Reject implicit promotion that cannot preserve raw integer values."""
        if dtype.kind not in "fc":
            return
        for array in (self._state.values.to_numpy(dtype=self._state.dtype), incoming.data):
            if array.dtype.kind not in "iu":
                continue
            # Object comparison avoids converting the original integers to float
            # during the check itself, including uint64 values above 2**63.
            if not np.array_equal(array.astype(object), array.astype(dtype).astype(object)):
                raise ValueError("Promotion would lose integer precision; convert the data explicitly")

    def _commit(self, draft):
        """Validate the draft, retaining stable IDs and current metadata ownership."""
        candidate = DataStorage.from_numpy(draft, headers=list(self.headers), roles=list(self.roles))
        candidate.schema = list(self._state.schema)
        candidate.values.columns = self.column_ids
        candidate.validate()
        # Metadata is outside the numerical transaction. Retain the authoritative
        # dictionary, including edits to nested values while a draft was open.
        candidate.metadata = self._state.metadata
        self._state = candidate

    def replace(self, array, schema=None):
        """Atomically replace numerical state, preserving positional IDs by default.

        Structural callers supply the complete schema so identities follow columns.
        Plain whole-array replacement retains overlapping positional columns and
        allocates identities only for additional columns.
        """
        self._check_writable()
        array = np.ma.asarray(array)
        if array.ndim != 2:
            raise ValueError("Data requires a two-dimensional array")
        if schema is None:
            schema = list(self._state.schema[:array.shape[1]])
            schema.extend(Column(str(uuid4()), f"Column_{i}")
                          for i in range(len(schema), array.shape[1]))
        candidate = DataStorage.from_numpy(array, headers=[c.header for c in schema],
                                           roles=[c.role for c in schema])
        candidate.schema = list(schema)
        candidate.values.columns = [c.id for c in schema]
        candidate.validate()
        candidate.metadata = self._state.metadata
        self._state = candidate

    def _selection(self, rows, columns):
        """Select rectangular numerical state and remap repeated column IDs."""
        positions = resolve_columns(self._state.schema, columns, force_list=True)
        row_positions = np.atleast_1d(np.arange(self._state.values.shape[0])[rows])
        array = self.to_numpy()[np.ix_(row_positions, positions)]
        schema, seen = [], set()
        for index in positions:
            column = self._state.schema[index]
            if column.id in seen:
                column = replace(column, id=str(uuid4()))
            seen.add(column.id)
            schema.append(column)
        return array, schema

    def select(self, rows=slice(None), columns=slice(None)):
        """Return independent rectangular state, including copied typed metadata."""
        array, schema = self._selection(rows, columns)
        package = DataStorage.from_numpy(array, headers=[c.header for c in schema],
                                         roles=[c.role for c in schema], metadata=self._state.metadata)
        package.schema = schema
        package.values.columns = [c.id for c in schema]
        return DataOwner(package)

    def take(self, rows=slice(None), columns=slice(None)):
        """Select rows/columns in place, keeping masks and identities together."""
        self._check_writable()
        array, schema = self._selection(rows, columns)
        self.replace(array, schema)

    def insert_rows(self, index, values):
        """Insert positional rows without losing exclusions or integer precision."""
        self._check_writable()
        incoming = np.ma.atleast_2d(values)
        current = self.to_numpy()
        if incoming.shape[1] != current.shape[1]:
            raise ValueError("Inserted rows must match the column count")
        dtype = np.result_type(current.dtype, incoming.dtype)
        self._check_integer_precision(dtype, incoming)
        raw = np.insert(current.data.astype(dtype), index, incoming.data, axis=0)
        mask = np.insert(np.ma.getmaskarray(current), index, np.ma.getmaskarray(incoming), axis=0)
        self.replace(np.ma.array(raw, mask=mask, fill_value=current.fill_value))

    def insert_columns(self, index, values, headers=None, roles=None, overwrite=False, schema=None, mask_padding=False):
        """Insert or replace columns, validating the entire result before commit."""
        self._check_writable()
        incoming = np.ma.asarray(values)
        if incoming.ndim == 1:
            incoming = incoming[:, None]
        if incoming.ndim != 2:
            raise ValueError("Inserted columns must be one- or two-dimensional")
        current = self.to_numpy()
        width = incoming.shape[1]
        if not 0 <= index <= current.shape[1]:
            raise IndexError(index)
        dtype = np.result_type(current.dtype, incoming.dtype)
        self._check_integer_precision(dtype, incoming)
        height = max(current.shape[0], incoming.shape[0])
        def padded(array):
            result = np.ma.zeros((height, array.shape[1]), dtype=dtype)
            if mask_padding:
                result.mask = True
            result[:len(array)] = array
            return result
        old, new = padded(current), padded(incoming)
        if schema is None:
            headers = [f"Column_{index + i}" for i in range(width)] if headers is None else list(headers)
            roles = ["."] * width if roles is None else list(roles)
            if len(headers) != width or len(roles) != width:
                raise ValueError("Headers and roles must match inserted columns")
            schema = [Column(self.column_ids[index+i] if overwrite and index+i < len(self.column_ids)
                             else str(uuid4()), header, role)
                      for i, (header, role) in enumerate(zip(headers, roles))]
        stop = index + width if overwrite else index
        final_schema = list(self._state.schema[:index]) + list(schema) + list(self._state.schema[stop:])
        seen = set()
        for i, column in enumerate(final_schema):
            if column.id in seen:
                final_schema[i] = replace(column, id=str(uuid4()))
            seen.add(final_schema[i].id)
        array = np.ma.concatenate((old[:, :index], new, old[:, stop:]), axis=1)
        array.fill_value = current.fill_value
        self.replace(array, final_schema)

    @contextmanager
    def edit_numpy(self):
        """Yield a detached fixed-shape/dtype MaskedArray and commit on normal exit.

        Yields:
            numpy.ma.MaskedArray:
                Editable raw values, exclusions and fill value. Exceptions roll back;
                retained drafts cannot change the owner after exit.

        Raises:
            RuntimeError:
                If an owner transaction is already active.
            ValueError:
                If the draft changes shape or dtype.
        """
        self._check_writable()
        draft = self.to_numpy()
        shape, dtype = draft.shape, draft.dtype
        self._editing = True
        try:
            yield draft
            if draft.shape != shape or draft.dtype != dtype:
                raise ValueError("Numerical editing must preserve shape and dtype")
            self._commit(draft)
        finally:
            self._editing = False

    @contextmanager
    def edit_pandas(self):
        """Yield a detached raw-value frame while retaining masks and schema.

        Yields:
            pandas.DataFrame:
                ID-labelled draft supporting positional cell edits. Axes and dtypes
                must remain fixed; structural transforms require package interchange.

        Notes:
            Do not sort or permute values alone: pandas cannot transform the external
            exclusion mask. The draft has no implicit access to owner metadata.
        """
        self._check_writable()
        original = self._state.values
        draft = original.copy(deep=True)
        self._editing = True
        try:
            yield draft
            if not draft.index.identical(original.index) or not draft.columns.identical(original.columns):
                raise ValueError("Pandas editing must preserve axes")
            if not draft.dtypes.equals(original.dtypes):
                raise ValueError("Pandas editing must preserve dtypes")
            self._state.values = draft.copy(deep=True)
        finally:
            self._editing = False
