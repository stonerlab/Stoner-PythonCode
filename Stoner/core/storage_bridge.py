"""Connect Data's public interfaces to its authoritative frame owner."""

from copy import deepcopy

import numpy as np

from .numerical import numerical_result
from .base import TypeHintedDict
from .setas import Setas
from .storage import DataStorage, resolve_columns, _copy_metadata
from .storage_owner import DataOwner




class _ResolvedSetas(Setas):
    """Reuse assignment syntax with the migration's shared column resolver."""

    def find_col(self, col, force_list=False):
        return resolve_columns(self._schema, col, force_list=force_list)

    def __getitem__(self, key):
        if isinstance(key, str) and len(key) == 2 and key[0] == "#" and key[1] in "xyzdefuvw":
            positions = [i for i, role in enumerate(self.to_list()) if role == key[1]]
            return positions[0] if len(positions) == 1 else positions
        return super().__getitem__(key)

    def _get_cols(self, what=None, startx=0, no_guess=False):
        result = super()._get_cols(startx=startx, no_guess=no_guess)
        if self.to_list().count("x") > 1 and result.xerr is not None:
            result.xerr += result.xcol
        if what in ("xcol", "xerr"):
            return result[what]
        if what in ("ycol", "zcol", "ucol", "vcol", "wcol", "yerr", "zerr"):
            return result[what][0]
        if what in ("ycols", "zcols", "ucols", "vcols", "wcols", "yerrs", "zerrs"):
            return result[what[:-1]]
        return result


class _BoundSetas:
    """Parse assignments on a detached Setas before committing complete roles."""

    def __init__(self, data):
        self._data = data

    def _snapshot(self):
        owner = self._data.__dict__["_storage_owner"]
        result = _ResolvedSetas(bless=self._data)
        result._schema = owner._state.schema
        result.shape = owner._state.values.shape
        result.column_headers = list(owner.headers)
        result(list(owner.roles))
        return result

    def __call__(self, *args, **kwargs):
        owner = self._data.__dict__["_storage_owner"]
        if not args and not {key for key in kwargs if key != "_self"}:
            return list(owner.roles)
        owner._check_writable()
        args = tuple(arg.to_list() if isinstance(arg, _BoundSetas) else arg for arg in args)
        draft = self._snapshot()
        result = draft(*args, **kwargs)
        owner.roles[:] = draft.to_list()
        return self if result is draft else result

    def __getitem__(self, key):
        return self._snapshot()[key]

    def __setitem__(self, key, value):
        owner = self._data.__dict__["_storage_owner"]
        owner._check_writable()
        if key is None:
            raise IndexError("A role assignment requires a column selector")
        draft = self._snapshot()
        if isinstance(key, str) and key in "xyzdefuvw.-" and len(key) == 1:
            draft[key] = value
        else:
            positions = resolve_columns(owner._state.schema, key, force_list=True)
            if isinstance(value, str) and len(value) == len(positions) and len(value) > 1:
                value = list(value)
            draft[positions] = value
        owner.roles[:] = draft.to_list()

    def __iter__(self):
        return iter(self.to_list())

    def __len__(self):
        return len(self.to_list())

    def _mutate(self, method, *args, **kwargs):
        owner = self._data._require_storage_owner()
        owner._check_writable()
        draft = self._snapshot()
        args = tuple(arg.clone if isinstance(arg, _BoundSetas) else arg for arg in args)
        result = getattr(draft, method)(*args, **kwargs)
        owner.roles[:] = draft.to_list()
        return self if result is draft else result

    def __getattr__(self, name):
        if name in {"unset", "pop", "popitem", "setdefault"}:
            return lambda *args, **kwargs: self._mutate(name, *args, **kwargs)
        if name in {"get", "keys", "values", "items", "set", "not_set", "ndim", "x", "y", "z", "_cols", "_size", "_unique_headers"}:
            return getattr(self._snapshot(), name)
        if name in self._snapshot().cols:
            return self._snapshot().cols[name]
        raise AttributeError(name)

    def __contains__(self, item):
        return item in self._snapshot()

    def __eq__(self, other):
        return self._snapshot() == (other.to_list() if isinstance(other, _BoundSetas) else other)

    def __delitem__(self, key):
        self._mutate("__delitem__", key)

    def __iadd__(self, other):
        return self._mutate("__iadd__", other)

    def __isub__(self, other):
        return self._mutate("__isub__", other)

    def __add__(self, other):
        return self._snapshot() + other

    def __sub__(self, other):
        return self._snapshot() - other

    @property
    def setas(self):
        """Expose an owner-bound mutable sequence of explicit roles."""
        return self._data._require_storage_owner().roles

    @setas.setter
    def setas(self, value):
        self(value)

    def to_list(self):
        """Return independent explicit roles."""
        return list(self._data.__dict__["_storage_owner"].roles)

    def to_string(self, encode=False):
        """Return the explicit role string."""
        return self._snapshot().to_string(encode=encode)

    def to_dict(self):
        """Return the existing Setas dictionary representation."""
        return self._snapshot().to_dict()

    def update(self, *args, **kwargs):
        """Commit a mapping update only after all assignments validate."""
        owner = self._data.__dict__["_storage_owner"]
        owner._check_writable()
        draft = self._snapshot()
        args = tuple(arg.clone if isinstance(arg, _BoundSetas) else arg for arg in args)
        draft.update(*args, **kwargs)
        owner.roles[:] = draft.to_list()
        return self

    def clear(self):
        """Clear all explicit roles without changing column identity."""
        self._data.__dict__["_storage_owner"].roles[:] = ["."] * len(self)

    def __str__(self):
        return self.to_string()

    @property
    def clone(self):
        """Return a detached Setas object."""
        return self._snapshot()

    @property
    def empty(self):
        """Report whether every explicit role is unset."""
        return self._snapshot().empty

    @property
    def shape(self):
        """Return the current owner shape."""
        return self._data.__dict__["_storage_owner"]._state.values.shape

    @property
    def cols(self):
        """Infer coordinate/error groups from a current snapshot."""
        return self._snapshot().cols

    @property
    def column_headers(self):
        """Return the owner's fixed-width header interface."""
        return self._data.__dict__["_storage_owner"].headers

    @column_headers.setter
    def column_headers(self, value):
        self._data.__dict__["_storage_owner"].headers[:] = value

    def _get_cols(self, *args, **kwargs):
        return self._snapshot()._get_cols(*args, **kwargs)

    def find_col(self, col, force_list=False):
        return resolve_columns(self._data.__dict__["_storage_owner"]._state.schema, col, force_list=force_list)


class StorageBridgeMixin:
    """Expose frame-backed Data construction, schema, snapshots and explicit edits.

    Notes:
        Construction and loading use the frame owner from the outset. NumPy masked array
        snapshots retain the public row/role result interface without owning storage.
    """

    @classmethod
    def from_storage(cls, package):
        """Construct a Data instance owning a detached validated storage package.

        Args:
            package (DataStorage):
                Values, masks, stable identities and typed metadata to import.

        Returns:
            Data:
                A frame-backed instance with independent validated state.
        """
        owner = DataOwner(package)
        result = cls()
        result.__dict__["_storage_owner"] = owner
        result.__dict__.pop("_data", None)
        result.__dict__.pop("_metadata", None)
        return result

    @classmethod
    def from_pandas(cls, frame, *, mask=None, setas=None, metadata=None, dtype=None, index="require_range"):
        """Construct frame-backed Data using explicit numeric and row-index policies.

        Args:
            frame (pandas.DataFrame):
                Native string-labelled or legacy header/role-labelled values.

        Keyword Arguments:
            mask (numpy.ndarray or None):
                Boolean exclusions, defaulting to all false.
            setas (sequence or None):
                Explicit role assignments overriding imported roles.
            metadata (TypeHintedDict or None):
                Typed state to copy independently.
            dtype (numpy.dtype or None):
                Explicit common numeric dtype for mixed numeric columns.
            index (str):
                Require positional rows by default; use ``discard`` to drop labels.

        Returns:
            Data:
                An independently owned frame-backed instance.
        """
        return cls.from_storage(DataStorage.from_pandas(frame, mask=mask, setas=setas, metadata=metadata,
                                                       dtype=dtype, index=index))

    def export_storage(self):
        """Return a detached package from the sole storage owner."""
        return self._require_storage_owner().export_storage()

    def to_numpy(self, *, masked=True, dtype=None):
        """Return detached numerical values, preserving exclusions by default."""
        return self._require_storage_owner().to_numpy(masked=masked, dtype=dtype)

    def edit_numpy(self):
        """Return the frame owner's fixed-shape/dtype numerical editing context."""
        return self._require_storage_owner().edit_numpy()

    def edit_pandas(self):
        """Return a raw-value editing context retaining exclusions and schema."""
        return self._require_storage_owner().edit_pandas()

    def _require_storage_owner(self):
        """Reject editing APIs until an instance has an authoritative frame owner."""
        owner = self.__dict__.get("_storage_owner")
        if owner is None:
            raise RuntimeError("Use Data.from_storage(data.export_storage()) before frame editing")
        return owner

    def _storage_array(self):
        """Return the detached read-only NumPy result."""
        owner = self._require_storage_owner()
        result = numerical_result(owner.to_numpy(), setas=self.setas.clone)
        result.column_headers = list(owner.headers)
        result.setas = list(owner.roles)
        result._setas = self.setas.clone
        result.fill_value = owner._state.fill_value
        result.setflags(write=False)
        result._mask.setflags(write=False)
        return result

    def _storage_slice(self, key):
        """Wrap an indexed numerical snapshot with its selected headers and roles."""
        owner = self._require_storage_owner()
        value = owner[key]
        if value is np.ma.masked or not isinstance(value, np.ndarray):
            return value
        rows, columns = key if isinstance(key, tuple) else (key, slice(None))
        positions = resolve_columns(owner._state.schema, columns, force_list=True)
        isrow = isinstance(rows, (int, np.integer))
        result = numerical_result(value.copy(), isrow=isrow)
        schema = [owner._state.schema[i] for i in positions]
        roles = _ResolvedSetas(row=isrow)
        roles._schema = schema
        roles.shape = result.shape
        roles.column_headers = [column.header for column in schema]
        roles([column.role for column in schema])
        result = numerical_result(value, setas=roles, isrow=isrow)
        if isrow:
            result.i = int(rows) % owner._state.values.shape[0]
        result.fill_value = owner._state.fill_value
        result.setflags(write=False)
        result._mask.setflags(write=False)
        return result

    @property
    def data(self):
        """Return a detached read-only NumPy masked array."""
        return self._storage_array()

    @data.setter
    def data(self, value):
        owner = self._require_storage_owner()
        owner._check_writable()
        array = np.ma.asarray(value)
        if array.ndim == 1:
            array = array[:, None]
        elif array.ndim == 0:
            array = array.reshape(1, 1)
        owner.replace(array)

    @property
    def setas(self):
        """Return the owner-bound assignment parser."""
        return _BoundSetas(self)

    @setas.setter
    def setas(self, value):
        if isinstance(value, (_BoundSetas, Setas)):
            value = value.to_list()
        self.setas(value)

    @property
    def column_headers(self):
        """Return the authoritative displayed-header interface."""
        return self._require_storage_owner().headers

    @column_headers.setter
    def column_headers(self, value):
        owner = self._require_storage_owner()
        value = [value] if isinstance(value, str) else list(value)
        if owner._state.values.shape == (0, 0):
            owner.replace(np.empty((0, len(value)), dtype=owner._state.dtype))
        owner.headers[:] = value

    @property
    def metadata(self):
        """Return the guarded owner metadata mapping."""
        return self._require_storage_owner().metadata

    @metadata.setter
    def metadata(self, value):
        owner = self._require_storage_owner()
        owner._check_writable()
        if hasattr(value, "_owner"):
            value = value._owner._state.metadata
        if not isinstance(value, TypeHintedDict):
            value = TypeHintedDict(value)
        owner._state.metadata = _copy_metadata(value)

    @property
    def mask(self):
        """Return the owner-aware exclusion interface."""
        return self._require_storage_owner().mask

    @mask.setter
    def mask(self, value):
        if callable(value):
            owner = self._require_storage_owner()
            owner._check_writable()
            value = np.asarray([value(row) for row in self.rows()], dtype=bool)
            if value.ndim == 1:
                value = value[:, None]
        self._require_storage_owner().mask[:] = value

    @property
    def column_ids(self):
        """Return stable column identities from the frame owner."""
        return self._require_storage_owner().column_ids

    @property
    def shape(self):
        """Read dimensions without materialising a numerical snapshot."""
        return self._require_storage_owner()._state.values.shape

    @property
    def dtype(self):
        """Read the explicit homogeneous dtype, including empty-column tables."""
        return self._require_storage_owner()._state.dtype

    @property
    def clone(self):
        """Clone without sharing numerical or metadata state."""
        return deepcopy(self)

    def __copy__(self):
        """Copy independently, as for clone and deepcopy."""
        return deepcopy(self)
