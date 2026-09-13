"""Provide validated, detached storage interchange for the Data migration.

These primitives support :class:`Stoner.Data` storage. Values and
exclusions remain separate so masked integers survive a lossless round trip.
"""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import re
from uuid import UUID, uuid4
import warnings

import numpy as np
import pandas as pd

from .base import TypeHintedDict

__all__ = ["Column", "DataStorage", "resolve_columns"]


@dataclass(frozen=True)
class Column:
    """Describe one stable column independently of its position.

    Attributes:
        id (str):
            Canonical UUID string, unique within a schema.
        header (str):
            Displayed name; duplicates are permitted.
        role (str):
            One of ``xyzdefuvw.``; role changes do not change identity.
    """

    id: str
    header: str
    role: str = "."

    def __post_init__(self):
        """Reject malformed identity, header and role state."""
        if not isinstance(self.id, str) or str(UUID(self.id)) != self.id:
            raise ValueError("Column identity must be a canonical UUID string")
        if not isinstance(self.header, str):
            raise TypeError("Column header must be a string")
        if not isinstance(self.role, str) or self.role not in tuple("xyzdefuvw."):
            raise ValueError("Column role must be one character from xyzdefuvw.")


def resolve_columns(schema, selector, *, force_list=False):
    """Resolve a Stoner selector against an ordered column schema.

    Args:
        schema (sequence of Column):
            Ordered displayed headers and identities.
        selector (int, str, re.Pattern, slice or sequence):
            Integers use ordinary negative bounds. Exact names precede patterns;
            string patterns select the first match, compiled patterns every match.

    Keyword Arguments:
        force_list (bool):
            Return a list even for a single positional or string result.

    Returns:
        int or list of int:
            Positions, retaining duplicate headers and repeated selections.

    Raises:
        IndexError:
            If an integer position is out of bounds.
        KeyError:
            If no header matches.
        re.error:
            If a string pattern is invalid.
        TypeError:
            If the selector type is unsupported, including Boolean indices.
    """
    count = len(schema)
    if isinstance(selector, (bool, np.bool_)):
        raise TypeError("Boolean column selectors are not integer positions")
    if isinstance(selector, (int, np.integer)):
        if not -count <= selector < count:
            raise IndexError(selector)
        result = int(selector) % count
    elif isinstance(selector, slice):
        result = list(range(count))[selector]
    elif isinstance(selector, str):
        matches = [i for i, column in enumerate(schema) if column.header == selector]
        result = matches[0] if matches else resolve_columns(schema, re.compile(selector))[0]
    elif isinstance(selector, re.Pattern):
        result = [i for i, column in enumerate(schema) if selector.search(column.header)]
        if not result:
            raise KeyError(selector.pattern)
    elif isinstance(selector, (bytes, bytearray)):
        raise TypeError("Column names must be text, not bytes")
    elif isinstance(selector, Sequence) or (isinstance(selector, np.ndarray) and selector.ndim == 1):
        result = []
        for item in selector:
            result.extend(resolve_columns(schema, item, force_list=True))
    else:
        raise TypeError(f"Unsupported column selector: {type(selector).__name__}")
    return [result] if force_list and isinstance(result, int) else result


def _copy_metadata(source, memo=None):
    """Copy existing type hints and nested values without reparsing strings."""
    if not isinstance(source, TypeHintedDict):
        raise TypeError("Storage metadata must be a TypeHintedDict")
    memo = {} if memo is None else memo
    result = TypeHintedDict()
    memo[id(source)] = result
    for key, value in source.items():
        try:
            result.types[key] = deepcopy(source.type(key), memo)
            super(TypeHintedDict, result).__setitem__(key, deepcopy(value, memo))
        except Exception as error:
            raise TypeError(f"Cannot copy metadata key {key!r}") from error
    return result


def _numeric_dtype(dtype):
    """Require a homogeneous NumPy numeric dtype, excluding extensions."""
    if not isinstance(dtype, np.dtype) or dtype.kind not in "biufc":
        raise TypeError("Storage requires a NumPy Boolean, integer, float or complex dtype")
    return dtype


@dataclass
class DataStorage:
    """Carry editable, detached version-one Data interchange state.

    Attributes:
        values (pandas.DataFrame):
            Raw numeric values, UUID-labelled columns and a positional RangeIndex.
        excluded (numpy.ndarray):
            Same-shaped Boolean mask; true means excluded, not missing.
        schema (list of Column):
            Ordered column identities, displayed headers and roles.
        metadata (TypeHintedDict):
            Typed user metadata, including explicit hints.
        fill_value (scalar):
            Dtype-compatible sentinel used only for filled representations.
        dtype (numpy.dtype):
            Common dtype, retained even when there are no columns.
        version (int):
            Package version, currently one.

    Notes:
        Direct record construction permits editing and does not validate or copy
        caller-owned fields. Use the factories or :meth:`copy` at owner boundaries.
        These validate and detach. This is in-memory interchange, not a file codec.
    """

    values: pd.DataFrame
    excluded: np.ndarray
    schema: list
    metadata: TypeHintedDict
    fill_value: object
    dtype: np.dtype
    version: int = 1

    def validate(self):
        """Reject inconsistent package state without changing any field.

        Raises:
            TypeError:
                If values, dtypes, masks or metadata have unsupported types.
            ValueError:
                If version, shape, schema, axes or fill value is inconsistent.
        """
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported storage version")
        if not isinstance(self.values, pd.DataFrame):
            raise TypeError("Storage values must be a DataFrame")
        _numeric_dtype(self.dtype)
        if not isinstance(self.schema, (list, tuple)) or any(not isinstance(col, Column) for col in self.schema):
            raise TypeError("Schema must be an ordered sequence of Column records")
        identities = [col.id for col in self.schema]
        if len(set(identities)) != len(identities) or list(self.values.columns) != identities:
            raise ValueError("Columns must match unique ordered schema identities")
        index = self.values.index
        if not isinstance(index, pd.RangeIndex) or not index.equals(pd.RangeIndex(len(self.values))):
            raise ValueError("Storage rows must use RangeIndex(0, nrows)")
        for dtype in self.values.dtypes:
            if _numeric_dtype(dtype) != self.dtype:
                raise ValueError("Frame columns must share the declared dtype")
        if not isinstance(self.excluded, np.ndarray) or self.excluded.dtype != np.dtype(bool):
            raise TypeError("Exclusions must be a Boolean ndarray")
        if self.excluded.shape != self.values.shape:
            raise ValueError("Exclusion shape differs from values")
        if not isinstance(self.metadata, TypeHintedDict):
            raise TypeError("Storage metadata must be a TypeHintedDict")
        if np.ndim(self.fill_value) != 0:
            raise ValueError("Fill value must be scalar")
        if np.asarray(self.fill_value).dtype.kind not in "biufc":
            raise ValueError("Fill value must be numeric")
        try:
            converted = np.ma.array(np.empty(0, dtype=self.dtype), fill_value=self.fill_value).fill_value
            if not (converted == self.fill_value or (np.isnan(converted) and np.isnan(self.fill_value))):
                raise ValueError("Fill value cannot be represented exactly")
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Fill value is incompatible with the dtype") from error

    @classmethod
    def from_numpy(cls, values, *, headers=None, roles=None, metadata=None):
        """Create independent state from a two-dimensional numeric array.

        Args:
            values (numpy.ndarray or numpy.ma.MaskedArray):
                Values and optional exclusions; masked raw values remain recoverable.

        Keyword Arguments:
            headers (sequence of str or None):
                Display names, defaulting to ``Column n``.
            roles (sequence of str or None):
                One role per column, defaulting to unset dots.
            metadata (TypeHintedDict or None):
                Independently copied typed state, defaulting to empty.

        Returns:
            DataStorage:
                A validated detached package with fresh column identities.
        """
        array = np.ma.asarray(values)
        _numeric_dtype(array.dtype)
        if array.ndim != 2:
            raise ValueError("Storage values must be two-dimensional")
        width = array.shape[1]
        headers = list(headers) if headers is not None else [f"Column {i}" for i in range(width)]
        roles = list(roles) if roles is not None else ["."] * width
        if len(headers) != width or len(roles) != width:
            raise ValueError("Schema width differs from values")
        schema = [Column(str(uuid4()), header, role) for header, role in zip(headers, roles)]
        # NumPy may store its generic integer default as int64 even on an int8
        # array. Import the sentinel that NumPy would actually use when filling.
        fill_value = np.ma.array(np.empty(0, dtype=array.dtype), fill_value=array.fill_value).fill_value
        result = cls(pd.DataFrame(array.data, columns=[col.id for col in schema], copy=True),
                     np.ma.getmaskarray(array).copy(), schema,
                     _copy_metadata(metadata) if metadata is not None else TypeHintedDict(),
                     fill_value, array.dtype)
        result.validate()
        return result

    def copy(self):
        """Validate and deep-copy all state, retaining IDs and metadata type hints."""
        self.validate()
        return deepcopy(self)

    def __deepcopy__(self, memo):
        """Preserve aliases within metadata while detaching it from the source."""
        self.validate()
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        result.values = self.values.copy(deep=True)
        result.excluded = self.excluded.copy()
        result.schema = list(self.schema)
        result.metadata = _copy_metadata(self.metadata, memo)
        result.fill_value = deepcopy(self.fill_value, memo)
        result.dtype, result.version = self.dtype, self.version
        return result

    def to_numpy(self, *, masked=True, dtype=None):
        """Export independent raw values or a recoverable masked representation.

        Keyword Arguments:
            masked (bool):
                Include exclusions and fill value by default; false exposes raw values.
            dtype (numpy.dtype or None):
                Optional numeric conversion of the export only.

        Returns:
            numpy.ndarray or numpy.ma.MaskedArray:
                Independent numerical data. A masked dtype conversion must also
                accommodate the stored fill value; use a raw export otherwise.
        """
        self.validate()
        target = self.dtype if dtype is None else _numeric_dtype(np.dtype(dtype))
        values = self.values.to_numpy(dtype=target, copy=True)
        if not masked:
            return values
        return np.ma.array(values, mask=self.excluded.copy(), fill_value=self.fill_value, copy=False)

    @classmethod
    def from_pandas(cls, frame, *, mask=None, setas=None, metadata=None, dtype=None, index="require_range"):
        """Import native or legacy columns with explicit positional row policy.

        Args:
            frame (pandas.DataFrame):
                Numeric frame with string headers or two-level header/role columns.

        Keyword Arguments:
            mask (numpy.ndarray or None):
                Boolean exclusions, defaulting to all false even at NaN values.
            setas (sequence of str or None):
                Explicit roles overriding imported roles.
            metadata (TypeHintedDict or None):
                Typed state to copy; backend attrs are not imported.
            dtype (numpy.dtype or None):
                Explicit common numeric dtype for mixed numeric input.
            index (str):
                ``require_range`` by default; ``discard`` explicitly drops row labels.

        Returns:
            DataStorage:
                A detached validated package with newly allocated IDs.
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Expected a DataFrame")
        if index not in ("require_range", "discard"):
            raise ValueError("Unknown row-index policy")
        if index == "require_range" and not frame.index.equals(pd.RangeIndex(len(frame))):
            raise ValueError("Use index='discard' to discard non-positional row labels")
        dtypes = [_numeric_dtype(item) for item in frame.dtypes]
        if dtype is None and len(set(dtypes)) > 1:
            raise TypeError("Mixed numeric columns require an explicit dtype")
        target = _numeric_dtype(np.dtype(dtype)) if dtype is not None else (dtypes[0] if dtypes else np.dtype(float))
        if isinstance(frame.columns, pd.MultiIndex):
            if frame.columns.nlevels != 2:
                raise ValueError("Expected two-level header/role columns")
            headers, roles = (list(frame.columns.get_level_values(i)) for i in range(2))
        else:
            headers, roles = list(frame.columns), None
        if mask is not None:
            if not isinstance(mask, np.ndarray) or mask.dtype != bool:
                raise TypeError("Exclusions must be a Boolean ndarray")
            if mask.shape != frame.shape:
                raise ValueError("Exclusion shape differs from frame")
        values = np.ma.array(frame.to_numpy(dtype=target, copy=True), mask=False if mask is None else mask)
        return cls.from_numpy(values, headers=headers, roles=setas if setas is not None else roles, metadata=metadata)

    def to_pandas(self, *, format="legacy", masked="nan"):
        """Export a lossy detached native frame, warning about exclusions/metadata.

        Keyword Arguments:
            format (str):
                ``legacy`` exports Headers/Setas MultiIndex; ``plain`` exports names.
            masked (str):
                ``nan`` replaces excluded exports with NaN; ``raw`` exposes hidden values.

        Returns:
            pandas.DataFrame:
                Values and displayed labels, without lossless side state.
        """
        if format not in ("legacy", "plain") or masked not in ("nan", "raw"):
            raise ValueError("Unknown pandas export policy")
        values = self.to_numpy(masked=False)
        if self.excluded.any() or self.metadata:
            warnings.warn("Pandas export omits exclusions or typed metadata; retain DataStorage for lossless use",
                          UserWarning, stacklevel=2)
        if masked == "nan" and self.excluded.any():
            if values.dtype.kind not in "fc":
                values = values.astype(np.result_type(values.dtype, float))
            values[self.excluded] = np.nan
        headers = [col.header for col in self.schema]
        columns = pd.MultiIndex.from_arrays([headers, [col.role for col in self.schema]],
                                           names=["Headers", "Setas"]) if format == "legacy" else headers
        return pd.DataFrame(values, columns=columns, copy=True)
