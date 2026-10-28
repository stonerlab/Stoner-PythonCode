"""Own ordered column descriptions and share pure scientific role operations."""

from collections import namedtuple
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from uuid import uuid4

import numpy as np


_Header = namedtuple("_Header", "header")
_ROLES = "xyzdefuvw"
_FIELDS = {"y": "ycol", "z": "zcol", "e": "yerr", "f": "zerr",
           "u": "ucol", "v": "vcol", "w": "wcol"}
_DEFAULT_ROLES = {2: "xy", 3: "xye", 4: "xdye", 5: "xyuvw", 6: "xyzuvw"}


@dataclass(frozen=True, eq=False)
class ColumnSchema(Sequence):
    """Keep one immutable ordered collection of column identities, headers and roles.

    Args:
        columns (iterable of Column):
            Validated records with unique stable identities. Displayed headers
            may repeat. Structural selection remaps additional identity occurrences.
    """

    _columns: tuple

    def __init__(self, columns=()):
        from .storage import Column

        records = tuple(columns)
        if any(not isinstance(column, Column) for column in records):
            raise TypeError("Schema must contain Column records")
        if len({column.id for column in records}) != len(records):
            raise ValueError("Column identities must be unique")
        object.__setattr__(self, "_columns", records)

    def __len__(self):
        return len(self._columns)

    def __getitem__(self, key):
        return self._columns[key]

    def __eq__(self, other):
        return isinstance(other, Sequence) and tuple(self) == tuple(other)

    def __deepcopy__(self, memo):
        """Share immutable records without copying any owner or numerical buffers."""
        memo[id(self)] = self
        return self

    @property
    def headers(self):
        """Return displayed names in positional order."""
        return tuple(column.header for column in self)

    @property
    def roles(self):
        """Return explicit roles without inferred defaults."""
        return tuple(column.role for column in self)

    @property
    def ids(self):
        """Return stable identities in positional order."""
        return tuple(column.id for column in self)

    def resolve(self, selector, *, force_list=False, first=False):
        """Apply the unchanged CS2 matching and cardinality rules."""
        from .storage import resolve_columns

        return resolve_columns(self, selector, force_list=force_list, first=first)

    def with_field(self, field, key, value):
        """Prepare a complete fixed-width header or role edit without mutation."""
        if field not in ("header", "role"):
            raise ValueError("Only headers and roles can be edited")
        values = list(self.headers if field == "header" else self.roles)
        values[key] = value
        if len(values) != len(self):
            raise ValueError("Schema field edits cannot change column count")
        return type(self)(replace(column, **{field: item}) for column, item in zip(self, values))

    def take(self, positions):
        """Select records, allocating fresh IDs for repeated occurrences."""
        return type(self)(_unique_records(self[position] for position in positions))

    def resized(self, width):
        """Preserve overlapping records and describe newly added numerical columns."""
        from .storage import Column

        records = list(self[:width])
        records.extend(Column(str(uuid4()), f"Column_{i}") for i in range(len(records), width))
        return type(self)(records)

    def insert(self, index, columns, *, overwrite=False):
        """Combine records and remap colliding identities in one prepared result."""
        if not 0 <= index <= len(self):
            raise IndexError(index)
        columns = tuple(columns)
        stop = index + len(columns) if overwrite else index
        return type(self)(_unique_records(self[:index] + columns + self[stop:]))

    def groups(self, shape, *, row=False, startx=0, no_guess=False, what=None):
        """Derive fresh coordinate/error groups without changing explicit roles."""
        return _role_groups(self.roles, shape, row=row, startx=startx, no_guess=no_guess, what=what)


def _unique_records(records):
    """Keep each first identity and allocate new identities for later occurrences."""
    seen = set()
    for column in records:
        if column.id in seen:
            column = replace(column, id=str(uuid4()))
        seen.add(column.id)
        yield column


def _resolve_roles(headers, selector, *, force_list=False, first=False):
    """Resolve detached headers with CS2 semantics without inventing column IDs."""
    from .storage import resolve_columns

    return resolve_columns(tuple(_Header(header) for header in headers), selector,
                           force_list=force_list, first=first)


def _assign_roles(headers, roles, key, value):
    """Prepare one role assignment, validating every selector and role first."""
    if key is None:
        raise IndexError("A role assignment requires a column selector")
    result = list(roles)
    if isinstance(key, str) and len(key) == 1 and key in _ROLES + ".-":
        positions = _resolve_roles(headers, value, force_list=True)
        values = [key] * len(positions)
    else:
        positions = _resolve_roles(headers, key, force_list=True)
        if isinstance(value, str) and len(value) == 1:
            values = [value] * len(positions)
        else:
            try:
                values = list(value)
            except TypeError as error:
                raise IndexError(f"Invalid column role assignment: {value!r}") from error
            if len(values) != len(positions):
                raise ValueError("Role assignments must match the selected column count")
    if any(not isinstance(role, str) or len(role) != 1 or role not in _ROLES + ".-" for role in values):
        raise ValueError("Invalid column role")
    for position, role in zip(positions, values):
        if role != "-":
            result[position] = role
    return result


def _parse_roles(headers, roles, args, options):
    """Prepare callable Setas syntax without mutating current roles or options."""
    from .utils import decode_string

    options = dict(options)
    reset = options.pop("reset", True)
    reset = reset if isinstance(reset, bool) else True
    result = list(roles)
    if args:
        value = args[0]
        if isinstance(value, str):
            if reset:
                result = ["."] * len(result)
            value = decode_string(value)
        elif not isinstance(value, (Mapping, Sequence, np.ndarray)):
            raise SyntaxError(f"setas requires a role string, sequence or mapping, not {type(value)}")
    else:
        value = options
        if reset:
            result = ["."] * len(result)
    if isinstance(value, Mapping):
        for key, role in value.items():
            if isinstance(key, str) and len(key) == 1 and key in _ROLES:
                result = _assign_roles(headers, result, key, role)
            elif isinstance(role, str) and len(role) == 1 and role in _ROLES:
                result = _assign_roles(headers, result, key, role)
            else:
                raise IndexError(f"Invalid column assignment: {key!r}:{role!r}")
        return result
    if value is None:
        raise ValueError("Set as column string ended with a number")
    values = list(value)[:len(result)]
    values.extend(["."] * (len(result) - len(values)))
    if any(not isinstance(role, str) or len(role) != 1 or role.lower() not in _ROLES + ".-" for role in values):
        raise ValueError("Invalid column role")
    return [old if role == "-" else role.lower() for old, role in zip(result, values)]


def _add_roles(headers, roles, other):
    """Overlay explicit roles while retaining current roles at unassigned positions."""
    if isinstance(other, Mapping):
        return _parse_roles(headers, roles, (other,), {})
    prepared = _parse_roles(headers, roles, (other,), {})
    return [old if role == "." else role for old, role in zip(roles, prepared)]


def _subtract_roles(headers, roles, other):
    """Prepare role/selector subtraction, preserving encoded role-string grammar."""
    from .utils import decode_string

    result = list(roles)
    if isinstance(other, str):
        if other in (".", "-"):
            return result
        if len(other) == 1 and other in _ROLES:
            return ["." if role == other else role for role in result]
        if other and all(character in "0123456789" + _ROLES + ".-" for character in other):
            other = list(decode_string(other))
    if isinstance(other, Mapping):
        requested = _parse_roles(headers, ["."] * len(result), (other,), {})
        for i, role in enumerate(requested):
            if role != ".":
                if result[i] != role:
                    raise ValueError(f"Column {i} is not assigned as {role}")
                result[i] = "."
    elif isinstance(other, (list, tuple)) and any(isinstance(item, str) and item in _ROLES + ".-" for item in other):
        for item in other:
            result = _subtract_roles(headers, result, item)
    else:
        for position in _resolve_roles(headers, other, force_list=True):
            result[position] = "."
    return result


def _update_roles(headers, roles, assignments):
    """Replace named role groups in mapping order, retaining unrelated roles."""
    result = list(roles)
    cleared = set()
    for key, value in assignments.items():
        role = key if isinstance(key, str) and len(key) == 1 and key in _ROLES else value
        if not isinstance(role, str) or len(role) != 1 or role not in _ROLES:
            raise IndexError(f"Invalid column assignment: {key!r}:{value!r}")
        if role not in cleared:
            result = ["." if current == role else current for current in result]
            cleared.add(role)
        result = _assign_roles(headers, result, key, value)
    return result


def _role_groups(roles, shape, *, row=False, startx=0, no_guess=False, what=None):
    """Derive fresh scientific groups with absolute uncertainty positions."""
    from ..tools import AttributeStore

    roles = tuple(roles)
    inferred = roles.count("x") == 0 and len(shape) >= 2 and shape[1] in _DEFAULT_ROLES and not no_guess
    if inferred:
        roles = tuple(_DEFAULT_ROLES[shape[1]])
    xpositions = [i for i, role in enumerate(roles) if role == "x"]
    if len(xpositions) == 1:
        xcol, start, end = xpositions[0], 0, len(roles)
    elif xpositions:
        eligible = [position for position in xpositions if position >= startx]
        if not eligible:
            raise ValueError("No assigned X group exists at or after startx")
        xcol = start = eligible[0]
        end = next((position for position in xpositions if position > xcol), len(roles))
    else:
        xcol, start, end = None, 0, len(roles)
    xerr = None if xcol is None else next((i for i in range(start, end) if roles[i] == "d"), None)
    result = AttributeStore(xcol=xcol, xerr=xerr)
    for role, field in _FIELDS.items():
        result[field] = [i for i in range(start, end) if roles[i] == role]
    axes = 0 if xcol is None else 1 if not result.ycol else 2 if not result.zcol else 3
    if axes == 2 and result.ucol and result.vcol:
        axes = 4
    elif axes == 3 and result.ucol and result.vcol:
        axes = 6 if result.wcol else 5
    if inferred and shape[1] == 5:
        axes, result.zcol = 5, None
    result.axes = axes
    for field in ("xcol", "xerr", "ycol", "yerr", "zcol", "zerr", "ucol", "vcol", "wcol", "axes"):
        result[f"has_{field}"] = result[field] is not None and result[field] != []
    result.has_uvw = result.has_ucol and result.has_vcol and result.has_wcol
    if what in ("xcol", "xerr"):
        return result[what]
    if what in _FIELDS.values():
        return result[what][0]
    if what in ("ycols", "zcols", "ucols", "vcols", "wcols", "yerrs", "zerrs"):
        return result[what[:-1]]
    return result
