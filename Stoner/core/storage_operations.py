"""Implement Data structural operations at the frame owner's mutation boundary."""

from collections.abc import Mapping
from functools import cmp_to_key
import re

import numpy as np

from .storage import _copy_metadata
from .base import TypeHintedDict


def _merge_metadata(left, right):
    """Retain left-owned values and incorporate independently imported right values."""
    result = TypeHintedDict()
    for source in (left, right):
        for key, value in source.items():
            if key not in result:
                super(TypeHintedDict, result).__setitem__(key, value)
                result.types[key] = source.type(key)
    return result


def add_column(datafile, column_data, header=None, index=None, func_args=None, replace=False, setas=None):
    """Prepare new columns and commit values, masks and schema in one operation."""
    owner = datafile._require_storage_owner()
    owner._check_writable()
    if callable(column_data):
        column_data = np.ma.asarray([column_data(row, **(func_args or {})) for row in datafile.rows()])
    if header is None:
        header = getattr(column_data, "column_headers", None)
    if isinstance(header, str):
        header = [header]
    width = len(owner.column_ids)
    index = width if index is None or index == width else datafile.find_col(index)
    owner.insert_columns(index, column_data, header, setas, overwrite=replace)
    return datafile


def del_column(datafile, col=None, duplicates=False):
    """Remove columns together with their exclusions and schema records."""
    owner = datafile._require_storage_owner()
    headers = list(owner.headers)
    if duplicates:
        target = None if col is None else headers[datafile.find_col(col)]
        seen, deleted = set(), []
        for i, header in enumerate(headers):
            if header in seen and (target is None or header == target):
                deleted.append(i)
            seen.add(header)
    elif col is None or col is False:
        deleted = np.flatnonzero(np.any(np.asarray(owner.mask), axis=0))
    elif col is True:
        deleted = [i for i, role in enumerate(owner.roles) if role == "."]
    elif isinstance(col, (list, np.ndarray)) and len(col) and all(isinstance(v, (bool, np.bool_)) for v in col):
        if len(col) != len(headers):
            raise ValueError("Column deletion mask must match column count")
        deleted = np.flatnonzero(col)
    else:
        deleted = datafile.find_col(col, force_list=True)
    owner.take(columns=[i for i in range(len(headers)) if i not in deleted])
    return datafile


def del_rows(datafile, col=None, val=None, invert=False):
    """Resolve deletion predicates once and select rows atomically."""
    owner = datafile._require_storage_owner()
    size = owner._state.values.shape[0]
    if col is None:
        deleted = np.any(np.asarray(owner.mask), axis=1)
    elif val is None and callable(col):
        deleted = np.array([bool(col(row)) for row in datafile.rows()])
    elif val is None:
        selection = np.asarray(col) if isinstance(col, (list, np.ndarray)) else col
        if isinstance(selection, np.ndarray) and selection.dtype.kind == "b":
            if len(selection) != size:
                raise ValueError("Row selection mask must match row count")
            # Historical Boolean input denotes rows to keep.
            deleted = ~selection
        else:
            deleted = np.zeros(size, dtype=bool)
            deleted[np.arange(size)[col]] = True
    else:
        column = datafile.find_col(col)
        if callable(val):
            deleted = np.array([row[column] is not np.ma.masked and bool(val(row[column], row))
                                for row in datafile.rows()])
        else:
            values = owner[:, column]
            if np.isscalar(val):
                deleted = np.ma.filled(values == val, False)
            else:
                lower, upper = min(val), max(val)
                deleted = np.ma.filled((values >= lower) & (values <= upper), False)
    owner.take(rows=deleted if invert else ~deleted)
    return datafile


def insert_rows(datafile, row, new_data):
    """Insert rows in place and retain column identities."""
    datafile._require_storage_owner().insert_rows(row, new_data)
    return datafile


def reorder_columns(datafile, cols, headers_too=True, setas_too=True):
    """Reorder complete columns; repeated selections receive fresh identities."""
    if not headers_too or not setas_too:
        raise ValueError("Frame-backed reordering always moves headers and roles with columns")
    datafile._require_storage_owner().take(columns=cols)
    return datafile


def swap_column(datafile, *swp, headers_too=True, **kwargs):
    """Swap complete columns, preserving schema association."""
    if kwargs or not headers_too:
        raise ValueError("Column swaps must move the complete schema")
    order = list(range(datafile.shape[1]))
    if len(swp) == 1 and isinstance(swp[0], list):
        swp = swp[0]
    for left, right in swp:
        left, right = datafile.find_col(left), datafile.find_col(right)
        order[left], order[right] = order[right], order[left]
    return reorder_columns(datafile, order)


def sort(datafile, *order, reverse=False):
    """Sort positional rows while applying the same permutation to exclusions."""
    owner = datafile._require_storage_owner()
    array = owner.to_numpy(masked=False)
    if len(order) == 1 and callable(order[0]):
        comparator = order[0]
        indices = sorted(range(len(array)), key=cmp_to_key(lambda a, b: comparator(array[a], array[b])))
    else:
        if not order:
            roles = datafile.setas.cols
            order = ([roles.xcol] if roles.xcol is not None else []) + roles.ycol + roles.zcol
            order = order or list(range(array.shape[1]))
        elif len(order) == 1 and isinstance(order[0], (list, tuple)):
            order = order[0]
        positions = datafile.find_col(list(order), force_list=True)
        indices = np.lexsort(tuple(array[:, i] for i in reversed(positions))) if positions else np.arange(len(array))
    if reverse:
        indices = indices[::-1]
    owner.take(rows=indices)
    return datafile


def append(datafile, other):
    """Append rows, aligning Data columns by header occurrence without pandas labels."""
    owner = datafile._require_storage_owner()
    owner._check_writable()
    if np.isscalar(other):
        return NotImplemented
    if isinstance(other, list):
        # Work on an independent object so a late invalid row cannot partly append.
        draft = datafile.clone
        for item in other:
            append(draft, np.asarray(item) if isinstance(item, list) else item)
        draft._require_storage_owner()._state.metadata = _copy_metadata(
            draft._require_storage_owner()._state.metadata, {id(draft): datafile})
        owner._state = draft._require_storage_owner()._state
        return datafile
    if isinstance(other, Mapping) and not hasattr(other, "export_storage"):
        draft = datafile.clone
        order = {}
        for header in other:
            try:
                order[header] = draft.find_col(header)
            except (KeyError, re.error):
                draft.add_column(np.ma.array(np.full(len(draft), np.nan), mask=True), header=header)
                order[header] = draft.shape[1] - 1
        row = np.ma.array(np.full(draft.shape[1], np.nan), mask=True)
        for header, position in order.items():
            row[position] = other[header]
        draft._require_storage_owner().insert_rows(len(draft), row)
        draft._require_storage_owner()._state.metadata = _copy_metadata(
            draft._require_storage_owner()._state.metadata, {id(draft): datafile})
        owner._state = draft._require_storage_owner()._state
        return datafile
    if hasattr(other, "export_storage"):
        package = other.export_storage()
        available = {}
        for i, column in enumerate(package.schema):
            available.setdefault(column.header, []).append(i)
        positions = []
        for header in owner.headers:
            if not available.get(header):
                positions.append(None)
            else:
                positions.append(available[header].pop(0))
        if not owner.column_ids:
            values = package.to_numpy()
        elif None in positions:
            dtype = np.result_type(package.dtype, np.float64)
            owner._check_integer_precision(dtype, package.to_numpy())
            values = np.ma.array(np.full((len(package.values), len(positions)), np.nan, dtype=dtype), mask=False)
            source = package.to_numpy()
            for i, position in enumerate(positions):
                if position is not None:
                    values[:, i] = source[:, position]
        else:
            values = package.to_numpy()[:, positions]
        metadata = _merge_metadata(owner._state.metadata, package.metadata)
    else:
        values = np.ma.atleast_2d(other)
        metadata = owner._state.metadata
    if not owner.column_ids:
        if hasattr(other, "export_storage"):
            owner.replace(values, package.schema)
        else:
            owner.replace(values)
    else:
        owner.insert_rows(len(datafile), values)
    owner._state.metadata = metadata
    return datafile


def concatenate_columns(datafile, other):
    """Append complete columns, remapping colliding IDs and merging metadata left first."""
    owner = datafile._require_storage_owner()
    if hasattr(other, "export_storage"):
        package = other.export_storage()
        metadata = _merge_metadata(owner._state.metadata, package.metadata)
        owner.insert_columns(len(owner.column_ids), package.to_numpy(), schema=package.schema, mask_padding=True)
        owner._state.metadata = metadata
    else:
        owner.insert_columns(len(owner.column_ids), other, mask_padding=True)
    return datafile
