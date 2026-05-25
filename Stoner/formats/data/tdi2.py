# -*- coding: utf-8 -*-
"""File loader routine for the TDI Format 2.0 files being developed as part of te stoner_measurement POython
based measurement code.

TDI Format 2.0 is very similar to the earlier version except dictionaries and lists in the metadata are
flattened before being placed in the first solumn as key0-valkue opairs and the type hints are Python
types rather than LabVIEDW Type Descriptors.
"""

import re
import ast
from typing import Any, Union
from pathlib import Path

import pandas as pd
import numpy as np

from ..decorators import register_loader, register_saver
from ...core.array import DataArray
from ...core.data import Data
from ...core.exceptions import StonerLoadError
from ...tools.file import FileManager, get_filename
from ...tools.typing import Args, Filename, Kwargs

_PATH_TOKEN_RE = re.compile(
    r"""
    ([^. \[\]]+)     # dict key
    |                # OR
    \[(\d+)\]        # list index
""",
    re.VERBOSE,
)


_ENTRY_RE = re.compile(
    r"""
    ^(.*?)           # path (non-greedy)
    \{([^}]+)\}      # {typename}
    =(.+)$           # =value
""",
    re.VERBOSE,
)


def _parse_entry(entry: str):
    match = _ENTRY_RE.match(entry)
    if not match:
        raise ValueError(f"Invalid entry format: {entry}")
    path, typename, value_str = match.groups()

    # Convert string to Python value safely
    value = ast.literal_eval(value_str)
    return path, value


def _parse_path(path: str):
    tokens = []
    for key, index in _PATH_TOKEN_RE.findall(path):
        if key:
            tokens.append(key)
        elif index:
            tokens.append(int(index))
    return tokens


def _ensure_list_size(lst, index):
    while len(lst) <= index:
        lst.append(None)


def _flatten_to_metadata(obj: Any, prefix: str = "") -> list[str]:
    """Recursively flatten a nested dict or list into TDI metadata cell strings.

    Each leaf value is formatted as ``{prefix}{typename}={repr(value)}``.
    Nested dict keys are joined to the prefix with a ``.`` separator; list
    indices use ``[{index}]`` notation.  Any object with an ``.item()`` method
    (e.g. a numpy scalar) is converted to its Python native equivalent before
    formatting so that type names and ``repr`` output are clean.

    Args:
        obj (Any):
            The value to flatten.  Typically the ``dict`` returned by a
            plugin's :meth:`~stoner_measurement.plugins.base_plugin.BasePlugin.to_json`
            method, or a scalar produced by evaluating a ``_values`` expression.

    Keyword Parameters:
        prefix (str):
            Dot-separated key path accumulated by recursive calls.  Pass an
            empty string (the default) to start from the root.

    Returns:
        (list[str]):
            Ordered list of ``"{key}{typename}={repr(value)}"`` strings, one
            per leaf value in *obj*.

    Examples:
        >>> _flatten_to_metadata({"a": {"b": 1}, "c": [{"A": 2.0}, {"B": 4}]})
        ['a.b{int}=1', 'c[0].A{float}=2.0', 'c[1].B{int}=4']
        >>> _flatten_to_metadata(42, "x")
        ['x{int}=42']
        >>> _flatten_to_metadata("hello", "s")
        ["s{str}='hello'"]
    """
    entries: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            child = f"{prefix}.{k}" if prefix else str(k)
            entries.extend(_flatten_to_metadata(v, child))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            entries.extend(_flatten_to_metadata(v, f"{prefix}[{i}]"))
    else:
        # Convert numpy scalars (or any object with .item()) to Python natives.
        if hasattr(obj, "item"):
            obj = obj.item()
        typename = type(obj).__name__
        entries.append(f"{prefix}{{{typename}}}={repr(obj)}")
    return entries


def _build_rows(
    metadata: list[str],
    columns: list[tuple[str, np.ndarray]],
) -> list[list[str]]:
    """Build row-wise TDI table data from metadata and numeric columns."""
    header_row = ["TDI Format 2.0"] + [col[0] for col in columns]
    max_data_len = max((len(col[1]) for col in columns), default=0)
    n_rows = max(len(metadata), max_data_len)
    yield header_row
    for i in range(n_rows):
        meta_cell = metadata[i] if i < len(metadata) else ""
        data_cells = [str(col[1][i]) if i < len(col[1]) else "" for col in columns]
        yield [meta_cell] + data_cells


def _inverse_flatten_metadata(entries: list[str]) -> Any:
    root: Union[dict, list, None] = {}

    for entry in entries:
        if not isinstance(entry, str):
            continue
        path, value = _parse_entry(entry)
        tokens = _parse_path(path)

        current = root
        parent = None
        parent_key = None

        for i, token in enumerate(tokens):
            is_last = i == len(tokens) - 1

            if isinstance(token, str):
                # Ensure dict
                if not isinstance(current, dict):
                    new_dict = {}
                    if isinstance(parent, list):
                        parent[parent_key] = new_dict
                    elif isinstance(parent, dict):
                        parent[parent_key] = new_dict
                    else:
                        root = new_dict
                    current = new_dict

                if is_last:
                    current[token] = value
                else:
                    if token not in current or current[token] is None:
                        # Decide next container type
                        next_token = tokens[i + 1]
                        current[token] = [] if isinstance(next_token, int) else {}
                    parent = current
                    parent_key = token
                    current = current[token]

            else:  # list index
                if not isinstance(current, list):
                    new_list = []
                    if isinstance(parent, dict):
                        parent[parent_key] = new_list
                    elif isinstance(parent, list):
                        parent[parent_key] = new_list
                    else:
                        root = new_list
                    current = new_list

                _ensure_list_size(current, token)

                if is_last:
                    current[token] = value
                else:
                    if current[token] is None:
                        next_token = tokens[i + 1]
                        current[token] = [] if isinstance(next_token, int) else {}
                    parent = current
                    parent_key = token
                    current = current[token]

    return root


@register_loader(
    patterns=[(".dat", 8), (".txt", 8), ("*", 8)],
    mime_types=[("application/tsv", 8), ("text/plain", 8), ("text/tab-separated-values", 8)],
    name="TDI_2_0",
    what="Data",
)
def load_tdi2_format(new_data: Data, *args: Args, **kwargs: Kwargs) -> Data:
    """Actually load the data from disc assuming a .tdi file format.

    Args:
        new_data (Data):
            A newly instantiated Data object into which the instance will be loaded.
        *args:
            Other arguments are used if filename is not specified.

    Keyword Arguments:
        **kwargs:
            Other keyword arguments are passed to get_filename.

    Returns:
        DataFile:
            A copy of the newly loaded :py:class`DataFile` object.

    Exceptions:
        StonerLoadError:
            Raised if the first row does not start with 'TDI Format 1.5' or 'TDI Format=1.0'.

    Note:
        The *_load* methods should be overridden in each child class to handle the process of loading data from
        disc. If they encounter unexpected data, then they should raise StonerLoadError to signal this, so that
        the loading class can try a different sub-class instead.
    """
    filename, args, kwargs = get_filename(args, kwargs)
    if filename is None or not filename:
        new_data.get_filename("r")
    else:
        new_data.filename = filename
    with FileManager(new_data.filename, "r", encoding="utf-8", errors="ignore") as datafile:
        line = datafile.readline()
        if not line.startswith("TDI Format 2.0"):
            raise StonerLoadError("Not a TDI 2.0 File")
    df = pd.read_csv(filename, delimiter="\t")
    metadata = _inverse_flatten_metadata(df["TDI Format 2.0"])
    column_headers = df.columns[1:].tolist()
    data = DataArray(df.iloc[:, 1:].values)
    new_data.data = data
    new_data.column_headers = column_headers
    new_data.metadata = metadata
    return new_data


@register_saver(
    patterns=[(".dat", 12), (".txt", 12), ("*", 12)],
    name="TDI_2_0",
    what="Data",
)
def save_tdi2_format(save_data: Data, *args: Args, **kwargs: Kwargs) -> Data:
    """Write out a DataFile to a tab delimited tdi text file.

    Args:
        save_data (Data):
            A newly instantiated Data object into which the instance will be loaded.
        *args:
            Other arguments are used if filename is not specified.

    Keyword Arguments:
        **kwargs:
            Other keyword arguments are passed to get_filename.
    """
    filename, args, kwargs = get_filename(args, kwargs)
    metadata = _flatten_to_metadata(save_data.metadata)
    columns = [(col, save_data.data[:, ix]) for ix, col in enumerate(save_data.column_headers)]

    dest = Path(filename)
    content = "\n".join("\t".join(row) for row in _build_rows(metadata, columns)) + "\n"
    dest.write_text(content, encoding="utf-8")
    save_data.filename = filename
    return save_data
