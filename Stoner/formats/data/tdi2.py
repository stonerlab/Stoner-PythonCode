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

import pandas as pd

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


def inverse_flatten_metadata(entries: list[str]) -> Any:
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
    metadata = inverse_flatten_metadata(df["TDI Format 2.0"])
    column_headers = df.columns[1:].tolist()
    data = DataArray(df.iloc[:, 1:].values)
    new_data.data = data
    new_data.column_headers = column_headers
    new_data.metadata = metadata
    return new_data
