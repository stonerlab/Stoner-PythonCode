"""Validate column cardinality at scientific and structural operation boundaries."""

import numpy as np


def _single_column(data, selector, name="column"):
    """Return one position or explain an ambiguous scientific selection.

    Args:
        data (Data):
            Owner supplying the shared resolver and displayed headers.
        selector (column selector):
            Selection which must identify exactly one column.
        name (str):
            Operation argument name used in the error message.

    Returns:
        int:
            The unique selected position.

    Raises:
        ValueError:
            If the selector identifies zero or several columns.
    """
    positions = data.find_col(selector, force_list=True)
    if len(positions) != 1:
        matches = [(position, data.column_headers[position]) for position in positions]
        raise ValueError(f"{name} must select exactly one column; {selector!r} matched {matches!r}")
    return positions[0]


def _insertion_position(data, selector):
    """Resolve one insertion anchor, retaining None and width as append sentinels."""
    if selector is None:
        return data.shape[1]
    if isinstance(selector, (int, np.integer)) and not isinstance(selector, (bool, np.bool_)):
        if selector == data.shape[1]:
            return int(selector)
    return _single_column(data, selector, "index")
