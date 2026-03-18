# -*- coding: utf-8 -*-
"""Tools for manipulating json."""


def flatten_json(data, parent_key=""):
    """Flatten a nested JSON-like structure into a dotted-key dictionary.

    Args:
        data: The JSON-like structure to flatten. May contain dictionaries,
            lists, and scalar values.
        parent_key: The prefix to prepend to keys in the flattened output.
            Used internally during recursion.

    Returns:
        dict: A flat dictionary mapping dotted/bracketed key paths to scalar
        values.

    This function recursively flattens nested dictionaries and lists into a
    single-level dictionary where:

      * Nested dictionary keys are joined using dot notation.
      * List indices are represented using bracket notation, e.g. "[0]".
      * Scalar values (str, int, float, bool, None) become the final values.

    The function is pure: it does not mutate input data and does not rely on
    side effects. Each recursive call returns a new dictionary, and the caller
    merges results.

    Examples:
        >>> flatten_json({"a": {"b": 1}, "c": [10, 20]})
        {'a.b': 1, 'c[0]': 10, 'c[1]': 20}

        >>> flatten_json({"x": {"y": {"z": True}}})
        {'x.y.z': True}
    """
    items = {}

    match data:
        case dict():
            for key, value in data.items():
                new_key = f"{parent_key}.{key}" if parent_key else key
                items.update(flatten_json(value, new_key))

        case list():
            for idx, value in enumerate(data):
                new_key = f"{parent_key}[{idx}]"
                items.update(flatten_json(value, new_key))

        case _:
            items[parent_key] = data

    return items


def find_paths(data, target_key, target_value, path=None):
    """Yield all paths leading to a key/value pair in a nested structure.

    Args:
        data: The JSON-like structure to search. May contain dictionaries,
            lists, and scalar values.
        target_key: The dictionary key to match.
        target_value: The value that must be associated with `target_key`
            for a path to be considered a match.
        path: Internal recursion parameter. A list representing the path
            taken so far. Users should not supply this argument.

    Yields:
        list[str]: A list of path components representing the full ancestry
        from the root to the matching key/value pair.


    This function recursively traverses a nested JSON-like structure
    (dictionaries, lists, and scalar values) and yields every path where
    `target_key` equals `target_value`. Paths are returned as lists of
    components, where dictionary keys are plain strings and list indices
    are represented as bracketed strings (e.g., "[0]").

    Examples:
        >>> data = {"A": {"B": {"HasData": True}}}
        >>> list(find_paths(data, "HasData", True))
        [['A', 'B', 'HasData']]

        >>> data = {"items": [{"HasData": True}, {"HasData": False}]}
        >>> list(find_paths(data, "HasData", True))
        [['items', '[0]', 'HasData']]
    """
    if path is None:
        path = []

    match data:
        case dict():
            for key, value in data.items():
                new_path = path + [key]
                if key == target_key and value == target_value:
                    yield new_path
                yield from find_paths(value, target_key, target_value, new_path)

        case list():
            for idx, value in enumerate(data):
                new_path = path + [idx]
                yield from find_paths(value, target_key, target_value, new_path)

        case _:
            return


def find_parent_dicts(data, target_key, target_value):
    """Yield dictionaries that contain a matching key/value pair.

    Args:
        data: The JSON-like structure to search. May contain dictionaries,
            lists, and scalar values.
        target_key: The dictionary key to match.
        target_value: The required value associated with `target_key`.

    Yields:
        dict: A dictionary that contains the matching key/value pair.

    This function recursively searches a nested JSON-like structure and
    yields every dictionary in which `target_key` exists and its value
    equals `target_value`. Unlike `find_paths`, this function returns the
    dictionary object itself, allowing callers to inspect sibling keys or
    modify the parent structure.

    Examples:
        >>> data = {"A": {"B": {"HasData": True, "Other": 5}}}
        >>> list(find_parent_dicts(data, "HasData", True))
        [{'HasData': True, 'Other': 5}]

        >>> data = {"items": [{"HasData": True}, {"HasData": False}]}
        >>> list(find_parent_dicts(data, "HasData", True))
        [{'HasData': True}]
    """
    match data:
        case dict():
            if target_key in data and data[target_key] == target_value:
                yield data
            for value in data.values():
                yield from find_parent_dicts(value, target_key, target_value)

        case list():
            for value in data:
                yield from find_parent_dicts(value, target_key, target_value)

        case _:
            return


if __name__ == "__main__":
    data = {
        "key1": {"subkey1": 1, "subkey2": 2},
        "key2": ["value2.1", {"subkey3": "value2.2.1", "subkey4": "value2.2.2", "HasData": True}],
    }
    output = flatten_json(data)
    output2 = [pth for pth in find_paths(data, "HasData", True)]
    output3 = [pth for pth in find_parent_dicts(data, "HasData", True)]
