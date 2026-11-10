"""Share constructor and HDF5 persistence boundaries for instrument scan stacks."""

from pathlib import PurePath
from os.path import basename

import numpy as np
import h5py

from ...core.base import TypeHintedDict

from ...core.exceptions import StonerLoadError


def parse_scan_args(args, kwargs, pattern, template, *, path_type=PurePath):
    """Consume a scan name or number while retaining instrument-specific naming."""
    args = [str(arg) if isinstance(arg, path_type) else arg for arg in args]
    if args and isinstance(args[0], str):
        root_name = args.pop(0)
        match = pattern.search(root_name)
        scan = int(match.groups()[0]) if match else -1
    elif args and isinstance(args[0], int):
        scan = args.pop(0)
        root_name = template.format(scan)
    else:
        root_name = kwargs.pop("root", None)
        scan = kwargs.pop("scan", -1)
    return args, root_name, scan


def raise_scan_error(openfile, message=""):
    """Raise the load exception while closing a supplied file when possible."""
    try:
        raise StonerLoadError(message)
    finally:
        try:
            openfile.close()
        except (AttributeError, TypeError, ValueError, IOError):
            pass


def write_typed_metadata(source, metadata, typehints):
    """Persist typed entries, retaining the established unsupported-type fallback."""
    for key in source:
        try:
            typehints.attrs[key] = source._typehints[key]
            metadata.attrs[key] = source[key]
        except TypeError:
            parts = source.export(key).split("=")
            metadata.attrs[key] = "=".join(parts[1:])


def write_scan_frame(signal, image, name, *, compression, compression_opts):
    """Store raw image values, exclusions, fill value and typed frame metadata."""
    # The scan owner depends on image storage, so resolve this only when saving.
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from ...Image.scan_metadata import write_exclusions

    signal.require_dataset(
        name, data=image.to_numpy(), shape=image.shape, dtype=image.dtype,
        compression=compression, compression_opts=compression_opts,
    )
    metadata = signal.require_group("metadata")
    typehints = signal.require_group("typehints")
    signal[name][...] = image.to_numpy(masked=False)
    write_exclusions(signal, image)
    write_typed_metadata(image.metadata, metadata, typehints)


def initialise_scan_metadata(scan, number):
    """Initialise the shared typed scan number through the owner-backed interface."""
    scan._common_metadata = TypeHintedDict()
    scan.scan_no = number
    scan._common_metadata["Scan #"] = number


def read_typed_metadata(target, metadata, typehints):
    """Restore explicit string type hints while retaining native HDF5 values."""
    for key in sorted(metadata):
        value = metadata[key]
        hint = typehints.get(key, "Detect")
        if isinstance(value, str) and hint != "Detect":
            target[f"{key}{{{hint}}}".strip()] = f"{value}".strip()
        else:
            target[key] = value


def read_scan_frame(group, image, name):
    """Restore image pixels, typed metadata and optional lossless exclusions."""
    # Image ownership depends on the scan bridge; restore only after import.
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from ...Image.scan_metadata import read_exclusions

    data = group[name]
    image.image = data[...] if np.prod(np.array(data.shape)) > 0 else [[]]
    metadata = group.require_group("metadata")
    typehints = group.get("typehints", None)
    typehints = typehints.attrs if isinstance(typehints, h5py.Group) else {}
    read_typed_metadata(image.metadata, metadata.attrs, typehints)
    image.filename = basename(group.name)
    return read_exclusions(group, image)


def write_common_metadata(group, source):
    """Reuse a parent's shared headers or write independent typed scan metadata."""
    if "common_metadata" in group.parent and "common_metadata" not in group:
        group["common_metadata"] = h5py.SoftLink(group.parent["common_metadata"].name)
        group["common_typehints"] = h5py.SoftLink(group.parent["common_typehints"].name)
    else:
        metadata = group.require_group("common_metadata")
        typehints = group.require_group("common_typehints")
        write_typed_metadata(source, metadata, typehints)


def read_common_metadata(target, group):
    """Validate and restore shared typed headers, returning the other child names."""
    names = list(group.keys())
    if "common_metadata" not in names or "common_typehints" not in names:
        raise_scan_error(group, message="Couldn't find common metadata groups, something is not right here!")
    read_typed_metadata(target, group["common_metadata"].attrs, group["common_typehints"].attrs)
    names.remove("common_metadata")
    names.remove("common_typehints")
    return names


def read_scan_groups(scan, group, names, read_frame, load_group, args, kwargs):
    """Restore grouped scans recursively and plain frames in their recorded order."""
    for name in names:
        child = group[name]
        if "type" in child.attrs:
            scan.groups[name] = load_group(child, *args, **kwargs)
        else:
            scan.append(read_frame(child))
