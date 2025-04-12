# -*- coding: utf-8 -*-
"""HDF5 format Data loader routines"""
import os

import h5py
import numpy as np
from importlib import import_module

from ...compat import string_types, bytes2str, path_types, str2bytes
from ...core.exceptions import StonerLoadError
from ..decorators import register_loader, register_saver
from ...tools.file import HDFFileManager, get_filename


@register_loader(
    patterns=[(".hdf", 16), (".hf5", 16)],
    mime_types=[("application/x-hdf", 16), ("application/x-hdf5", 16)],
    name="HDF5File",
    what="Data",
)
def load_hdf(new_data, *args, **kargs):  # pylint: disable=unused-argument
    """Create a new HDF5File from an actual HDF file."""
    filename, args, kargs = get_filename(args, kargs)
    with HDFFileManager(filename, "r") as f:
        if "data" in f.keys():
            data = f["data"]
        else:
            raise StonerLoadError("Not our hdf5 file format.")
        if np.prod(np.array(data.shape)) > 0:
            new_data.data = data[...]
        else:
            new_data.data = [[]]
        metadata = f.require_group("metadata")
        typehints = f.get("typehints", None)
        if not isinstance(typehints, h5py.Group):
            typehints = {}
        else:
            typehints = typehints.attrs
        if "column_headers" in f.attrs:
            new_data.column_headers = [bytes2str(x) for x in f.attrs["column_headers"]]
            if isinstance(new_data.column_headers, string_types):
                new_data.column_headers = new_data.metadata.string_to_type(new_data.column_headers)
            new_data.column_headers = [bytes2str(x) for x in new_data.column_headers]
        else:
            raise StonerLoadError("Couldn't work out where my column headers were !")
        for i in sorted(metadata.attrs):
            v = metadata.attrs[i]
            t = typehints.get(i, "Detect")
            if isinstance(v, string_types) and t != "Detect":  # We have typehints and this looks like it got exported
                new_data.metadata[f"{i}{{{t}}}".strip()] = f"{v}".strip()
            else:
                new_data[i] = metadata.attrs[i]
        if isinstance(f, h5py.Group):
            if f.name != "/":
                new_data.filename = os.path.join(f.file.filename, f.name)
            else:
                new_data.filename = os.path.realpath(f.file.filename)
        else:
            new_data.filename = os.path.realpath(f.filename)
    return new_data


@register_saver(patterns=[(".hdf", 16), (".hf5", 16)], name="HDF5File", what="Data")
def save_hdf(save_data, filename=None, **kargs):  # pylint: disable=unused-argument
    """Write the current object into  an hdf5 file or group within a file.

    Writes the data in afashion that is compatible with being loaded in again.

    Args:
        filename (string or h5py.Group):
            Either a string, of h5py.File or h5py.Group object into which to save the file. If this is a string,
            the corresponding file is opened for writing, written to and save again.

    Returns
        A copy of the object
    """
    if filename is None:
        filename = save_data.filename
    if filename is None or (isinstance(filename, bool) and not filename):  # now go and ask for one
        filename = save_data.__file_dialog("w")
        save_data.filename = filename
    if isinstance(filename, path_types):
        mode = "r+" if os.path.exists(filename) else "w"
    else:
        mode = ""
    with HDFFileManager(filename, mode) as f:
        f.require_dataset(
            "data",
            data=save_data.data,
            shape=save_data.data.shape,
            dtype=save_data.data.dtype,
        )
        metadata = f.require_group("metadata")
        typehints = f.require_group("typehints")
        for k in save_data.metadata:
            try:
                typehints.attrs[k] = save_data.metadata._typehints[k]
                metadata.attrs[k] = save_data[k]
            except TypeError:
                # We get this for trying to store a bad data type - fallback to metadata export to string
                parts = save_data.metadata.export(k).split("=")
                metadata.attrs[k] = "=".join(parts[1:])
        f.attrs["column_headers"] = [str2bytes(x) for x in save_data.column_headers]
        f.attrs["filename"] = save_data.filename
        f.attrs["type"] = type(save_data).__name__
        f.attrs["module"] = type(save_data).__module__
    return save_data


def _scan_SLS_meta(new_data, group):
    """Scan the HDF5 Group for attributes and datasets and sub groups and recursively add them to the metadata."""
    root = ".".join(group.name.split("/")[2:])
    for name, thing in group.items():
        parts = thing.name.split("/")
        name = ".".join(parts[2:])
        if isinstance(thing, h5py.Group):
            _scan_SLS_meta(new_data, thing)
        elif isinstance(thing, h5py.Dataset):
            if thing.ndim > 1:
                continue
            if np.prod(thing.shape) == 1:
                new_data.metadata[name] = thing[0]
            else:
                new_data.metadata[name] = thing[...]
    for attr in group.attrs:
        new_data.metadata[f"{root}.{attr}"] = group.attrs[attr]


@register_loader(
    patterns=[(".hdf", 16), (".hdf5", 16)],
    mime_types=[("application/x-hdf", 16), ("application/x-hdf5", 16)],
    name="SLS_STXMFile",
    what="Data",
)
def load_sls_stxm(new_data, *args, **kargs):
    """Load data from the hdf5 file produced by Pollux.

    Args:
        h5file (string or h5py.Group):
            Either a string or an h5py Group object to load data from

    Returns:
        itnew_data after having loaded the data
    """
    filename, args, kargs = get_filename(args, kargs)
    new_data.filename = filename
    if isinstance(filename, path_types):  # We got a string, so we'll treat it like a file...
        try:
            f = h5py.File(filename, "r")
        except IOError as err:
            raise StonerLoadError(f"Failed to open {filename} as a n hdf5 file") from err
    elif isinstance(filename, h5py.File) or isinstance(filename, h5py.Group):
        f = filename
    else:
        raise StonerLoadError(f"Couldn't interpret {filename} as a valid HDF5 file or group or filename")
    items = [x for x in f.items()]
    if len(items) == 1 and items[0][0] == "entry1":
        group1 = [x for x in f["entry1"]]
        if "definition" in group1 and bytes2str(f["entry1"]["definition"][0]) == "NXstxm":  # Good HDF5
            pass
        else:
            raise StonerLoadError("HDF5 file lacks single top level group called entry1")
    else:
        raise StonerLoadError("HDF5 file lacks single top level group called entry1")
    root = f["entry1"]
    data = root["counter0"]["data"]
    if np.prod(np.array(data.shape)) > 0:
        new_data.data = data[...]
    else:
        new_data.data = [[]]
    _scan_SLS_meta(new_data, root)
    if "file_name" in f.attrs:
        new_data["original filename"] = f.attrs["file_name"]
    elif isinstance(f, h5py.Group):
        new_data["original filename"] = f.name
    else:
        new_data["original filename"] = f.file.filename

    if isinstance(filename, path_types):
        f.file.close()
    new_data["Loaded from"] = new_data.filename

    if "instrument.sample_x.data" in new_data.metadata:
        new_data.metadata["actual_x"] = new_data.metadata["instrument.sample_x.data"].reshape(new_data.shape)
    if "instrument.sample_y.data" in new_data.metadata:
        new_data.metadata["actual_y"] = new_data.metadata["instrument.sample_y.data"].reshape(new_data.shape)
    new_data.metadata["sample_x"], new_data.metadata["sample_y"] = np.meshgrid(
        new_data.metadata["counter0.sample_x"], new_data.metadata["counter0.sample_y"]
    )
    if "control.data" in new_data.metadata:
        mod = import_module("Stoner.Image")
        ImageArray = getattr(mod, "ImageArray")

        new_data.metadata["beam current"] = ImageArray(new_data.metadata["control.data"].reshape(new_data.data.shape))
        new_data.metadata["beam current"].metadata = new_data.metadata
    new_data.data = new_data.data[::-1]
    return new_data


def _hgx_scan_group(new_data, grp, pth):
    """Recursively list HDF5 Groups."""
    if pth in new_data.seen:
        return None
    new_data.seen.append(pth)
    if not isinstance(grp, h5py.Group):
        return None
    if new_data.debug:
        if new_data.debug:
            print(f"Scanning in {pth}")
    for x in grp:
        if pth == "":
            new_pth = x
        else:
            new_pth = pth + "." + x
        if pth == "" and x == "data":  # Special case for main data
            continue
        if isinstance(grp[x], type(grp)):
            _hgx_scan_group(new_data, grp[x], new_pth)
        elif isinstance(grp[x], h5py.Dataset):
            y = grp[x][...]
            new_data[new_pth] = y
    return None


def _hgx_main_data(new_data, data_grp):
    """Work through the main data group and build something that looks like a numpy 2D array."""
    if not isinstance(data_grp, h5py.Group) or data_grp.name != "/current/data":
        raise StonerLoadError("HDF5 file not in expected format")
    root = data_grp["datasets"]
    for ix in root:  # Hack - iterate over all items in root, but actually data is in Groups not DataSets
        dataset = root[ix]
        if isinstance(dataset, h5py.Dataset):
            continue
        x = dataset["x"][...]
        y = dataset["y"][...]
        e = dataset["error"][...]
        new_data &= x
        new_data &= y
        new_data &= e
        new_data.column_headers[-3] = bytes2str(dataset["x_command"][()])
        new_data.column_headers[-2] = bytes2str(dataset["y_command"][()])
        new_data.column_headers[-1] = bytes2str(dataset["error_command"][()])
        new_data.column_headers = [str(ix) for ix in new_data.column_headers]


@register_loader(
    patterns=(".hgx", 16),
    mime_types=[("application/x-hdf", 32), ("application/x-hdf5", 32)],
    name="HGXFile",
    what="Data",
)
def _load(new_data, *args, **kargs):
    """Load a GenX HDF file.

    Args:
        filename (string or bool):
            File to load. If None then the existing filename is used, if False, then a file dialog will be used.

    Returns:
        A copy of the itnew_data after loading the data.
    """
    filename, args, kargs = get_filename(args, kargs)
    new_data.filename = filename
    new_data.seen = []
    with HDFFileManager(new_data.filename, "r") as f:
        try:
            if "current" in f and "config" in f["current"]:
                pass
            else:
                raise StonerLoadError("Looks like an unexpected HDF layout!.")
            _hgx_scan_group(new_data, f["current"], "")
            _hgx_main_data(new_data, f["current"]["data"])
        except IOError as err:
            raise StonerLoadError("Looks like an unexpected HDF layout!.") from err
    return new_data
