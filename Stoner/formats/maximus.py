# -*- coding: utf-8 -*-
"""Support files from the Bessy Maximus instrument."""

__all__ = ["hdr_to_dict", "read_scan", "MaximusImage"]

import json
import re
from pathlib import Path
from os import path
from copy import deepcopy

import numpy as np
import h5py

# Imports for use in Stoner package
from ..core.exceptions import StonerLoadError
from ..core.base import TypeHintedDict
from ..Image import ImageFile, ImageStack, ImageArray
from ..Core import DataFile
from ..compat import string_types
from ..HDF5 import HDFFileManager
from ..tools.file import FileManager, get_filename

SCAN_NO = re.compile(r"MPI_(\d+)")


def _raise_error(openfile, message=""):
    """Raise a StonerLoadError after trying to close file."""
    try:
        raise StonerLoadError(message)
    finally:
        try:
            openfile.close()
        except (AttributeError, TypeError, ValueError, IOError):
            pass


class MaximusSpectra(DataFile):
    """Provides a :py:class:`Stoner.DataFile` subclass for loading Point spectra from Maximus."""

    # We treat the hdr file as the key file type
    _patterns = ["*.hdr", "*.xsp"]
    mime_type = ["text/plain"]

    priority = 16

    def _load(self, *args, **kargs):
        """Maximus xsp file loader routine.

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.
        """
        filename, args, kargs = get_filename(args, kargs)
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename
        # Open the file and read the main file header and unpack into a dict
        try:
            pth = Path(self.filename)
        except (TypeError, ValueError) as err:
            raise StonerLoadError("Can only open things that can be converted to paths!") from err
        if pth.suffix != ".hdr":  # Passed a .xim or .xsp file in instead of the hdr file.
            pth = Path("_".join(str(pth).split("_")[:-1]) + ".hdr")
        stem = pth.parent / pth.stem

        try:
            hdr = _flatten_header(hdr_to_dict(pth))
            if "Point Scan" not in hdr["ScanDefinition.Type"]:
                raise StonerLoadError("Not an Maximus Single Image File")
        except (StonerLoadError, ValueError, TypeError, IOError) as err:
            raise StonerLoadError("Error loading as Maximus File") from err
        header, data, dims = read_scan(stem)
        self.metadata.update(_flatten_header(header))
        self.data = np.column_stack((dims[0], data))
        headers = [self.metadata["ScanDefinition.Regions.PAxis.Name"]]
        if len(dims) == 2:
            headers.extend([str(x) for x in dims[1]])
        else:
            headers.append(self.metadata["ScanDefinition.Channels.Name"])
        self.column_headers = headers
        self.setas = "xy"
        return self


class MaximusImage(ImageFile):
    """Provide a STXMImage like class for the Maximus Beamline."""

    _patterns = ["*.hdr", "*.xim"]
    mime_type = ["text/plain"]
    priority = 16

    def _load(self, *args, **kargs):
        """Load an ImageFile by calling the ImageArray method instead."""
        filename, args, kargs = get_filename(args, kargs)
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename
        pth = Path(self.filename)
        if pth.suffix != ".hdr":  # Passed a .xim or .xsp file in instead of the hdr file.
            pth = Path("_".join(str(pth).split("_")[:-1]) + ".hdr")
        stem = pth.parent / pth.stem

        try:
            hdr = _flatten_header(hdr_to_dict(pth))
            if hdr["ScanDefinition.Type"] != "Image Scan":
                raise StonerLoadError("Not an Maximus Single Image File")
        except (StonerLoadError, ValueError, TypeError, IOError) as err:
            raise StonerLoadError("Error loading as Maximus File") from err
        data = read_scan(stem)[1]
        self.metadata.update(hdr)
        self.image = data
        return self


class MaximusStackMixin:
    """Handle a stack of Maximus Images."""

    _defaults = {"type": MaximusImage, "pattern": "*.hdr"}

    def __init__(self, *args, **kargs):
        """Construct the attocube subclass of ImageStack."""
        args = list(args)
        if len(args) > 0:
            for ix, arg in enumerate(args):
                if isinstance(arg, Path):
                    args[ix] = str(arg)
        if len(args) > 0 and isinstance(args[0], string_types):
            root_name = args.pop(0)
            scan = SCAN_NO.search(root_name)
            if scan:
                scan = int(scan.groups()[0])
            else:
                scan = -1
        elif len(args) > 0 and isinstance(args[0], int):
            scan = args.pop(0)
            root_name = f"MPI_{scan:039d}"
        else:
            root_name = kargs.pop("root", None)
            scan = kargs.pop("scan", -1)

        super().__init__(*args, **kargs)

        self._common_metadata = TypeHintedDict()

        self.scan_no = scan

        if root_name:
            self._load(root_name)

        self._common_metadata["Scan #"] = scan

        self.compression = "gzip"
        self.compression_opts = 6

    def _load(self, *args, **kargs):
        """Load an ImageStack from either an hdf file or textfiles."""
        filename, args, kargs = get_filename(args, kargs)
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename
        pth = Path(self.filename)
        if h5py.is_hdf5(self.filename):
            return self.__class__.read_hdf5(self.filename)

        if pth.suffix != ".hdr":  # Passed a .xim or .xsp file in instead of the hdr file.
            pth = pth.parent / f"MPI_{self.scan_no:09d}.hdr"
        stem = pth.parent / pth.stem

        try:
            hdr = _flatten_header(hdr_to_dict(pth))
            if hdr["ScanDefinition.Type"] != "NEXAFS Image Scan":
                raise StonerLoadError("Not an Maximus ImageStack")
        except (StonerLoadError, ValueError, TypeError, IOError) as err:
            raise StonerLoadError("Error loading as Maximus File") from err

        metadata, stack, _ = read_scan(stem)
        self._common_metadata.update(_flatten_header(metadata))
        self._stack = stack
        self._names = [f"{stem}_a{ix:03d}" for ix in range(stack.shape[2])]
        self._sizes = np.ones((stack.shape[2], 2), dtype=int) * stack.shape[:2]
        for name, point in zip(self._names, self._common_metadata["ScanDefinition.StackAxis.Points"]):
            self._metadata.setdefault(name, {})
            self._metadata[name].update({self._common_metadata["ScanDefinition.StackAxis.Name"]: point})
        return self

    def _instantiate(self, idx):
        """Reconstructs the data type."""
        r, c = self._sizes[idx]
        if issubclass(
            self.type, ImageArray
        ):  # IF the underlying type is an ImageArray, then return as a view with extra metadata
            tmp = self._stack[:r, :c, idx].view(type=self.type)
        else:  # Otherwise it must be something with a data attribute
            tmp = self.type()  # pylint: disable=E1102
            tmp.data = self._stack[:r, :c, idx]
        tmp.metadata = deepcopy(self._common_metadata)
        tmp.metadata.update(self._metadata[self.__names__()[idx]])
        tmp.metadata["Scan #"] = self.scan_no
        tmp._fromstack = True
        return tmp

    def __clone__(self, other=None, attrs_only=False):
        """Do whatever is necessary to copy attributes from self to other.

        Note:
            We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!


        """
        other = super().__clone__(other, attrs_only)
        other._common_metadata = deepcopy(self._common_metadata)
        return other

    def _read_image(self, g):
        """Read an image array and return a member of the image stack."""
        if "image" not in g:
            _raise_error(g.parent, message=f"{g.name} does not have a signal dataset !")
        tmp = self.type()  # pylint: disable=E1102
        data = g["image"]
        if np.prod(np.array(data.shape)) > 0:
            tmp.image = data[...]
        else:
            tmp.image = [[]]
        metadata = g.require_group("metadata")
        typehints = g.get("typehints", None)
        if not isinstance(typehints, h5py.Group):
            typehints = dict()
        else:
            typehints = typehints.attrs
        for i in sorted(metadata.attrs):
            v = metadata.attrs[i]
            t = typehints.get(i, "Detect")
            if isinstance(v, string_types) and t != "Detect":  # We have typehints and this looks like it got exported
                tmp.metadata[f"{i}{{{t}}}".strip()] = f"{v}".strip()
            else:
                tmp[i] = metadata.attrs[i]
        tmp.filename = path.basename(g.name)
        return tmp

    def to_hdf5(self, filename=None):
        """Save the AttocubeScan to an hdf5 file."""
        mode = "r"
        if filename is None:
            filename = path.join(self.directory, f"MPI_{self.scan_no:09d}.hdf5")
        if isinstance(filename, Path):
            filename = str(filename)
        if filename is None or (isinstance(filename, bool) and not filename):  # now go and ask for one
            filename = self.__file_dialog("w")
            self.filename = filename
        if isinstance(filename, string_types):
            mode = "r+" if path.exists(filename) else "w"
        with HDFFileManager(filename, mode) as f:
            f.attrs["type"] = type(self).__name__
            f.attrs["module"] = type(self).__module__
            f.attrs["scan_no"] = self.scan_no
            f.attrs["groups"] = list(self.groups.keys())
            f.attrs["names"] = self._names
            if "common_metadata" in f.parent and "common_metadata" not in f:
                f["common_metadata"] = h5py.SoftLink(f.parent["common_metadata"].name)
                f["common_typehints"] = h5py.SoftLink(f.parent["common_typehints"].name)
            else:
                metadata = f.require_group("common_metadata")
                typehints = f.require_group("common_typehints")
                for k in self._common_metadata:
                    try:
                        typehints.attrs[k] = self._common_metadata._typehints[k]
                        metadata.attrs[k] = self._common_metadata[k]
                    except TypeError:
                        # We get this for trying to store a bad data type - fallback to metadata export to string
                        parts = self._common_metadata.export(k).split("=")
                        metadata.attrs[k] = "=".join(parts[1:])

            for g in self.groups:  # Recurse to save groups
                grp = f.require_group(g)
                self.groups[g].to_hdf5(grp)

            for ch in self._names:
                signal = f.require_group(ch)
                data = self[ch]
                signal.require_dataset(
                    "image",
                    data=data.data,
                    shape=data.shape,
                    dtype=data.dtype,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )
                metadata = signal.require_group("metadata")
                typehints = signal.require_group("typehints")
                for k in self._metadata[ch]:
                    try:
                        typehints.attrs[k] = data.metadata._typehints[k]
                        metadata.attrs[k] = data.metadata[k]
                    except TypeError:
                        # We get this for trying to store a bad data type - fallback to metadata export to string
                        parts = data.metadata.export(k).split("=")
                        metadata.attrs[k] = "=".join(parts[1:])
        return self

    @classmethod
    def read_hdf5(cls, filename, *args, **kargs):
        """Create a new instance from an hdf file."""
        self = cls(regrid=False)
        if filename is None or not filename:
            self.get_filename("r")
            filename = self.filename
        else:
            self.filename = filename
        with HDFFileManager(self.filename, "r") as f:
            self.scan_no = f.attrs["scan_no"]
            if "groups" in f.attrs:
                sub_grps = f.attrs["groups"]
            else:
                sub_grps = None
            if "names" in f.attrs:
                names = f.attrs["names"]
            else:
                names = []
            grps = list(f.keys())
            if "common_metadata" not in grps or "common_typehints" not in grps:
                _raise_error(f, message="Couldn;t find common metadata groups, something is not right here!")
            metadata = f["common_metadata"].attrs
            typehints = f["common_typehints"].attrs
            for i in sorted(metadata):
                v = metadata[i]
                t = typehints.get(i, "Detect")
                if (
                    isinstance(v, string_types) and t != "Detect"
                ):  # We have typehints and this looks like it got exported
                    self._common_metadata[f"{i}{{{t}}}".strip()] = f"{v}".strip()
                else:
                    self._common_metadata[i] = metadata[i]
            grps.remove("common_metadata")
            grps.remove("common_typehints")
            if sub_grps is None:
                sub_grps = grps
            for grp in sub_grps:
                if "type" in f[grp].attrs:
                    self.groups[grp] = cls.read_hdf5(f[grp], *args, **kargs)
                    continue
                g = f[grp]
                self.append(self._read_image(g))
            for grp in names:
                g = f[grp]
                self.append(self._read_image(g))
        return self


class MaximusStack(MaximusStackMixin, ImageStack):
    """Process an image scan stack from the Bessy Maximus beamline as an ImageStack subclass."""


def hdr_to_dict(filename, to_python=True):
    """Convert .hdr metadata file to  json or python dictionary.

    Args:
        filename (str):
            Name of file to read (can also be a pathlib.Path).

    Keyword Arguments:
        to_python (bool):
            If true, return a python dictionary, otherwise return a json text string.

    Returns:
        (dict or str):
            Either the header file as a python dictionary, or a json string.
    """
    bare = re.compile(r"([\s\{])([A-Za-z][A-Za-z0-9_]*)\s\:")  # Match for keys
    term = re.compile(r",\s*([\]\}])")  # match for extra , at the end of a dict or list
    nan = re.compile(r"([\-0-9\.]+\#QNAN)")  # Handle NaN values

    # Use oathlib to suck in the file
    with FileManager(filename, "r") as f:
        hdr = f.read()
    # Simple string replacements first
    stage1 = hdr.replace("=", ":").replace(";", ",").replace("(", "[").replace(")", "]")
    # Wrap in { }
    stage2 = f"{{{stage1}}}"
    # Regexp replacements next
    stage3 = bare.sub('\\1"\\2" :', stage2)
    stage4 = term.sub("\\n\\1", stage3)
    stage5 = nan.sub("NaN", stage4)

    if to_python:
        ret = _process_key(json.loads(stage5))
    else:  # orettyify the json
        ret = json.dumps(json.loads(stage5), indent=4)

    return ret


def read_scan(file_root):
    """Find the .hdr and .xim/,xsp files for a scan load them into memory.

    Args:
        file_root (str): This is the part of the hdr filename beore the extension.

    Returns:
        (dict,ndarray, (n-1d arrays)):
            Returns the metadata and an ndarray of the scan data and then n 1D arrays of the axes.
    """
    hdr = Path(str(file_root) + ".hdr")
    header = hdr_to_dict(hdr, to_python=True)
    scan_type = header["ScanDefinition"]["Type"]

    if "Image Scan" in scan_type:
        data, dims = _read_images(hdr.parent.glob(f"{hdr.stem}*.xim"), header)
    elif "Point Scan" in scan_type:
        data, dims = _read_pointscan(hdr.parent.glob(f"{hdr.stem}*.xsp"), header)
    else:
        raise ValueError(f"Unrecognised scan type {scan_type}")

    return header, data, dims


def _flatten_header(value):
    """Flatten nested dictionaries."""
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict):
        return value
    dels = []
    adds = {}
    for key, item in value.items():
        item = _flatten_header(item)
        if isinstance(item, dict):
            for item_key, item_val in item.items():
                adds[f"{key}.{item_key}"] = item_val
            dels.append(key)
    for key in dels:
        del value[key]
    value.update(adds)
    return value


def _process_key(value):
    """Carry out post loading processing of data structures."""
    if isinstance(value, dict):
        for key, val in value.items():
            value[key] = _process_key(val)
        return value
    if isinstance(value, list):
        if len(value) > 0 and isinstance(value[0], int) and len(value) == value[0] + 1:  # Lists prepended with size
            del value[0]
        a = np.array(value)
        if a.dtype.kind in ["f", "i", "u", "U"]:  # convert arrays to arrays
            return a
        value = [_process_key(v) for v in value]
    return value


def _read_images(files, header):
    """Read one or more .xim files and construct the data array.

    Args:
        files (glob): glob pattern of xim files to read.
        header (dict): contents of the .jdr file.

    Returns:
        data (ndarray): 2D or 3D data.
        dims (tuple of 1D arrays): 2 or 3 1D arrays corresponding to the dimensions of data.
    """
    xims = list(files)
    scandef = header["ScanDefinition"]
    region = scandef["Regions"][0]  # FIXME assumes a single region in the data
    if len(xims) > 1:
        data = np.stack([np.genfromtxt(x)[::-1] for x in xims]).T
    elif len(xims) == 1:
        data = np.genfromtxt(xims[0])[::-1]
    else:  # no files !
        raise IOError("No Images located")
    xpts = region["PAxis"]["Points"]
    ypts = region["QAxis"]["Points"]
    if data.ndim == 3:
        zpts = scandef["StackAxis"]["Points"]
        dims = (xpts, ypts, zpts)
    else:
        dims = (xpts, ypts)
    return data, dims


def _read_pointscan(files, header):
    """Read one or more .xsp files and construct the data array.

    Args:
        files (glob): glob pattern of xim files to read.
        header (dict): contents of the .jdr file.

    Returns:
        data (ndarray): 2D or 3D data.
        dims (tuple of 1D arrays): 2 or 3 1D arrays corresponding to the dimensions of data.
    """
    xsps = list(files)
    scandef = header["ScanDefinition"]
    region = scandef["Regions"][0]  # FIXME assumes a single region in the data
    if len(xsps) > 1:
        data = np.stack([np.genfromtxt(x)[:, 1] for x in xsps]).T
    elif len(xsps) == 1:
        data = np.genfromtxt(xsps[0])[:, 1]
    else:  # No files !
        raise IOError("No Spectra located")
    xpts = region["PAxis"]["Points"]
    if data.ndim == 2:
        zpts = scandef["StackAxis"]["Points"]
        dims = (xpts, zpts)
    else:
        dims = (xpts,)
    return data, dims


if __name__ == "__main__":
    # Test by reading all files
    for infile in Path(".").glob("*.hdr"):
        hdr, data, dims = read_scan(infile.stem)
