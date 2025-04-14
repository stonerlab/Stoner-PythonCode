# -*- coding: utf-8 -*-
"""Module to work with scan files from an AttocubeSPM running Daisy."""
__all__ = ["AttocubeScan"]
import pathlib
from os import path
from copy import deepcopy
from glob import glob
import re
import importlib

from numpy import genfromtxt, linspace, meshgrid, array, prod
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
import h5py

from ..compat import string_types, bytes2str
from ..core.base import TypeHintedDict
from ..Image import ImageStack, ImageFile, ImageArray
from ..HDF5 import HDFFileManager
from ..tools.file import FileManager, get_filename
from ..core.exceptions import StonerLoadError

PARAM_RE = re.compile(r"^([\d\\.eE\+\-]+)\s*([\%A-Za-z]\S*)?$")
SCAN_NO = re.compile(r"SC_(\d+)")


def parabola(X, cx, cy, a, b, c):
    """Parabola in the X-Y plane for levelling an image."""
    x, y = X
    return a * (x - cx) ** 2 + b * (y - cy) ** 2 + c


def plane(X, a, b, c):
    """Plane equation for levelling an image."""
    x, y = X
    return a * x + b * y + c


def _raise_error(openfile, message=""):
    """Raise a StonerLoadError after trying to close file."""
    try:
        raise StonerLoadError(message)
    finally:
        try:
            openfile.close()
        except (AttributeError, TypeError, ValueError, IOError):
            pass


class AttocubeScanMixin:
    """Provides the specialist methods for dealing with Attocube SPM scan files.

    See :py:class:`AttocubeScan` for details."""

    def __init__(self, *args, **kargs):
        """Construct the attocube subclass of ImageStack."""
        args = list(args)
        if len(args) > 0:
            for ix, arg in enumerate(args):
                if isinstance(arg, pathlib.PurePath):
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
            root_name = f"SC_{scan:03d}"
        else:
            root_name = kargs.pop("root", None)
            scan = kargs.pop("scan", -1)

        regrid = kargs.pop("regrid", False)

        super().__init__(*args, **kargs)

        self._common_metadata = TypeHintedDict()

        if root_name:
            self._load_files(root_name, regrid)

        self.scan_no = scan

        self._common_metadata["Scan #"] = scan

        self.compression = "gzip"
        self.compression_opts = 6

    def __clone__(self, other=None, attrs_only=False):
        """Do whatever is necessary to copy attributes from self to other.

        Note:
            We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!


        """
        other = super().__clone__(other, attrs_only)
        other._common_metadata = deepcopy(self._common_metadata)
        return other

    def __getitem__(self, name):
        """Allow scans to be got by channel name as well as normal indexing."""
        if isinstance(name, string_types):
            for ix, ch in enumerate(self.channels):
                if name in ch:
                    return self[ix]
        return super().__getitem__(name)

    @property
    def channels(self):
        """Get the list of channels saved in the scan."""
        if len(self) > 0:
            return self.metadata.slice("display", values_only=True)
        return []

    def _load(self, *args, **kargs):
        """Load data from a hdf5 file.

        Args:
            h5file (string or h5py.Group):
                Either a string or an h5py Group object to load data from

        Returns:
            itself after having loaded the data
        """
        filename, args, kargs = get_filename(args, kargs)
        if filename is None or not filename:
            self.get_filename("r")
            filename = self.filename
        else:
            self.filename = filename
        with HDFFileManager(self.filename, "r") as f:
            if "type" not in f.attrs:
                _raise_error(f, message="HDF5 Group does not specify the type attribute used to check we can load it.")
            typ = bytes2str(f.attrs["type"])
            if typ != type(self).__name__ and "module" not in f.attrs:
                _raise_error(
                    f,
                    message=f"HDF5 Group is not a {type(self).__name__} and does not specify a module to use to load.",
                )
            loader = None
            if typ == type(self).__name__:
                loader = getattr(type(self), "read_hdf5")
            else:
                mod = importlib.import_module(bytes2str(f.attrs["module"]))
                cls = getattr(mod, typ)
                loader = getattr(cls, "read_hdf5")
            if loader is None:
                raise StonerLoadError("Could not et loader for {bytes2str(f.attrs['module'])}.{typ}")

        return loader(f, *args, **kargs)

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

    def _load_files(self, root_name, regrid):
        """Build the image stack from a stack of files.

        Args:
            root_name (str):
                The scan prefix e.g. SC_###
            regrid (bool):
                Use the X and Y positions if available to regrid the data.
        """
        if not path.exists(path.join(self.directory, f"{root_name}-Parameters.txt")):
            return False
        self._load_parameters(root_name)
        for data in glob(path.join(self.directory, f"{root_name}*.asc")):
            if data.endswith("fwd.asc"):
                if "fwd" not in self.groups:
                    self.add_group("fwd")
                target = self.groups["fwd"]
            elif data.endswith("bwd.asc"):
                if "bwd" not in self.groups:
                    self.add_group("bwd")
                target = self.groups["bwd"]
            else:
                target = self
            target._load_asc(data)
        if regrid:
            if "fwd" in self.groups:
                self.groups["fwd"].regrid(in_place=True)
            if "bwd" in self.groups:
                self.groups["bwd"].regrid(in_place=True)
            if not self.groups:
                self.regrid(in_place=True)
        return self

    def _load_parameters(self, root_name):
        """Load the scan parameters text file.

        Args:
            root_name (str):
                The scan prefix e.g. SC_###

        Returns:
            self:
                The modififed scan stack.
        """
        filename = path.join(self.directory, f"{root_name}-Parameters.txt")
        with FileManager(filename, "r") as parameters:
            if not parameters.readline().startswith("Daisy Parameter Snapshot"):
                raise IOError("Parameters file exists but does not have correct header")
            for line in parameters:
                if not line.strip():
                    continue
                parts = [x.strip() for x in line.strip().split(":")]
                key = parts[0]
                value = ":".join(parts[1:])
                units = PARAM_RE.match(value)
                if units and units.groups()[1]:
                    key += f" [{units.groups()[1]}]"
                    value = units.groups()[0]
                self._common_metadata[key] = value
        return self

    def _load_asc(self, filename):
        """Load a single scan file from ascii data."""
        with FileManager(filename, "r") as data:
            if not data.readline().startswith("# Daisy frame view snapshot"):
                raise ValueError(f"{filename} lacked the correct header line")
            tmp = ImageFile()
            for line in data:
                if not line.startswith("# "):
                    break
                parts = [x.strip() for x in line[2:].strip().split(":")]
                key = parts[0]
                value = ":".join(parts[1:])
                units = PARAM_RE.match(value)
                if units and units.groups()[1]:
                    key += f" [{units.groups()[1]}]"
                    value = units.groups()[0]
                tmp.metadata[key] = value
        xpx = tmp["x-pixels"]
        ypx = tmp["y-pixels"]
        metadata = tmp.metadata
        tmp.image = genfromtxt(filename).reshape((xpx, ypx))
        tmp.metadata = metadata
        tmp.filename = tmp["display"]
        self.append(tmp)
        return self

    def _read_signal(self, g):
        """Read a signal array and return a member of the image stack."""
        if "signal" not in g:
            raise StonerLoadError(f"{g.name} does not have a signal dataset !")
        tmp = self.type()  # pylint: disable=E1102
        data = g["signal"]
        if prod(array(data.shape)) > 0:
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

    def regrid(self, **kargs):
        """Regrid the data sets based on PosX and PosY channels.

        Keyword Parameters:
            x_range, y_range (tuple of start, stop, points):
                Range of x-y co-rdinates to regrid the data to. Used as an argument to :py:func:`np.linspace` to
                generate the coordinate
                vector.
            in_place (bool):
                If True then replace the existing datasets with the regridded data, otherwise create a new copy
                of the scan object. Default is False.

        Returns:
            (AttocubeScan):
                Scan object with regridded data. May be the same as the source object if in_place is True.
        """
        if not kargs.get("in_place", False):
            new = self.clone
        else:
            new = self
        try:
            x = self["PosX"]
            y = self["PosY"]
        except KeyError:  # Can't get X and Y data
            return new

        xrange = kargs.pop("x_range", (x[:, 0].max(), x[:, -1].min(), x.shape[1]))
        yrange = kargs.pop("y_range", (y[0].max(), y[-1].min(), y.shape[0]))
        nx, ny = meshgrid(linspace(*xrange), linspace(*yrange))
        for data in self.channels:
            if "PosX" in data or "PosY" in data:
                continue
            z = self[data]
            nz = griddata((x.ravel(), y.ravel()), z.ravel(), (nx.ravel(), ny.ravel()), method="cubic").reshape(
                nx.shape
            )
            new[data].data = nz
        new["PosX"].data = nx
        new["PosY"].data = ny

        return new

    def level_image(self, method="plane", signal="Amp"):
        """Remove a background signla by fitting an appropriate function.

        Keyword Arguments:
            method (str or callable):
                Eirther the name of a fitting function in the global scope, or a callable. *plane* and *parabola*
                are already defined.
            signal (str):
                The name of the dataset to be flattened. Defaults to the Amplitude signal

        Returns:
            (AttocubeScan):
                The current scan object with the data modified.
        """
        if isinstance(method, string_types):
            method = globals()[method]
        if not callable(method):
            raise ValueError("Could not get a callable method to flatten the data")
        data = self[signal]
        ys, xs = data.shape
        X, Y = meshgrid(linspace(-1, 1, xs), linspace(-1, 1, ys))
        Z = data.data
        X = X.ravel()
        Y = Y.ravel()
        Z = Z.ravel()

        popt = curve_fit(method, (X, Y), Z)[0]

        nZ = method((X, Y), *popt)
        Z -= nZ

        data.data = Z.reshape(xs, ys)
        return self

    def to_hdf5(self, filename=None):
        """Save the AttocubeScan to an hdf5 file."""
        mode = ""
        if filename is None:
            filename = path.join(self.directory, f"SC_{self.scan_no:03d}.hdf5")
        if isinstance(filename, pathlib.PurePath):
            filename = str(filename)
        if filename is None or (isinstance(filename, bool) and not filename):  # now go and ask for one
            filename = self.__file_dialog("w")
        if isinstance(filename, string_types):
            mode = "r+" if path.exists(filename) else "w"
        self.filename = filename
        with HDFFileManager(self.filename, mode=mode) as f:
            f.attrs["type"] = type(self).__name__
            f.attrs["module"] = type(self).__module__
            f.attrs["scan_no"] = self.scan_no
            f.attrs["groups"] = list(self.groups.keys())
            f.attrs["channels"] = list(self.channels)
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

            for ch in self.channels:
                signal = f.require_group(ch)
                data = self[ch]
                signal.require_dataset(
                    "signal",
                    data=data.data,
                    shape=data.shape,
                    dtype=data.dtype,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )
                metadata = signal.require_group("metadata")
                typehints = signal.require_group("typehints")
                for k in [x for x in data.metadata if x not in self._common_metadata]:
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
        with HDFFileManager(filename, "r") as f:
            self.scan_no = f.attrs["scan_no"]
            if "groups" in f.attrs:
                sub_grps = f.attrs["groups"]
            else:
                sub_grps = None
            if "channels" in f.attrs:
                channels = f.attrs["channels"]
            else:
                channels = []
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
                self.append(self._read_signal(g))
            for grp in channels:
                g = f[grp]
                self.append(self._read_signal(g))
        return self


class AttocubeScan(AttocubeScanMixin, ImageStack):
    """An ImageStack subclass that can load scans from the AttocubeScan SPM System.

    AttocubeScan represents a scan from an Attocube SPM system as a 3D stack of scan data with
    associated metadata. Indexing the AttocubeScan with either an integer or a partial match to one the
    signals saved in the scan will pull out that particular scan as a :py:class:`Stoner.Image.ImageFile`.

    If the scan was a dual pass scan with forwards and backwards data, then the root AttocubeScan will
    contain the common metadata derived from the Scan parameters and then two sub-stacks that represent
    the forwards ('fwd') and backwards ('bwd') scans.

    The AttocubeScan constructor will with take a *rrot name* of the scan - e.g. "SC_099" or alternatively
    a scan number integer. It will then look in the stack's directory for matching files and builds the scan stack
    from them. Currently, it uses the scan parameters.txt file and any ASCII stream files .asc files. 2D linescans
    are not currently supported or imported.

    The native file format for an AttocubeScan is an HDF5 file with a particilar structure. The stack is saved into
    an HDF5 group which then has a *type* and *module* attribute that specifies the class and module pf the Python
    object that created the group - sof for an AttocubeScan, the type attribute is *AttocubeScan*.

    There is a class method :py:meth:`AttocubeSca.read_hdf5` to read the stack from the HDSF format and an instance
    method :py:meth:`AttocubeScan.to_hdf` that will save to either a new or existing HDF file format.

    The class provides other methods to regrid and flatten images and may gain other capabilities in the future.

    Todo:
        Implement load and save to/from multipage TIFF files.

    Attrs:
        scan_no (int):
            The scan number as defined in the Attocube software.
        compression (str):
            The HDF5 compression algorithm to use when writing files
        compression_opts (int):
            The lelbel of compression to use (depends on compression algorithm)
    """
