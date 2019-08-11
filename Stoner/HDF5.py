"""Stoner.HDF5 - Defines classes that use the hdf5 file format to store data on disc.

Classes include

* HDF5File - A :py:class:`Stoner.Code.DataFile` subclass that can save and load data from hdf5 files
* HDF5Folder - A :py:class:`Stoner.Folders.DataFolder` subclass that can save and load data from a single hdf5 file

It is only necessary to import this module for the subclasses of :py:class:`Stoner.Core.DataFile` to become available
to :py:class:`Stoner.Core.Data`.

"""
__all__ = ["HDF5File", "HDF5Folder", "HGXFile", "SLS_STXMFile", "STXMImage"]
from .compat import string_types, bytes2str, get_filedialog
import h5py
import numpy as _np_
from .Core import StonerLoadError, metadataObject, DataFile
from . import Data, DataFolder
from .Image.core import ImageFile
import os.path as path
import os


def _raise_error(f, message="Not a valid hdf5 file."):
    """Try to clsoe the filehandle f and raise a StonerLoadError."""
    try:
        f.file.close()
        raise StonerLoadError(message)
    except Exception:
        raise StonerLoadError(message)


def _open_filename(filename):
    """Examine a file to see if it is an HDF5 file and open it if so.

    Args:
        filename (str): Name of the file to open

    Returns:
        (f5py.Group): Valid h5py.Group containg data/

    Raises:
        StonerLoadError if not a valid file.
    """
    parts = filename.split(os.pathsep)
    filename = parts.pop(0)
    group = ""
    while len(parts) > 0:
        if not path.exists(path.join(filename, parts[0])):
            group = "/".join(parts)
        else:
            path.join(filename, parts.pop(0))

    with open(filename, "rb") as sniff:  # Some code to manaully look for the HDF5 format magic numbers
        sniff.seek(0, 2)
        size = sniff.tell()
        sniff.seek(0)
        blk = sniff.read(8)
        if not blk == b"\x89HDF\r\n\x1a\n":
            c = 0
            while sniff.tell() < size and len(blk) == 8:
                sniff.seek(512 * 2 ** c)
                c += 1
                blk = sniff.read(8)
                if blk == b"\x89HDF\r\n\x1a\n":
                    break
            else:
                raise StonerLoadError("Couldn't find the HD5 format singature block")
    try:
        f = h5py.File(filename, "r+")
        for grp in group.split("/"):
            if grp.strip() != "":
                f = f[grp]
    except IOError:
        _raise_error(f, message="Failed to open {} as a n hdf5 file".format(filename))
    except KeyError:
        _raise_error(f, message="Could not find group {} in file {}".format(group, filename))
    return f


class HDF5File(DataFile):

    """A sub class of DataFile that sores itself in a HDF5File or group.

    Args:
        args (tuple): Supplied arguments, only recognises one though !
        kargs (dict): Dictionary of keyword arguments

    If the first non-keyword arguement is not an h5py File or Group then
    initialises with a blank parent constructor and then loads data, otherwise,
    calls parent constructor.

    Datalayout is dead simple, the numerical data is in a dataset called *data*,
    metadata are attribtutes of a group called *metadata* with the keynames being the
    full name + typehint of the stanard DataFile metadata dictionary
    *column_headers* are an attribute of the root file/group
    *filename* is set from either an attribute called filename, or from the
    group name or from the hdf5 filename.
    The root has an attribute *type* that must by 'HDF5File' otherwise the load routine
    will refuse to load it. This is to try to avoid loading rubbish from random hdf files.
    """

    priority = 16
    compression = "gzip"
    compression_opts = 6
    patterns = ["*.hdf", "*.hf5"]
    mime_type = ["application/x-hdf"]

    def __init__(self, *args, **kargs):
        """Constructor to catch initialising with an h5py.File or h5py.Group
           """
        if args and isinstance(args[0], (h5py.File, h5py.Group)):
            args = list(args)
            grp = args.pop(0)
        else:
            grp = None
        super(HDF5File, self).__init__(*args, **kargs)
        if grp is not None:
            self._load(grp, **kargs)

    def _load(self, filename, **kargs):
        """Loads data from a hdf5 file

        Args:
            h5file (string or h5py.Group): Either a string or an h5py Group object to load data from

        Returns:
            itself after having loaded the data
        """
        if filename is None or not filename:
            self.get_filename("r")
            filename = self.filename
        else:
            self.filename = filename
        if isinstance(filename, string_types):  # We got a string, so we'll treat it like a file...
            f = _open_filename(filename)
        elif isinstance(filename, h5py.File) or isinstance(filename, h5py.Group):
            f = filename
        else:
            _raise_error(f, message="Couldn't interpret {} as a valid HDF5 file or group or filename".format(filename))
        if (
            "type" not in f.attrs or bytes2str(f.attrs["type"]) != "HDF5File"
        ):  # Ensure that if we have a type attribute it tells us we're the right type !
            _raise_error(f, message="HDF5 group doesn't hold an HD5File")
        data = f["data"]
        if _np_.product(_np_.array(data.shape)) > 0:
            self.data = data[...]
        else:
            self.data = [[]]
        metadata = f.require_group("metadata")
        typehints = f.get("typehints", None)
        if not isinstance(typehints, h5py.Group):
            typehints = dict()
        else:
            typehints = typehints.attrs
        if "column_headers" in f.attrs:
            self.column_headers = [x.decode("utf8") for x in f.attrs["column_headers"]]
            if isinstance(self.column_headers, string_types):
                self.column_headers = self.metadata.string_to_type(self.column_headers)
            self.column_headers = [bytes2str(x) for x in self.column_headers]
        else:
            raise StonerLoadError("Couldn't work out where my column headers were !")
        for i in sorted(metadata.attrs):
            v = metadata.attrs[i]
            t = typehints.get(i, "Detect")
            if isinstance(v, string_types) and t != "Detect":  # We have typehints and this looks like it got exported
                self.metadata["{}{{{}}}".format(i, t).strip()] = "{}".format(v).strip()
            else:
                self[i] = metadata.attrs[i]
        if isinstance(f, h5py.Group):
            if f.name != "/":
                self.filename = os.path.join(f.file.filename, f.name)
            else:
                self.filename = os.path.realpath(f.file.filename)
        else:
            self.filename = os.path.realpath(f.filename)
        if isinstance(filename, string_types):
            f.file.close()
        return self

    def save(self, filename=None, **kargs):
        """Writes the current object into  an hdf5 file or group within a file in afashion that is compatible with being loaded in again.

        Args:
            filename (string or h5py.Group): Either a string, of h5py.File or h5py.Group object into which
                to save the file. If this is a string, the corresponding file is opened for
                writing, written to and save again.

        Returns
            A copy of the object
        """
        if filename is None:
            filename = self.filename
        if filename is None or (isinstance(filename, bool) and not filename):  # now go and ask for one
            filename = self.__file_dialog("w")
            self.filename = filename
        if isinstance(filename, string_types):
            mode = "r+" if os.path.exists(filename) else "w"
            f = h5py.File(filename, mode)
        elif isinstance(filename, h5py.File) or isinstance(filename, h5py.Group):
            f = filename
        try:
            f.require_dataset(
                "data",
                data=self.data,
                shape=self.data.shape,
                dtype=self.data.dtype,
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
            metadata = f.require_group("metadata")
            typehints = f.require_group("typehints")
            for k in self.metadata:
                try:
                    typehints.attrs[k] = self.metadata._typehints[k]
                    metadata.attrs[k] = self[k]
                except TypeError:  # We get this for trying to store a bad data type - fallback to metadata export to string
                    parts = self.metadata.export(k).split("=")
                    metadata.attrs[k] = "=".join(parts[1:])
            f.attrs["column_headers"] = [x.encode("utf8") for x in self.column_headers]
            f.attrs["filename"] = self.filename
            f.attrs["type"] = "HDF5File"
        except Exception as e:
            if isinstance(filename, str):
                f.file.close()
            raise e
        if isinstance(f, h5py.File):
            self.filename = f.filename
        elif isinstance(f, h5py.Group):
            self.filename = f.file.filename
        else:
            self.filename = filename
        if isinstance(filename, string_types):
            f.file.close()

        return self


class HGXFile(DataFile):

    """A subclass of DataFile for reading GenX HDF Files.

    These files typically have an extension .hgx. This class has been based on a limited sample
    of hgx files and so may not be sufficiently general to handle all cases.
    """

    priority = 16
    pattern = ["*.hgx"]
    mime_type = ["application/x-hdf"]

    def _load(self, filename=None, *args, **kargs):
        """GenX HDF file loader routine.

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.
        """
        self.seen = []
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename
        with open(filename, "rb") as sniff:  # Some code to manaully look for the HDF5 format magic numbers
            sniff.seek(0, 2)
            sniff.seek(0)
            blk = sniff.read(8)
            if not blk == b"\x89HDF\r\n\x1a\n":
                c = 0
                while len(blk) == 8:
                    sniff.seek(512 * 2 ** c)
                    c += 1
                    blk = sniff.read(8)
                    if blk == b"\x89HDF\r\n\x1a\n":
                        break
                else:
                    raise StonerLoadError("Couldn't find the HD5 format singature block")

        try:
            with h5py.File(filename) as f:
                if "current" in f and "config" in f["current"]:
                    pass
                else:
                    raise StonerLoadError("Looks like an unexpected HDF layout!.")
                self.scan_group(f["current"], "")
                self.main_data(f["current"]["data"])
        except IOError:
            raise StonerLoadError("Looks like an unexpected HDF layout!.")
        return self

    def scan_group(self, grp, pth):
        """Recursively list HDF5 Groups."""
        if pth in self.seen:
            return None
        else:
            self.seen.append(pth)
        if not isinstance(grp, h5py.Group):
            return None
        if self.debug:
            if self.debug:
                print("Scanning in {}".format(pth))
        for x in grp:
            if pth == "":
                new_pth = x
            else:
                new_pth = pth + "." + x
            if pth == "" and x == "data":  # Special case for main data
                continue
            if isinstance(grp[x], type(grp)):
                self.scan_group(grp[x], new_pth)
            elif isinstance(grp[x], h5py.Dataset):
                y = grp[x][...]
                self[new_pth] = y
        return None

    def main_data(self, data_grp):
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
            self &= x
            self &= y
            self &= e
            self.column_headers[-3] = bytes2str(dataset["x_command"][()])
            self.column_headers[-2] = bytes2str(dataset["y_command"][()])
            self.column_headers[-1] = bytes2str(dataset["error_command"][()])
            self.column_headers = [str(ix) for ix in self.column_headers]


class HDF5FolderMixin(object):

    """A mixin class for :py:class:`Stoner.Folders.DataFolder` that provides a method to load and save data from a single HDF5 file with groups.

    See :py:class:`Stoner.Folders.DataFolder` for documentation on constructor.

    Datalayout consistns of sub-groups that are either instances of HDF5Files (i.e. have a type attribute that contains 'HDF5File')
    or are themsleves HDF5Folder instances (with a type attribute that reads 'HDF5Folder').
    """

    def __init__(self, *args, **kargs):
        """Constructor for the HDF5Folder Class."""
        self.File = None
        super(HDF5FolderMixin, self).__init__(*args, **kargs)

    def __getter__(self, name, instantiate=True):
        """Loads the specified name from a file on disk.

        Parameters:
            name (key type): The canonical mapping key to get the dataObject. By default
                the baseFolder class uses a :py:class:`regexpDict` to store objects in.

        Keyword Arguments:
            instatiate (bool): IF True (default) then always return a :py:class:`Stoner.Core.Data` object. If False,
                the __getter__ method may return a key that can be used by it later to actually get the
                :py:class:`Stoner.Core.Data` object.

        Returns:
            (metadataObject): The metadataObject
        """
        try:
            return super(HDF5FolderMixin, self).__getter__(name, instantiate=instantiate)
        except (AttributeError, IndexError, KeyError, OSError, IOError) as err:
            if self.debug:
                print(err)
        names = list(os.path.split(name))
        if names[0] == "/":  # Prune leading ./
            names = names[1:]
        if self.File is None:
            closeme = True
            self.File = h5py.File(self.directory, "r+")
        else:
            closeme = False
        grp = self.File
        while len(names) > 0:
            next_group = names.pop(0)
            if next_group not in grp:
                raise IOError("Cannot find {} in {}".format(name, self.File.filename))
            grp = grp[next_group]
        tmp = self.loader(grp)
        tmp.filename = grp.name
        tmp = self.on_load_process(tmp)
        tmp = self._update_from_object_attrs(tmp)
        self.__setter__(name, tmp)
        if closeme:
            self.File.close()
            self.File = None
        return tmp

    def _dialog(self, message="Select Folder", new_directory=True, mode="r+"):
        """Creates a file dialog box for working with

        Args:
            message (string): Message to display in dialog
            new_file (bool): True if allowed to create new directory

        Returns:
            A directory to be used for the file operation.
        """
        file_wildcard = "hdf file (*.hdf5)|*.hdf5|Data file (*.dat)|\
        *.dat|All files|*"

        if mode == "r":
            mode2 = "Open HDF file"
        elif mode == "w":
            mode2 = "Save HDF file as"

        dlg = get_filedialog(
            "file", title=mode2, filetypes=file_wildcard, message=message, mustexist=not new_directory
        )
        if len(dlg) != 0:
            self.directory = dlg
            self.File = h5py.File(self.directory, mode)
            self.File.close()
            return self.directory
        else:
            return None

    def _visit_func(self, name, obj):
        """Walker of the HDF5 tree."""
        if isinstance(obj, h5py.Group) and "type" in obj.attrs:
            cls = globals()[obj.attrs["type"]]
            if issubclass(self.loader, cls):
                self.__setter__(obj.name, obj.name)
            elif obj.attrs["type"] == "HDF5Folder" and self.recursive:
                self.groups[obj.name.split(path.sep)[-1]] = self.__class__(
                    obj, pattern=self.pattern, type=self.type, recursive=self.recursive, loader=self.loader
                )

    def close(self):
        """Close the cirrent hd5 file."""
        if isinstance(self.File, h5py.File):
            self.File.close()
            self.File = None
        else:
            raise IOError("HDF5 File not open!")

    def getlist(self, recursive=None, directory=None, flatten=False):
        """Reads the HDF5 File to construct a list of file HDF5File objects"""
        if recursive is None:
            recursive = self.recursive
        self.files = []
        self.groups = {}
        closeme = True
        for d in [directory, self.directory, self.File, True]:
            if isinstance(d, bool) and d:
                d = self._dialog()
            directory = d
            if d is not None:
                break
        if directory is None:
            return None
        if isinstance(directory, string_types):
            try:
                self.directory = directory
                directory = h5py.File(directory, "r+")
                self.File = directory
                closeme = True
            except OSError:
                return super(HDF5FolderMixin, self).getlist(recursive, directory, flatten)
        elif isinstance(directory, h5py.File) or isinstance(directory, h5py.Group):  # Bug out here
            self.File = directory.file
            self.directory = self.File.filename
            closeme = False
        # At this point directory contains an open h5py.File object, or possibly a group
        directory.visititems(self._visit_func)
        if flatten:
            self.flatten()
        if closeme:
            self.File.close()
            self.File = None
        return self

    def save(self, root=None):
        """Saves a load of files to a single HDF5 file, creating groups as it goes.

        Keyword Arguments:
            root (string): The name of the HDF5 file to save to if set to None, will prompt for a filename.

        Return:
            A list of group paths in the HDF5 file
        """
        closeme = False
        if root is None and isinstance(self.File, h5py.File):
            root = self.File
        elif root is None and not isinstance(self.File, h5py.File):
            root = h5py.File(self.directory)
            self.File = root
            closeme = True
        if root is None or (isinstance(root, bool) and not root):
            # now go and ask for one
            root = self._dialog()
            root = h5py.File(root)
            self.File = root
            closeme = True
        if isinstance(root, string_types):
            mode = "r+" if path.exists(root) else "w"
            root = h5py.File(root, mode)
            self.File = root
            closeme = True
        if not isinstance(root, (h5py.File, h5py.Group)):
            raise IOError("Can't save Folder without an HDF5 file or Group!")
        # root should be an open h5py file
        root.attrs["type"] = "HDF5Folder"
        for ix, obj in enumerate(self):
            name = os.path.basename(getattr(obj, "filename", "obj-{}".format(ix)))
            if name not in root:
                root.create_group(name)
            name = root[name]
            self.loader(obj).save(name)
        for grp in self.groups:
            if grp not in root:
                root.create_group(grp)
            self.groups[grp].save(root[grp])

        if closeme and self.File is not None:
            self.File.close()
            self.File = None
        return self


class HDF5Folder(HDF5FolderMixin, DataFolder):

    """Just enforces the loader attriobute to be an HDF5File!"""

    def __init__(self, *args, **kargs):
        self.loader = HDF5File
        super(HDF5Folder, self).__init__(*args, **kargs)


class SLS_STXMFile(DataFile):

    """Load images from the Swiss Light Source Pollux beamline"""

    priority = 16
    compression = "gzip"
    compression_opts = 6
    patterns = ["*.hdf"]
    mime_type = ["application/x-hdf"]

    def _load(self, filename, *args, **kargs):
        """Loads data from a hdf5 file

        Args:
            h5file (string or h5py.Group): Either a string or an h5py Group object to load data from

        Returns:
            itself after having loaded the data
        """
        if filename is None or not filename:
            self.get_filename("r")
            filename = self.filename
        else:
            self.filename = filename
        if isinstance(filename, string_types):  # We got a string, so we'll treat it like a file...
            try:
                f = h5py.File(filename, "r+")
            except IOError:
                raise StonerLoadError("Failed to open {} as a n hdf5 file".format(filename))
        elif isinstance(filename, h5py.File) or isinstance(filename, h5py.Group):
            f = filename
        else:
            raise StonerLoadError("Couldn't interpret {} as a valid HDF5 file or group or filename".format(filename))
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
        if _np_.product(_np_.array(data.shape)) > 0:
            self.data = data[...]
        else:
            self.data = [[]]
        self.scan_meta(root)
        if "file_name" in f.attrs:
            self["original filename"] = f.attrs["file_name"]
        elif isinstance(f, h5py.Group):
            self["original filename"] = f.name
        else:
            self["original filename"] = f.file.filename

        if isinstance(filename, string_types):
            f.file.close()
        self["Loaded from"] = self.filename

        if "instrument.sample_x.data" in self.metadata:
            self.metadata["actual_x"] = self.metadata["instrument.sample_x.data"].reshape(self.shape)
        if "instrument.sample_y.data" in self.metadata:
            self.metadata["actual_y"] = self.metadata["instrument.sample_y.data"].reshape(self.shape)
        self.metadata["sample_x"], self.metadata["sample_y"] = _np_.meshgrid(
            self.metadata["counter0.sample_x"], self.metadata["counter0.sample_y"]
        )

        return self

    def scan_meta(self, group):
        """Scan the HDF5 Group for atributes and datasets and sub groups and recursively add them to the metadata."""
        root = ".".join(group.name.split("/")[2:])
        for name, thing in group.items():
            parts = thing.name.split("/")
            name = ".".join(parts[2:])
            if isinstance(thing, h5py.Group):
                self.scan_meta(thing)
            elif isinstance(thing, h5py.Dataset):
                if len(thing.shape) > 1:
                    continue
                if _np_.product(thing.shape) == 1:
                    self.metadata[name] = thing[0]
                else:
                    self.metadata[name] = thing[...]
        for attr in group.attrs:
            self.metadata["{}.{}".format(root, attr)] = group.attrs[attr]


class STXMImage(ImageFile):

    """An instance of KerrArray that will load itself from a Swiss Light Source STXM image"""

    _reduce_metadata = False

    def __init__(self, *args, **kargs):
        """Construct a STXMImage file.

        Keyword Args:
            regrid (bool): If set True, the gridimage() method is automatically called to re-grid the image to known co-ordinates."""
        regrid = kargs.pop("regrid", False)
        if len(args) > 0 and isinstance(args[0], string_types):
            d = SLS_STXMFile(args[0])
            args = args[1:]
        else:
            d = Data()
        super(STXMImage, self).__init__(*args[1:], **kargs)
        self.image = d.data
        self.metadata.update(d.metadata)
        self.filename = d.filename
        if isinstance(regrid, tuple):
            self.gridimage(*regrid)
        elif isinstance(regrid, dict):
            self.gridimage(**regrid)
        elif regrid:
            self.gridimage()

    def __floordiv__(self, other):
        if isinstance(other, metadataObject):
            if self["collection.polarization.value"] > 0 and other["collection.polarization.value"] < 0:
                plus, minus = self, other
            elif self["collection.polarization.value"] < 0 and other["collection.polarization.value"] > 0:
                plus, minus = other, self
            else:
                raise ValueError("XMCD Ratio can only be found from a positive and minus image")
            ret = self.clone
            ret.image = (plus.image - minus.image) / (plus.image + minus.image)
            return ret
        else:
            raise TypeError("Can only do XMCD calculation with another STXMFile")
