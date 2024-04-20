"""Stoner.HDF5 - Defines classes that use the hdf5 file format to store data on disc.

Classes include

* HDF5File - A :py:class:`Stoner.Code.DataFile` subclass that can save and load data from hdf5 files
* HDF5Folder - A :py:class:`Stoner.Folders.DataFolder` subclass that can save and load data from a single hdf5 file

It is only necessary to import this module for the subclasses of :py:class:`Stoner.Core.DataFile` to become available
to :py:class:`Stoner.Core.Data`.

"""
__all__ = ["HDF5Folder"]
import os.path as path
import os
import h5py

from .compat import get_filedialog, path_types
from .folders import DataFolder


class HDF5FolderMixin:

    """Provides a method to load and save data from a single HDF5 file with groups.

    See :py:class:`Stoner.Folders.DataFolder` for documentation on constructor.

    Datalayout consistns of sub-groups that are either instances of HDF5Files (i.e. have a type attribute that
    contains 'HDF5File') or are themsleves HDF5Folder instances (with a type attribute that reads 'HDF5Folder').
    """

    def __init__(self, *args, **kargs):
        """Initialise the File aatribute."""
        self.File = None
        super().__init__(*args, **kargs)

    def __getter__(self, name, instantiate=True):
        """Load the specified name from a file on disk.

        Parameters:
            name (key type):
                The canonical mapping key to get the dataObject. By default
                the baseFolder class uses a :py:class:`regexpDict` to store objects in.

        Keyword Arguments:
            instantiate (bool):
                If True (default) then always return a :py:class:`Stoner.Core.Data` object. If False,
                the __getter__ method may return a key that can be used by it later to actually get the
                :py:class:`Stoner.Core.Data` object.

        Returns:
            (metadataObject):
                The metadataObject
        """
        try:
            return super().__getter__(name, instantiate=instantiate)
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
                raise IOError(f"Cannot find {name} in {self.File.filename}")
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
        """Create a file dialog box for working with.

        Args:
            message (string):
                Message to display in dialog
            new_file (bool):
                True if allowed to create new directory

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
        return None

    def _visit_func(self, name, obj):  # pylint: disable=unused-argument
        """Walker of the HDF5 tree."""
        if isinstance(obj, h5py.Group) and "type" in obj.attrs:
            cls = globals()[obj.attrs["type"]]
            if issubclass(self.loader, cls):
                self.__setter__(obj.name, obj.name)
            elif obj.attrs["type"] == "HDF5Folder" and self.recursive:
                self.groups[obj.name.split(path.sep)[-1]] = type(self)(
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
        """Read the HDF5 File to construct a list of file HDF5File objects."""
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
        if isinstance(directory, path_types):
            try:
                self.directory = directory
                directory = h5py.File(directory, "r+")
                self.File = directory
                closeme = True
            except OSError:
                return super().getlist(recursive, directory, flatten)
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
        """Save a load of files to a single HDF5 file, creating groups as it goes.

        Keyword Arguments:
            root (string):
                The name of the HDF5 file to save to if set to None, will prompt for a filename.

        Return:
            A list of group paths in the HDF5 file
        """
        closeme = False
        if root is None and isinstance(self.File, h5py.File):
            root = self.File
        elif root is None and not isinstance(self.File, h5py.File):
            root = h5py.File(self.directory, mode="w")
            self.File = root
            closeme = True
        if root is None or (isinstance(root, bool) and not root):
            # now go and ask for one
            root = self._dialog()
            root = h5py.File(root, mode="a")
            self.File = root
            closeme = True
        if isinstance(root, path_types):
            mode = "r+" if path.exists(root) else "w"
            root = h5py.File(root, mode)
            self.File = root
            closeme = True
        if not isinstance(root, (h5py.File, h5py.Group)):
            raise IOError("Can't save Folder without an HDF5 file or Group!")
        # root should be an open h5py file
        root.attrs["type"] = "HDF5Folder"
        for ix, obj in enumerate(self):
            name = os.path.basename(getattr(obj, "filename", f"obj-{ix}"))
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

    """Just enforces the loader attriobute to be an HDF5File."""

    def __init__(self, *args, **kargs):
        """Ensure the loader routine is set for HDF5Files."""
        self.loader = "HDF5File"
        super().__init__(*args, **kargs)
