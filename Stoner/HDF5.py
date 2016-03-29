"""Stoner.HDF5 - Defines classes that use the hdf5 file format to store data on disc.

Classes include

* HDF5File - A :py:class:`Stoner.Code.DataFile` subclass that can save and load data from hdf5 files
* HDF5Folder - A :py:class:`Stoner.Folders.DataFolder` subclass that can save and load data from a single hdf5 file

"""
from Stoner.compat import *
import h5py
import numpy as _np_
from .Core import DataFile, StonerLoadError
from .Folders import DataFolder
import os.path as path


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
    compression = 'gzip'
    compression_opts = 6
    patterns = ["*.hdf", "*.hf5"]
    mime_type=["application/x-hdf"]

    #    def __init__(self,*args,**kargs):
    #       """Constructor to catch initialising with an h5py.File or h5py.Group
    #       """
    #       if len(args)>0:
    #           other=args[0]
    #           if isinstance(other,h5py.File) or isinstance(other,h5py.Group):
    #               super(HDF5File,self).__init__(**kargs)
    #               self.load(other,**kargs)
    #           else:
    #               super(HDF5File,self).__init__(*args,**kargs)

    def _load(self, filename=None, **kargs):
        """Loads data from a hdf5 file

        Args:
            h5file (string or h5py.Group): Either a string or an h5py Group object to load data from

        Returns:
            itself after having loaded the data
        """
        if filename is None or not filename:
            self.get_filename('r')
            filename = self.filename
        else:
            self.filename = filename
        if isinstance(filename, string_types):  #We got a string, so we'll treat it like a file...
            try:
                f = h5py.File(filename, 'r')
            except IOError:
                raise StonerLoadError("Failed to open {} as a n hdf5 file".format(filename))
        elif isinstance(filename, h5py.File) or isinstance(filename, h5py.Group):
            f = filename
        else:
            raise StonerLoadError("Couldn't interpret {} as a valid HDF5 file or group or filename".format(filename))
        if "type" not in f.attrs or bytes2str(
            f.attrs["type"]) != "HDF5File":  #Ensure that if we have a type attribute it tells us we're the right type !
            f.file.close()
            raise StonerLoadError("HDF5 group doesn't hold an HD5File")
        data = f["data"]
        if _np_.product(_np_.array(data.shape)) > 0:
            self.data = data[...]
        else:
            self.data = [[]]
        metadata = f.require_group('metadata')
        if "column_headers" in f.attrs:
            self.column_headers = f.attrs["column_headers"].astype("U")
            if isinstance(self.column_headers, string_types):
                self.column_headers = self.metadata.string_to_type(self.column_headers)
            self.column_headers = [bytes2str(x) for x in self.column_headers]
        else:
            raise StonerLoadError("Couldn't work out where my column headers were !")
        for i in metadata.attrs:
            self[i] = metadata.attrs[i]
        if "filename" in f.attrs:
            self.filename = f.attrs["filename"]
        elif isinstance(f, h5py.Group):
            self.filename = f.name
        else:
            self.filename = f.file.filename

        if isinstance(filename, string_types):
            f.file.close()
        return self

    def save(self, h5file=None):
        """Writes the current object into  an hdf5 file or group within a file in a
        fashion that is compatible with being loaded in again with the same class.

        Args:
            h5file (string or h5py.Group): Either a string, of h5py.File or h5py.Group object into which
                to save the file. If this is a string, the corresponding file is opened for
                writing, written to and save again.

        Returns
            A copy of the object
        """
        if h5file is None:
            h5file = self.filename
        if h5file is None or (isinstance(h5file, bool) and not h5file):  # now go and ask for one
            h5file = self.__file_dialog('w')
            self.filename = h5file
        if isinstance(h5file, string_types):
            f = h5py.File(h5file, 'w')
        elif isinstance(h5file, h5py.File) or isinstance(h5file, h5py.Group):
            f = h5file
        try:
            data = f.require_dataset("data",
                                     data=self.data,
                                     shape=self.data.shape,
                                     dtype=self.data.dtype,
                                     compression=self.compression,
                                     compression_opts=self.compression_opts)
            data = self.data
            metadata = f.require_group("metadata")
            for k in self.metadata:
                try:
                    metadata.attrs[k] = self[k]
                except TypeError:  # We get this for trying to store a bad data type - fallback to metadata export to string
                    parts = self.metadata.export(k).split('=')
                    metadata[parts[0]] = "=".join(parts[1:])
            f.attrs["column_headers"] = self.column_headers
            f.attrs["filename"] = self.filename
            f.attrs["type"] = "HDF5File"
        except Exception as e:
            if isinstance(h5file, str):
                f.file.close()
            raise e
        if isinstance(h5file, string_types):
            f.file.close()

        return self


class HGXFile(DataFile):
    """A subclass of DataFile for reading GenX HDF Files.

    These files typically have an extension .hgx. This class has been based on a limited sample
    of hgx files and so may not be sufficiently general to handle all cases.
    """

    priority=16
    pattern=["*.hgx"]
    mime_type=["application/x-hdf"]

    def _load(self, filename=None, *args, **kargs):
        """GenX HDF file loader routine.

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.
            """
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        try:
            f=h5py.File(filename)
            if "current" in f and "config" in f["current"]:
                pass
            else:
                f.close()
                raise StonerLoadError("Looks like an unexpected HDF layout!.")
        except IOError:
            raise StonerLoadError("Looks like an unexpected HDF layout!.")
        else:
            f.close()

        with h5py.File(self.filename, "r") as f:
            self.scan_group(f["current"],"")
            self.main_data(f["current"]["data"])


        return self

    def scan_group(self,grp,pth):
        """Recursively list HDF5 Groups."""

        if not isinstance(grp,h5py.Group):
            return None
        for i,x in enumerate(grp):
            if pth=="":
                new_pth=x
            else:
                new_pth=pth+"."+x
            if pth=="" and x=="data": # Special case for main data
                continue
            if isinstance(grp[x],type(grp)):
                self.scan_group(grp[x],new_pth)
            elif isinstance(grp[x],h5py.Dataset):
                self[new_pth]=grp[x].value
        return None

    def main_data(self,data_grp):
        """Work through the main data group and build something that looks like a numpy 2D array."""
        if not isinstance(data_grp,h5py.Group) or data_grp.name!="/current/data":
            raise StonerLoadError("HDF5 file not in expected format")
        datasets=data_grp["_counter"].value
        root=data_grp["datasets"]
        for ix in root: # Hack - iterate over all items in root, but actually data is in Groups not DataSets
            dataset=root[ix]
            if isinstance(dataset,h5py.Dataset):
                continue
            x=dataset["x"].value
            y=dataset["y"].value
            e=dataset["error"].value
            self&=x
            self&=y
            self&=e
            self.column_headers[-3]=dataset["x_command"].value
            self.column_headers[-2]=dataset["y_command"].value
            self.column_headers[-1]=dataset["error_command"].value
            self.column_headers=[str(x) for x in self.column_headers]



class HDF5Folder(DataFolder):
    """A sub class of :py:class:`Stoner.Folders.DataFolder` that provides a
    method to load and save data from a single HDF5 file with groups.

    See :py:class:`Stoner.Folders.DataFolder` for documentation on constructor.

    Datalayout consistns of sub-groups that are either instances of HDF5Files (i.e. have a type attribute that contains 'HDF5File')
    or are themsleves HDF5Folder instances (with a type attribute that reads 'HDF5Folder').

    """

    def __init__(self, *args, **kargs):
        """Constructor for the HDF5Folder Class.
        """
        self.File = None
        self.type = HDF5File

        super(HDF5Folder, self).__init__(*args, **kargs)

    def _dialog(self, message="Select Folder", new_directory=True, mode='r'):
        """Creates a file dialog box for working with

        Args:
            message (string): Message to display in dialog
            new_file (bool): True if allowed to create new directory

        Returns:
            A directory to be used for the file operation."""
        try:
            from enthought.pyface.api import FileDialog, OK
        except ImportError:
            from pyface.api import FileDialog, OK
        # Wildcard pattern to be used in file dialogs.
        file_wildcard = "hdf file (*.hdf5)|*.hdf5|Data file (*.dat)|\
        *.dat|All files|*"

        if mode == "r":
            mode2 = "open"
        elif mode == "w":
            mode2 = "save as"

        dlg = FileDialog(action=mode2, wildcard=file_wildcard)
        dlg.open()
        if dlg.return_code == OK:
            self.directory = dlg.path
            self.File = h5py.File(self.directory, mode)
            self.File.close()
            return self.directory
        else:
            return None

    def getlist(self, recursive=None, directory=None, flatten=None):
        """Reads the HDF5 File to construct a list of file HDF5File objects"""

        if recursive is None:
            recursive = self.recursive
        self.files = []
        self.groups = {}
        for d in [directory, self.directory, self.File, True]:
            if isinstance(d, bool) and d:
                d = self._dialog()
            directory = d
            if d is not None:
                break
        if directory is None:
            return None
        if isinstance(directory, string_types):
            self.directory = directory
            directory = h5py.File(directory, 'r')
            self.File = directory
        elif isinstance(directory, h5py.File) or isinstance(directory, h5py.Group):
            self.File = directory.file
            self.directory = self.File.filename
        #At this point directory contains an open h5py.File object, or possibly a group
        for obj in directory:
            obj = directory[obj]
            if isinstance(obj, h5py.Group) and "type" in obj.attrs:
                if obj.attrs["type"] == "HDF5File":
                    self.files.append(obj.name)
                elif obj.attrs["type"] == "HDF5Folder" and recursive:
                    self.groups[obj.name.split(path.sep)[-1]] = self.__class__(obj,
                                                                               pattern=self.pattern,
                                                                               type=self.type)
                else:
                    raise IOError("Found a group {} that isn't recognised".format(obj.name))

        return self

    def __read__(self, f):
        """Override the _-read method to handle pulling files from the HD5File"""
        if isinstance(f, DataFile):  # This is an already loaded DataFile
            tmp = f
            f = tmp.filename
        elif isinstance(f, h5py.Group) or isinstance(f, h5py.File):  # This is a HDF5 file or group
            tmp = HDF5File(f)
            f = f.name
        elif isinstance(f, str) or isinstance(
            f, unicode):  #This sis a string, so see if it maps to a path in the current File
            if f in self.File and "type" in self.File[f].attrs and self.File[f].attrs["type"] == "HDF5File":
                tmp = HDF5File(self.File[f])
            else:  # Otherwise fallback and try to laod from disc
                tmp = super(HDF5Folder, self).__read__(f)
        else:
            raise RuntimeError("Unable to workout how to read from {}".format(f))
        tmp["Loaded from"] = f
        if self.read_means:
            if len(tmp) == 0:
                pass
            elif len(tmp) == 1:
                for h in tmp.column_headers:
                    tmp[h] = tmp.column(h)[0]
            else:
                for h in tmp.column_headers:
                    tmp[h] = _np_.mean(tmp.column(h))

        return tmp

    def save(self, root=None):
        """Saves a load of files to a single HDF5 file, creating groups as it goes.

        Keyword Arguments:
            root (string): The name of the HDF5 file to save to if set to None, will prompt for a filename.

        Return:
            A list of group paths in the HDF5 file
        """

        if root is None:
            root = self._dialog(mode='w')
        elif isinstance(root, bool) and not root and isinstance(self.File, h5py.File):
            root = self.File.filename
            self.File.close()
        self.File = h5py.File(root, 'w')
        tmp = self.walk_groups(self._save)
        self.File.file.close()
        return tmp

    def _save(self, f, trail):
        """Create a virtual path of groups in the HDF5 file and save data.

        Args:
            f(DataFile):  A DataFile instance to save
            trail (list): The trail of groups

        Returns:
            The new filename of the saved DataFile.

        Ensure we have created the trail as a series of sub groups, then create a sub-groupfor the filename
        finally cast the DataFile as a HDF5File and save it, passing the new group as the filename which
        will ensure we create a sub-group in the main HDF5 file

        This routine is used by a walk_groups call - hence the prototype matches that required for
        :py:meth:`Stoner.Folders.DataFolder.walk_groups`.

        """
        tmp = self.File
        if not isinstance(f, DataFile):
            f = DataFile(f)
        for g in trail:
            if g not in tmp:
                tmp.create_group(g)
            tmp = tmp[g]
        tmp.create_group(f.filename)
        tmp = tmp[f.filename]
        f = HDF5File(f)
        f.save(tmp)
        return f.filename
