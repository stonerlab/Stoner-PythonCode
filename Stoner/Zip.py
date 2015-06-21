# -*- coding: utf-8 -*-
"""
Stoner.Zip module - sipport reading DataFile like objects into and outof standard zip files.

Classes Include

* ZipFile - A :py:class:`Stoner.Code.DataFile` subclass that can save and load data from a zip files
* ZipFolder - A :py:class:`Stoner.Folders.DataFolder` subclass that can save and load data from a single zip file

Created on Tue Jan 13 16:39:51 2015

@author: phygbu
"""

from Stoner.compat import *
import zipfile as zf
import zlib
import itertools
import numpy as _np_
from .Core import DataFile,StonerLoadError
from .Folders import DataFolder
import os.path as path


def test_is_zip(filename, member=""):
    """Recursively searches for a zipfile in the tree."""
    if not filename or filename == "":
        return False
    elif zf.is_zipfile(filename):
        return filename, member
    else:
        part = path.basename(filename)
        newfile = path.dirname(filename)
        if newfile == filename:  #reached the end of the line
            part = filename
            newfile = ""
        if member != "":
            newmember = path.join(part, member)
        else:
            newmember = part
        return test_is_zip(newfile, newmember)


class ZipFile(DataFile):
    """A sub class of DataFile that sores itself in a zip file.

    Methods:
        _load(): prorivate method to allow :py:class:`Stoner.Core.DataFile` to load
                    a file from zip file
        save(): Save a dataset to an zip file.

    If the first non-keyword arguement is not an :py:class:`zipfile:ZipFile` then
    initialises with a blank parent constructor and then loads data, otherwise,
    calls parent constructor.

    """

    priority = 32
    patterns = ["*.zip"]

    def __init__(self, *args, **kargs):
        """Constructor to catch initialising with an h5py.File or h5py.Group
        """
        if len(args) > 0:
            other = args[0]
            if isinstance(other, zf.ZipFile):
                if len(args) == 2 and isinstance(args[1], string_types):
                    kargs["file"] = args[1]
                elif "file" not in kargs:
                    kargs["file"] = other.namelist()[0]
                if kargs["file"] not in other.namelist():
                    raise StonerLoadError("File {} not found in zip file {}".format(name, other.filename))
                #Ok, by this point we have a zipfile which has a file in it. Construct ourselves and then load
                super(ZipFile, self).__init__(**kargs)
                self._extract(other, kargs["file"])
            elif isinstance(other, string_types):  # Passed a string - so try as a zipfile
                if zf.is_zipfile(other):
                    other = zf.ZipFile(other, "a")
                    args = (other, )
                elif test_is_zip(other):
                    args = test_is_zip(other)
                self.__init__(*args, **kargs)
            else:
                super(ZipFile, self).__init__(*args, **kargs)

    def _extract(self, archive, member):
        """Responsible for actually reading the zip file archive.

        Args:
            archive (zipfile.ZipFile): An open zip archive
            member (string): The name of one member of the zip file

        Return:
            A datafile like instance
        """
        tmp = DataFile()
        info = archive.getinfo(member)
        data = archive.read(info)
        self.__init__(tmp << data)
        self.filename = path.join(archive.filename, member)
        return self

    def _load(self, filename=None, *args, **kargs):
        """Load a file from the zip file, openining it as necessary"""
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename

        if isinstance(self.filename, zf.ZipFile):
            if not self.filename.fp:
                other = zf.ZipFile(self.filename.filename, "r")
                close_me = True
            else:
                other = self.filename
                close_me = False
            member = other.namelist()[0]
        elif isinstance(self.filename, string_types) and zf.is_zipfile(self.filename):
            other = zf.ZipFile(self.filename, "a")
            member = other.namelist()[0]
            close_me = True
        elif isinstance(self.filename, string_types) and test_is_zip(self.filename):
            other, member = test_is_zip(other)
            other = zf.ZipFile(other, "r")
            close_me = True
        else:
            raise StonerLoadError("{} does  not appear to be a real zip file".format(self.filename))

#Ok we can try reading now
        self._extract(other, member)
        if close_me:
            other.close()
        return self

    def save(self, filename=None):
        """Overrides the save method to allow ZipFile to be written out to disc (as a mininmalist output)

        Args:
            filename (string or zipfile.ZipFile instance): Filename to save as (using the same rules as for the load routines)

        Returns:
            A copy of itself."""
        if filename is None:
            filename = self.filename
        if filename is None or (isinstance(filename, bool) and not filename):  # now go and ask for one
            filename = self.__file_dialog('w')

        if isinstance(filename, string_types):  #We;ve got a string filename
            if test_is_zip(filename):  # We can find an existing zip file somewhere in the filename
                zipfile, member = test_is_zip(filename)
                zipfile = zf.ZipFile(zipfile, "a")
                close_me = True
            elif path.exists(filename):  # The fiule exists but isn't a zip file
                raise IOError("{} Should either be a zip file or a new zip file".format(filename))
            else:  # Path doesn't exist, use extension of file part to find where the zip file should be
                parts = path.split(filename)
                for i, part in enumerate(parts):
                    if path.splitext(part)[1].lower() == ".zip":
                        break
                else:
                    raise IOError("Can't figure out where the zip file is in {}".format(filename))
                zipfile = zf.ZipExtFile(path.join(*parts[:i + 1]), "w")
                close_me = True
                member = path.join(*parts[i + 1:])
        elif isinstance(filename, zf.ZipFile):  #Handle\ zipfile instance, opening if necessary
            if not filename.fp:
                filename = zf.ZipFile(filename.filename, 'a')
                close_me = True
            else:
                close_me = False
            zipfile = filename
            member = ""

        if member == "":  # Is our file object a bare zip file - if so create a default member name
            if len(zipfile.listname()) > 0:
                member = zipfile.listname()[1]
            else:
                member = "DataFile.txt"

        zipfile.writestr(member, str(self))
        if close_me:
            zipfile.close()
        return self


class ZipFolder(DataFolder):
    """A sub class of :py:class:`Stoner.Folders.DataFolder` that provides a
    method to load and save data from a single Zip file.

    See :py:class:`Stoner.Folders.DataFolder` for documentation on constructor.
    """

    def __init__(self, *args, **kargs):
        """Constructor for the HDF5Folder Class.
        """
        self.File = None
        self.type = ZipFile

        if len(args) > 1:
            if isinstance(args[0], string_types) and zf.is_zipfile(args[0]):
                this.File = zf.ZipFile(args[0], "a")
            elif isinstance(args[0], zf.ZipFile):
                if args[0].fp:
                    this.File = args[0]
                else:
                    this.File = zf.ZipFile(args[0].filename, "a")

        super(ZipFolder, self).__init__(*args, **kargs)

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
        file_wildcard = "zip file (*.zip)|All files|*"

        if mode == "r":
            mode2 = "open"
        elif mode == "w":
            mode2 = "save as"

        dlg = FileDialog(action=mode2, wildcard=file_wildcard)
        dlg.open()
        if dlg.return_code == OK:
            self.directory = dlg.path
            self.File = zf.ZipFile(self.directory, mode)
            self.File.close()
            return self.directory
        else:
            return None

    def getlist(self, recursive=None, directory=None, flatten=None):
        """Reads the Zip File to construct a list of ZipFile objects"""

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
        if isinstance(directory, string_types) and zf.is_zipfile(directory):
            self.directory = directory
            directory = zf.ZipFile(directory, 'a')
            self.File = directory
            close_me = True
        elif isinstance(directory, zf.ZipFile):
            if directory.fp:
                self.File = directory
                close_me = False
            else:
                self.File = zf.ZipFile(directory, "a")
                close_me = True
            self.directory = self.File.filename
        elif isinstance(directory, string_types) and path.isdir(directory):  #Fall back to DataFolder
            return super(ZipFolder, self).getlist(recursive, directory, flatten)
        else:
            raise IOError("{} does not appear to be zip file!".format(directory))
        #At this point directory contains an open h5py.File object, or possibly a group
        self.files = directory.namelist()
        if flaten is None or not flatten:
            self.unflatten()
        if close_me:
            directory.close()
        return self

    def __read__(self, f):
        """Override the _-read method to handle pulling files from the zip file"""
        if isinstance(f, DataFile):  # This is an already loaded DataFile
            tmp = f
            f = tmp.filename
        elif isinstance(f, string_types):  #This sis a string, so see if it maps to a path in the current File
            if isinstance(self.File, zf.ZipFile) and f in self.File.namelist():
                if not self.File.fp:
                    with zf.ZipFile(self.File.filename, "r") as z:
                        tmp = ZipFile(z, f)
                else:
                    tmp = ZipFile(self.File, f)
            else:  # Otherwise fallback and try to laod from disc
                tmp = super(ZipFolder, self).__read__(f)
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
                    tmp[h] = numpy.mean(tmp.column(h))

        return tmp

    def save(self, root=None):
        """Saves a load of files to a single Zip file, creating members as it goes.

        Keyword Arguments:
            root (string): The name of the Zip file to save to if set to None, will prompt for a filename.

        Return:
            A list of group paths in the Zip file
        """

        if root is None:
            root = self._dialog(mode='w')
        elif isinstance(root, bool) and not root and isinstance(self.File, zf.ZipFile):
            root = self.File.filename
            self.File.close()
        self.File = zf.ZipFile(root, 'a')
        tmp = self.walk_groups(self._save)
        self.File.close()
        return tmp

    def _save(self, f, trail):
        """Create a virtual path of groups in the Zip file and save data.

        Args:
            f(DataFile):  A DataFile instance to save
            trail (list): The trail of groups

        Returns:
            The new filename of the saved DataFile.

        ZipFiles are really a flat heirarchy, so concatentate the trail and save the data using
        :py:meth:`Stoner.Zip.ZipFile.save`

        This routine is used by a walk_groups call - hence the prototype matches that required for
        :py:meth:`Stoner.Folders.DataFolder.walk_groups`.

        """
        if not isinstance(f, DataFile):
            f = DataFile(f)
        member = path.join(self.File.filename, *trail)
        member = path.abspath(path.join(member, f.filename))
        f = ZipFile(f)
        f.save(member)
        return f.filename
