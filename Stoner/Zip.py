# -*- coding: utf-8 -*-
"""Support reading DataFile like objects into and outof standard zip files.

Classes Include

* ZippedFile - A :py:class:`Stoner.Code.DataFile` subclass that can save and load data from a zip files
* ZipFolder - A :py:class:`Stoner.Folders.DataFolder` subclass that can save and load data from a single zip file
"""
__all__ = ["test_is_zip", "ZippedFile", "ZipFolderMixin", "ZipFolder"]
import zipfile as zf
import os.path as path
from traceback import format_exc
import fnmatch

from .compat import string_types, str2bytes, get_filedialog, _pattern_type, path_types
from .Core import DataFile, StonerLoadError
from .Folders import DiskBasedFolderMixin
from .folders.core import baseFolder
from .folders.utils import pathjoin
from .tools import copy_into, make_Data
from .tools.file import get_filename


def test_is_zip(filename, member=""):
    """Recursively searches for a zipfile in the tree.

    Args:
        filename (str):
            Path to test whether it is a zip file or not.

    Keyword Arguments:
        member (str):
            Used in recursive calls to identify the path within the zip file

    Returns:
        False or (filename,member):
            Returns False if not a zip file, otherwise the actual filename of the zip file and the nanme of the
            member within that
        zipfile.
    """
    if not filename or str(filename) == "":
        return False
    if zf.is_zipfile(filename):
        return filename, member
    part = path.basename(filename)
    newfile = path.dirname(filename)
    if newfile == filename:  # reached the end of the line
        part = filename
        newfile = ""
    if member != "":
        newmember = path.join(part, member)
    else:
        newmember = part
    return test_is_zip(newfile, newmember)


class ZippedFile(DataFile):
    """A sub class of DataFile that sores itself in a zip file.

    If the first non-keyword argument is not an :py:class:`zipfile:ZipFile` then
    initialises with a blank parent constructor and then loads data, otherwise,
    calls parent constructor.

    """

    priority = 32
    patterns = ["*.zip"]

    mime_type = ["application/zip"]

    def __init__(self, *args, **kargs):
        """Catch initialising with an open zf.ZipFile."""
        if len(args) > 0:
            other = args[0]
            if isinstance(other, zf.ZipFile):
                otherdir = other.namelist()
                if len(args) == 2 and isinstance(args[1], string_types):  # ZippedFile(open_zip,"filename")
                    kargs["filename"] = args[1].replace("\\", "/")
                elif "filename" not in kargs:  # ZippedFile(open_zip) - assume we use tyhe first zipped file in there
                    kargs["filename"] = other.namelist()[0]
                # Attempt to normalise start of path
                if kargs["filename"].startswith("./") and not (
                    otherdir[0].startswith("/") or otherdir[0].startswith("./")
                ):
                    kargs["filename"] = kargs["filename"][2:]
                if kargs["filename"] not in other.namelist():  # New file not in the zip file yet
                    raise StonerLoadError(f"File {kargs['filename']} not found in zip file {other.filename}")
                # Ok, by this point we have a zipfile which has a file in it. Construct ourselves and then load
                super().__init__(**kargs)
                self._extract(other, kargs["filename"])
            elif isinstance(other, path_types):  # Passed a string - so try as a zipfile
                if zf.is_zipfile(other):
                    other = zf.ZipFile(other, "a")
                    args = args = list(args)
                    args[0] = other
                elif test_is_zip(other):
                    args = test_is_zip(other)
                self.__init__(*args, **kargs)
            else:
                super().__init__(*args, **kargs)
        else:
            super().__init__(*args, **kargs)

    def _extract(self, archive, member):
        """Responsible for actually reading the zip file archive.

        Args:
            archive (zipfile.ZipFile):
                An open zip archive
            member (string):
                The name of one member of the zip file

        Return:
            A datafile like instance
        """
        info = archive.getinfo(member)
        data = archive.read(info)  # In Python 3 this would be a bytes
        tmp = make_Data(data)
        copy_into(tmp, self)
        # self.__init__(tmp << data)
        self.filename = path.join(archive.filename, member)
        return self

    def _load(self, *args, **kargs):
        filename, args, kargs = get_filename(args, kargs)
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename
        try:
            if isinstance(self.filename, zf.ZipFile):  # Loading from an ZipFile
                if not self.filename.fp:  # Open zipfile if necessary
                    other = zf.ZipFile(self.filename.filename, "r")
                    close_me = True
                else:  # Zip file is already open
                    other = self.filename
                    close_me = False
                member = kargs.get("member", other.namelist()[0])
                solo_file = len(other.namelist()) == 1
            elif isinstance(self.filename, path_types) and zf.is_zipfile(
                self.filename
            ):  # filename is a string that is a zip file
                other = zf.ZipFile(self.filename, "a")
                member = kargs.get("member", other.namelist()[0])
                close_me = True
                solo_file = len(other.namelist()) == 1
            else:
                raise StonerLoadError(f"{self.filename} does  not appear to be a real zip file")
        except StonerLoadError:
            raise
        except Exception as err:  # pylint: disable=W0703 # Catching everything else here
            try:
                exc = format_exc()
                other.close()
            except (AttributeError, NameError, ValueError, TypeError, zf.BadZipFile, zf.LargeZipFile):
                pass
            raise StonerLoadError(f"{self.filename} threw an error when opening\n{exc}") from err
        # Ok we can try reading now
        self._extract(other, member)
        if close_me:
            other.close()
        if solo_file:
            self.filename = str(filename)
        return self

    def save(self, filename=None, **kargs):
        """Override the save method to allow ZippedFile to be written out to disc (as a mininmalist output).

        Args:
            filename (string or zipfile.ZipFile instance):
                Filename to save as (using the same rules as for the load routines)

        Returns:
            A copy of itself.
        """
        if filename is None:
            filename = self.filename
        if filename is None or (isinstance(filename, bool) and not filename):  # now go and ask for one
            filename = self.__file_dialog("w")
        compression = kargs.pop("compression", zf.ZIP_DEFLATED)
        try:
            if isinstance(filename, path_types):  # We;ve got a string filename
                if test_is_zip(filename):  # We can find an existing zip file somewhere in the filename
                    zipfile, member = test_is_zip(filename)
                    zipfile = zf.ZipFile(zipfile, "a")
                    close_me = True
                elif path.exists(filename):  # The fiule exists but isn't a zip file
                    raise IOError(f"{filename} Should either be a zip file or a new zip file")
                else:  # Path doesn't exist, use extension of file part to find where the zip file should be
                    parts = path.split(filename)
                    for i, part in enumerate(parts):
                        if path.splitext(part)[1].lower() == ".zip":
                            break
                    else:
                        raise IOError(f"Can't figure out where the zip file is in {filename}")
                    zipfile = zf.ZipFile(path.join(*parts[: i + 1]), "w", compression, True)
                    close_me = True
                    member = path.join("/", *parts[i + 1 :])
            elif isinstance(filename, zf.ZipFile):  # Handle\ zipfile instance, opening if necessary
                if not filename.fp:
                    filename = zf.ZipFile(filename.filename, "a")
                    close_me = True
                else:
                    close_me = False
                zipfile = filename
                member = ""

            if (
                member == "" or member == "/"
            ):  # Is our file object a bare zip file - if so create a default member name
                if len(zipfile.namelist()) > 0:
                    member = zipfile.namelist()[-1]
                    self.filename = path.join(filename, member)
                else:
                    member = "DataFile.txt"
                    self.filename = filename

            zipfile.writestr(member, str2bytes(str(self)))
            if close_me:
                zipfile.close()
        except (zipfile.BadZipFile, IOError, TypeError, ValueError) as err:
            error = format_exc()
            try:
                zipfile.close()
            finally:
                raise IOError(f"Error saving zipfile\n{error}") from err
        return self


class ZipFolderMixin:
    """Provides methods to load and save data from a single Zip file.

    See :py:class:`Stoner.Folders.DataFolder` for documentation on constructor.

    Note:
        As this mixin class provides both read and write storage, it cannot be mixed in with another class that
        provides a __setter__ method without causing a problem.
    """

    _defaults = {
        "pattern": ["*.*"],
        "type": None,
        "exclude": ["*.tdms_index"],
        "read_means": False,
        "recursive": True,
        "flat": False,
        "readlist": True,
    }

    def __init__(self, *args, **kargs):
        """Initialise the file attribute."""
        self.File = None
        super().__init__(*args, **kargs)

    @property
    def full_key(self):
        """Generate a full canonical path through the zip archive to this file."""
        filename = path.splitdrive(self.File.filepath)[1]
        name = path.splitdrive(self.path)[1]
        return path.relpath(name, filename).replace(path.sep, "/")

    @property
    def key(self):
        """Return the immediate filename of the stored file."""
        return path.basename(self.directory)

    @key.setter
    def key(self, value):
        """Set the immediate filename that will be used when the file is saved."""
        self.path = pathjoin(self.directory, value)

    def _dialog(self, mode="r"):
        """Create a file dialog box for working with.

        Args:
            mode (string):
                Where the dialog is for opening or saving the file.

        Returns:
            A directory to be used for the file operation.
        """
        file_wildcard = [("zip file", "*.zip"), ("All files", "*.*")]

        if mode == "r":
            what = "file"
        else:
            what = "save"

        dlg = get_filedialog(what=what, filetypes=file_wildcard)
        if dlg is not None:
            self.directory = dlg
            self.File = zf.ZipFile(self.directory, mode)
            self.File.close()
            return self.directory
        return None

    def getlist(self, recursive=None, directory=None, flatten=None):
        """Read the Zip File to construct a list of ZipFile objects."""
        if recursive is None:
            recursive = self.recursive
        self.files = []
        self.groups = {}

        if flatten is None:
            flatten = self.flat

        if self.File is None and directory is None:
            self.File = zf.ZipFile(self._dialog(), "r")
            close_me = True
        elif isinstance(directory, zf.ZipFile):
            if directory.fp:
                self.File = directory
                close_me = False
            else:
                self.File = zf.ZipFile(directory, "r")
                close_me = True
        elif isinstance(directory, path_types) and path.isdir(directory):  # Fall back to DataFolder
            return super().getlist(recursive, directory, flatten)
        elif isinstance(directory, path_types) and zf.is_zipfile(directory):
            self.File = zf.ZipFile(directory, "r")
            close_me = True
        elif isinstance(self.File, zf.ZipFile):
            if self.File.fp:
                close_me = False
            else:
                self.File = zf.ZipFile(self.File.filename, "r")
                close_me = True
        else:
            raise IOError(f"{directory} does not appear to be zip file!")
        # At this point directory contains an open h5py.File object, or possibly a group
        self.path = self.File.filename
        files = [x.filename for x in self.File.filelist]
        for p in self.exclude:  # Remove excluded files
            if isinstance(p, string_types):
                for f in list(fnmatch.filter(files, p)):
                    del files[files.index(f)]
            if isinstance(p, _pattern_type):
                matched = []
                # For reg expts we iterate over all files, but we can't delete matched
                # files as we go as we're iterating over them - so we store the
                # indices and delete them later.
                for f in files:
                    if p.search(f):
                        matched.append(files.index(f))
                matched.sort(reverse=True)
                for i in matched:  # reverse sort the matching indices to safely delete
                    del files[i]

        for p in self.pattern:  # pattern is a list of strings and regeps
            if isinstance(p, string_types):
                for f in fnmatch.filter(files, p):
                    del files[files.index(f)]
                    f.replace(path.sep, "/")
                    self.append(f)
            elif isinstance(p, _pattern_type):
                matched = []
                # For reg expts we iterate over all files, but we can't delete matched
                # files as we go as we're iterating over them - so we store the
                # indices and delete them later.
                for ix, f in enumerate(files):
                    if p.search(f):
                        f.replace(path.sep, "/")
                        self.append(f)
                    else:
                        matched.append(ix)
                for i in reversed(matched):  # reverse sort the matching indices to safely delete
                    del files[i]

        self._zip_contents = files

        if flatten is None or not flatten:
            self.unflatten()
        if close_me:
            self.File.close()
        return self

    def __clone__(self, other=None, attrs_only=False):
        """Do whatever is necessary to copy attributes from self to other."""
        if other is None:
            if attrs_only:
                other = type(self)(readlist=False)
            else:
                other = type(self)()
        for arg in self._defaults:
            if hasattr(self, arg):
                setattr(other, arg, getattr(self, arg))
        return super().__clone__(other=other, attrs_only=attrs_only)

    def __getter__(self, name, instantiate=True):
        """Load the specified name from a compressed archive.

        Parameters:
            name (key type):
                The canonical mapping key to construct the path from.

        Keyword Arguments:
            instantiate (bool):
                IF True (default) then always return a :py:class:`Stoner.Core.Data` object. If False,
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

        if instantiate:
            try:
                return self.type(ZippedFile(path.join(self.File.filename, name)))
            except AttributeError:  # closed zip file?
                filename = test_is_zip(self.directory)[0]
                self.File = zf.ZipFile(filename, "a")
                tmp = self.type(ZippedFile(path.join(self.File.filename, name)))
                self.File.close()
                return tmp
        else:
            return name

    def __lookup__(self, name):
        """Look for a given name in the ZipFolder namelist.

        Parameters:
            name(str):
                Name of an object

        Returns:
            A canonical key name for that file

        Note:
            We try two things - first a direct lookup in the namelist if there is an exact match to the key and then
            we prepend the ZipFolder's path to try for a match with just the final part of the filename.
        """
        try:  # try to go back to the base to see if it's already loaded
            return self._storage_class.__lookup__(self, name)
        except (AttributeError, IndexError, KeyError):  # Ok, that failed, so let's
            pass

        try:
            if isinstance(name, string_types):
                name = name.replace(path.sep, "/")
                # First try the direct lookup - will work if we have a full name
                if name in self.File.namelist():
                    return name
                pth = path.normpath(path.join(self.full_key, name)).replace(path.sep, "/")
                if pth in self.File.namelist():
                    return pth
        except AttributeError:
            pass
        return super().__lookup__(name)

    def save(self, root=None):
        """Save a load of files to a single Zip file, creating members as it goes.

        Keyword Arguments:
            root (string):
                The name of the Zip file to save to if set to None, will prompt for a filename.

        Return:
            A list of group paths in the Zip file
        """
        if root is None:
            root = self._dialog(mode="w")
        elif isinstance(root, bool) and not root and isinstance(self.File, zf.ZipFile):
            root = self.File.filename
            self.File.close()
        mode = "a" if path.exists(root) else "w"
        self.File = zf.ZipFile(root, mode)
        self.File.close()  # Close the file having created it
        tmp = self.walk_groups(self._save)
        self.File.close()
        return tmp

    def _save(self, f, trail):
        """Create a virtual path of groups in the Zip file and save data.

        Args:
            f(DataFile):
                A DataFile instance to save
            trail (list):
                The trail of groups

        Returns:
            The new filename of the saved DataFile.

        ZipFiles are really a flat hierarchy, so concatenate the trail and save the data using
        :py:meth:`Stoner.Zip.ZipFile.save`

        This routine is used by a walk_groups call - hence the prototype matches that required for
        :py:meth:`Stoner.Folders.DataFolder.walk_groups`.

        """
        if not isinstance(f, DataFile):
            f = DataFile(f)
        filename = path.splitdrive(f.filename)[1]
        bits = [self.File.filename] + trail + [filename]
        pathsep = path.join("a", "b")[1]
        for ix, b in enumerate(bits):
            if ix == 0 or not b.startswith(pathsep):
                continue
            bits[ix] = b[1:]
        member = path.join(*bits)
        f = ZippedFile(f)
        f.save(member)
        return f.filename


class ZipFolder(ZipFolderMixin, DiskBasedFolderMixin, baseFolder):
    """A sub class of DataFile that sores itself in a zip file.

    If the first non-keyword argument is not an :py:class:`zipfile:ZipFile` then
    initialises with a blank parent constructor and then loads data, otherwise,
    calls parent constructor.

    """
