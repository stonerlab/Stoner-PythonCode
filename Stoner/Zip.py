# -*- coding: utf-8 -*-
"""
Stoner.Zip module - sipport reading DataFile like objects into and outof standard zip files.

Classes Include

* ZippedFile - A :py:class:`Stoner.Code.DataFile` subclass that can save and load data from a zip files
* ZipFolder - A :py:class:`Stoner.Folders.DataFolder` subclass that can save and load data from a single zip file
"""
__all__ = ['test_is_zip','ZippedFile','ZipFolderMixin','ZipFolder']
from Stoner.compat import string_types,bytes2str,str2bytes
import zipfile as zf
from .Core import DataFile,StonerLoadError,metadataObject
from .Folders import baseFolder,pathjoin
import os.path as path
from traceback import format_exc
import numpy as _np_
from copy import copy,deepcopy
import fnmatch
import re


def test_is_zip(filename, member=""):
    """Recursively searches for a zipfile in the tree.

    Args:
        filename (str): Path to test whether it is a zip file or not.

    Keyword Arguments:
        member (str): Used in recursive calls to identify the path within the zip file

    Returns:
        False or (filename,member): Returns False if not a zip file, otherwise the actual filename of the zip file and the nanme of the member within that
        zipfile.
    """
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


class ZippedFile(DataFile):

    """A sub class of DataFile that sores itself in a zip file.

    If the first non-keyword arguement is not an :py:class:`zipfile:ZipFile` then
    initialises with a blank parent constructor and then loads data, otherwise,
    calls parent constructor.

    """

    priority = 32
    patterns = ["*.zip"]

    mime_type=["application/zip"]


    def __init__(self, *args, **kargs):
        "Constructor to catch initialising with an open zf.ZipFile"
        if len(args) > 0:
            other = args[0]
            if isinstance(other, zf.ZipFile):
                if len(args) == 2 and isinstance(args[1], string_types): #ZippedFile(open_zip,"filename")
                    kargs["filename"] = args[1]
                elif "filename" not in kargs: # ZippedFile(open_zip) - assume we use tyhe first zipped file in there
                    kargs["filename"] = other.namelist()[0]
                if kargs["filename"] not in other.namelist(): # New file not in the zip file yet
                    raise StonerLoadError("File {} not found in zip file {}".format(kargs["name"], other.filename))
                #Ok, by this point we have a zipfile which has a file in it. Construct ourselves and then load
                super(ZippedFile, self).__init__(**kargs)
                self._extract(other, kargs["filename"])
            elif isinstance(other, string_types):  # Passed a string - so try as a zipfile
                if zf.is_zipfile(other):
                    other = zf.ZipFile(other, "a")
                    args = (other, )
                elif test_is_zip(other):
                    args = test_is_zip(other)
                self.__init__(*args, **kargs)
            else:
                super(ZippedFile, self).__init__(*args, **kargs)

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
        data = bytes2str(archive.read(info)) # In Python 3 this would be a bytes
        self.__init__(tmp << data)
        self.filename = path.join(archive.filename, member)
        return self

    def _load(self, filename=None, *args, **kargs):
        "Load a file from the zip file, openining it as necessary"
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        try:
            if isinstance(self.filename, zf.ZipFile): # Loading from an ZipFile
                if not self.filename.fp: # Open zipfile if necessarry
                    other = zf.ZipFile(self.filename.filename, "r")
                    close_me = True
                else: #Zip file is already open
                    other = self.filename
                    close_me = False
                member = kargs.get("member",other.namelist()[0])
                solo_file=len(other.namelist())==1
            elif isinstance(self.filename, string_types) and zf.is_zipfile(self.filename): #filename is a string that is a zip file
                other = zf.ZipFile(self.filename, "a")
                member = kargs.get("member",other.namelist()[0])
                close_me = True
                solo_file=len(other.namelist())==1
            elif isinstance(self.filename, string_types) and test_is_zip(self.filename): #Filename is something buried in a zipfile
                other, member = test_is_zip(other)
                other = zf.ZipFile(other, "r")
                close_me = True
                solo_file=len(other.namelist())==1
            else:
                raise StonerLoadError("{} does  not appear to be a real zip file".format(self.filename))
        except StonerLoadError as e:
            raise e
        except Exception as e:
            try:
                exc=format_exc()
                other.close()
            except Exception:
                pass
            raise StonerLoadError("{} threw an error when opening\n{}".format(self.filename,exc))
#Ok we can try reading now
        self._extract(other, member)
        if close_me:
            other.close()
        if solo_file:
            self.filename=str(filename)
        return self

    def save(self, filename=None,compression=zf.ZIP_DEFLATED):
        """Overrides the save method to allow ZippedFile to be written out to disc (as a mininmalist output)

        Args:
            filename (string or zipfile.ZipFile instance): Filename to save as (using the same rules as for the load routines)

        Returns:
            A copy of itself.
        """
        if filename is None:
            filename = self.filename
        if filename is None or (isinstance(filename, bool) and not filename):  # now go and ask for one
            filename = self.__file_dialog('w')

        try:
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
                    zipfile = zf.ZipFile(path.join(*parts[:i + 1]), "w",compression,True)
                    close_me = True
                    member = path.join("/",*parts[i + 1:])
            elif isinstance(filename, zf.ZipFile):  #Handle\ zipfile instance, opening if necessary
                if not filename.fp:
                    filename = zf.ZipFile(filename.filename, 'a')
                    close_me = True
                else:
                    close_me = False
                zipfile = filename
                member = ""

            if member == "" or member =="/":  # Is our file object a bare zip file - if so create a default member name
                if len(zipfile.namelist()) > 0:
                    member = zipfile.namelist()[-1]
                    self.filename=path.join(filename,member)
                else:
                    member = "DataFile.txt"
                    self.filename=filename

            zipfile.writestr(member, str2bytes(str(self)))
            if close_me:
                zipfile.close()
        except Exception:
            error=format_exc()
            try:
                zipfile.close()
            except Exception:
                pass
            raise IOError("Error saving zipfile\n{}".format(error))
        return self

class ZipFolderMixin(object):

    """A sub class of :py:class:`Stoner.Folders.DataFolder` that provides a method to load and save data from a single Zip file.

    See :py:class:`Stoner.Folders.DataFolder` for documentation on constructor.

    Note:
        As this mixin class provides both read and write storage, it cannot be mixed in with another class that
        provides a __setter__ method without causing a problem.
    """

    _defaults={"pattern":["*.*"],
              "type":None,
              "exclude":["*.tdms_index"],
              "read_means":False,
              "recursive":True,
              "flat":False,
              "readlist":True,
              }


    def __init__(self, *args, **kargs):
        "Constructor for the ZipFolderMixin Class."
        for cls in self.__class__.__mro__[1:]: #Trail back to find a parent that might actually be handling the storage
            if "__setter__" in cls.__dict__ and "__getter__" in cls.__dict__:
                self._storage_class=cls
                break
        else:
            self._storage_class=baseFolder # Fall back to the base class
        self.File = None
        self.path=""
        defaults=copy(self._defaults)
        defaults.update(kargs)
        if defaults["type"] is None:
            from Stoner import Data
            defaults["type"]=Data
        if len(args) > 0:
            if isinstance(args[0], string_types) and zf.is_zipfile(args[0]):
                self.File = zf.ZipFile(args[0], "a")
            elif isinstance(args[0], zf.ZipFile):
                if args[0].fp:
                    self.File = args[0]
                else:
                    self.File = zf.ZipFile(args[0].filename, "a")
        else:
            self.File=None

        for k in defaults:
            setattr(self,k,kargs.pop(k,defaults[k]))

        self._zip_contents=[]

        super(ZipFolderMixin, self).__init__(*args, **kargs)

        if self.readlist:
            self.getlist()

    @property
    def directory(self):
        return self.path

    @directory.setter
    def directory(self,value):
        self.path=value
        
    @property
    def full_key(self):
        return path.relpath(self.path,self.File.filename).replace(path.sep,"/")

    @property
    def key(self):
        return path.basename(self.path)

    @key.setter
    def key(self,value):
        self.path=pathjoin(self.path,value)



    def _dialog(self, message="Select Folder", new_directory=True, mode='r'):
        """Creates a file dialog box for working with

        Args:
            message (string): Message to display in dialog
            new_file (bool): True if allowed to create new directory

        Returns:
            A directory to be used for the file operation.
        """
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
        "Reads the Zip File to construct a list of ZipFile objects"
        if recursive is None:
            recursive = self.recursive
        self.files = []
        self.groups = {}

        if flatten is None:
            flatten=self.flat

        if self.File is None and directory is None:
            self.File=zf.ZipExtFile(self._dialog(),"a")
            close_me = True
        elif isinstance(directory, zf.ZipFile):
            if directory.fp:
                self.File = directory
                close_me = False
            else:
                self.File = zf.ZipFile(directory, "a")
                close_me = True
        elif isinstance(directory, string_types) and path.isdir(directory):  #Fall back to DataFolder
            return super(ZipFolderMixin, self).getlist(recursive, directory, flatten)
        elif isinstance(self.File,zf.ZipFile):
            close_me=False
        else:
            raise IOError("{} does not appear to be zip file!".format(directory))
        #At this point directory contains an open h5py.File object, or possibly a group
        self.path=self.File.filename
        files=[x.filename for x in self.File.filelist]
        for p in self.exclude: #Remove excluded files
            if isinstance(p,string_types):
                for f in list(fnmatch.filter(files,p)):
                    del files[files.index(f)]
            if isinstance(p,re._pattern_type):
                matched=[]
                # For reg expts we iterate over all files, but we can't delete matched
                # files as we go as we're iterating over them - so we store the
                # indices and delete them later.
                for f in files:
                    if p.search(f):
                        matched.append(files.index(f))
                matched.sort(reverse=True)
                for i in matched: # reverse sort the matching indices to safely delete
                    del(files[i])

        for p in self.pattern: # pattern is a list of strings and regeps
            if isinstance(p,string_types):
                for f in fnmatch.filter(files, p):
                    del(files[files.index(f)])
                    f.replace(path.sep,"/")
                    self.append(f)
            elif isinstance(p,re._pattern_type):
                matched=[]
                # For reg expts we iterate over all files, but we can't delete matched
                # files as we go as we're iterating over them - so we store the
                # indices and delete them later.
                for ix,f in enumerate(files):
                    if p.search(f):
                        f.replace(path.sep,"/")
                        self.append(f)
                    else:
                        matched.append(ix)
                for i in reversed(matched): # reverse sort the matching indices to safely delete
                    del(files[i])

        self._zip_contents=files

        if flatten is None or not flatten:
            self.unflatten()
        if close_me:
            directory.close()
        return self

    def __clone__(self,other=None,attrs_only=False):
        """Do whatever is necessary to copy attributes from self to other."""
        if other is None and attrs_only:
            other=self.__class__(readlist=False)
        for arg in self._defaults:
            if hasattr(self,arg):
                setattr(other,arg,getattr(self,arg))
        return super(ZipFolderMixin,self).__clone__(other=other,attrs_only=attrs_only)            
    
    def __getter__(self,name,instantiate=True):
        """Loads the specified name from a compressed archive.

        Parameters:
            name (key type): The canonical mapping key to construct the path from.

        Keyword Arguments:
            instatiate (bool): IF True (default) then always return a :py:class:`Stoner.Core.Data` object. If False,
                the __getter__ method may return a key that can be used by it later to actually get the
                :py:class:`Stoner.Core.Data` object.

        Returns:
            (metadataObject): The metadataObject
        """
        try: # try to go back to the base to see if it's already loaded
            return self._storage_class.__getter__(self,name=name,instantiate=instantiate)
        except (AttributeError,IndexError,KeyError): #Ok, that failed, so let's
            pass
        
        #name=self.__lookup__(name)
        if instantiate:
            try:
                return self.type(ZippedFile(path.join(self.File.filename,name)))
            except AttributeError: # closed zip file?
                filename=test_is_zip(self.path)[0]
                self.File = zf.ZipFile(filename, "a")
                tmp=self.type(ZippedFile(path.join(self.File.filename,name)))
                self.File.close()
                return tmp
        else:
            return name
       
    def __lookup__(self,name):
        """Look for a given name in the ZipFolder namelist.

        Parameters:
            name(str): Name of an object

        Returns:
            A canonical key name for that file

        Note:
            We try two things - first a direct lookup in the namelist if there is an exact match to the key and then
            we preprend the ZipFolder's path to try for a match with just the final part of the filename.
        """
        try: # try to go back to the base to see if it's already loaded
            return self._storage_class.__lookup__(self,name)
        except (AttributeError,IndexError,KeyError): #Ok, that failed, so let's
            pass

        try:
            if isinstance(name,string_types):
                name=name.replace(path.sep,"/")
                #First try tthe direct lookup - will work if we have a full name
                if name in self.File.namelist():
                    return name
                pth=path.normpath(path.join(self.full_key,name)).replace(path.sep,"/")
                if pth in self.File.namelist():
                    return pth
        except AttributeError:
            pass
        return super(ZipFolderMixin,self).__lookup__(name)


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
        f = ZippedFile(f)
        f.save(member)
        return f.filename

class ZipFolder(ZipFolderMixin,baseFolder):

    """A sub class of DataFile that sores itself in a zip file.

    If the first non-keyword arguement is not an :py:class:`zipfile:ZipFile` then
    initialises with a blank parent constructor and then loads data, otherwise,
    calls parent constructor.

    """


    pass

