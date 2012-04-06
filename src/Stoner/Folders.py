#############################################
#
# Classes for working directories of datafiles
#
# $Id: Folders.py,v 1.5 2012/04/06 19:36:08 cvs Exp $
#
# $Log: Folders.py,v $
# Revision 1.5  2012/04/06 19:36:08  cvs
# Update DataFolder to support regexps in pattern and filter. When used as a pattern named capturing groups can be used to feed metadata. Minor improvements in Core and fix to RasorFile
#
# Revision 1.4  2012/04/04 23:04:11  cvs
# Improvements to AnalyseFile and DataFolder
#
# Revision 1.3  2012/03/25 21:18:10  cvs
# Documentation updates and minor fixes
#
# Revision 1.2  2012/03/24 22:39:40  cvs
# Update focumentation for new DataFolder class. Add more functionality to DataFolder including indexing, representation and len support.
#
# Revision 1.1  2012/03/24 00:36:04  cvs
# Add a new DataFolder class with methods for sorting and grouping data files
#
#

import os
import re
import os.path as path
import fnmatch

from .Core import DataFile

class DataFolder(object):
    """Implements a class that manages lists of data files (e.g. the contents of a directory) and can sort and group them in arbitary ways

    This class is intended to help process large groups of datasets in a natural and convenient way."""

    type=DataFile
    pattern="*.*"
    directory=False
    files=[]
    groups={}

    def __init__(self, *args, **kargs):
        """Constructor of DataFolder.
        @param args Non keyword arguments
        @param kargs keyword arguments
        @return The newly constructed instance

        Handles constructors like:

        DataFolder('directory',type=DataFile,pattern='*.*', nolist=False)"""
        self.files=[]
        self.groups={}
        if not "type" in kargs:
            self.type=DataFile
        elif issubclass(kargs["type"], DataFile):
            self.type=kargs["type"]
        else:
            raise ValueError("type keyword arguemtn must be an instance of Stoner.Core.DataFile or a subclass instance")
        if not "pattern" in kargs:
            self.pattern="*.*"
        else:
            self.pattern=kargs["pattern"]
        if not "nolist" in kargs:
            nolist=False
        else:
            nolist=kargs["nolist"]
        if len(args)>0:
            if isinstance(args[0], str):
                self.directory=args[0]
        else:
            self.directory=os.getcwd()
        if "recursive" in kargs:
            recursive=kargs["recursive"]
        else:
            recursive=True
        if not nolist:
            self.getlist(recursive=recursive)

    def _dialog(self, message="Select Folder",  new_directory=True):
        """Creates a directory dialog box for working with

        @param message Message to display in dialog
        @param new_directory True if allowed to create new directory
        @return A directory to be used for the file operation."""
        from enthought.pyface.api import DirectoryDialog, OK
        # Wildcard pattern to be used in file dialogs.

        if isinstance(self.directory, str):
            dirname = self.directory
        else:
            dirname = os.getcwd()
        dlg = DirectoryDialog(action="open",  default_path=dirname,  message=message,  new_directory=new_directory)
        dlg.open()
        if dlg.return_code == OK:
            self.directory = dlg.path
            return self.directory
        else:
            return None

    def __iter__(self):
        """Returns the files iterator object
        @return self.files.__iter__"""
        for f in self.files:
            tmp= self.type(f)
            if isinstance(self.pattern,re._pattern_type ):
                m=self.pattern.search(f)
                for k in m.groupdict():
                    tmp.metadata[k]=tmp.metadata.string_to_type(m.groups(k))
            yield tmp

    def __len__(self):
        """Pass through to return the length of the files array
        @return len(self.files)"""
        return len(self.files)

    def __getitem__(self, i):
        """Load and returen DataFile type objects based on the filenames in self.files
        @param i The index(eces) of the files to return Can also be a string in which case it is interpreted as one of self.files
        @return One or more instances of DataFile objects

        This is the canonical method for producing a DataFile from a DataFolder. Various substitutions are done as the file is created:

        Firstly, the filename is inserted into the metadata key "Loaded From"

        Secondly, if the pattern was given as a regular exression then any named matching groups are interpreted as metadata key-value pairs (where the key
        will be the name of the capturing group and the value will be the match to the group. This allows metadata to be imported from the filename as it is loaded."""""
        if isinstance(i, str): # Ok we've done a DataFolder['filename']
            try:
                i=self.files.index(i)
            except ValueError:
                i=self.basenames.index(i)
        files=self.files[i]
        tmp= self.type(files)
        if isinstance(self.pattern, re._pattern_type):
            m=self.pattern.search(files)
            for k in m.groupdict():
                tmp.metadata[k]=tmp.metadata.string_to_type(m.group(k))
            tmp['Loaded from']=files
        return tmp

    def __getattr__(self, item):
        """Handles some special case attributes that provide alternative views of the DataFolder
        @param  item The attribute name being requested
        @return Depends on the attribute

        Currently supported attributes:
        \b basenames Returns the list of files after passing through os.path.basename()"""
        if item=="basenames":
            return [path.basename(x) for x in self.files]


    def __repr__(self):
        """Prints a summary of the DataFolder structure
        @return A string representation of the current DataFolder object"""
        return "DataFolder("+self.directory+") with pattern "+str(self.pattern)+" has "+str(len(self.files))+" files in "+str(len(self.groups))+" groups\n"+str(self.groups)

    def getlist(self, recursive=True, directory=None):
        """Scans the current directory, optionally recursively to build a list of filenames
        @param recursive Do a walk through all the directories for files
        @param directory Either a string path ot a new directory or False to open a dialog box or not set in which case existing director is used.
        @return A copy of the current DataFoder directory with the files stored in the files attribute"""
        self.files=[]
        if isinstance(directory,  bool) and not directory:
            self._dialog()
        elif isinstance(directory, str):
            self.directory=directory
        if isinstance(self.directory, bool) and not self.directory:
            self._dialog()
        if not recursive:
            root=self.directory
            dirs=[]
            files=[]
            for f in os.listdir(root):
                if path.isdir(path.join(root, f)):
                    dirs.append(f)
                elif path.isfile(path.join(root, f)):
                    files.append(f)
            target=[(root, dirs, files)]
        else:
            target=[x for x in os.walk(self.directory)]
        for root, dirs, files in target:
            if isinstance(self.pattern, str):
                for f in fnmatch.filter(files, self.pattern):
                    self.files.append(path.join(root, f))
            elif isinstance(self.pattern, re._pattern_type):
                newfiles=[]
                for f in files:
                    if self.pattern.search(f) is not None:
                        newfiles.append(f)
                self.files=newfiles
        return self

    def filterout(self, filter):
        """Synonym for self.filter(filter,invert=True)

        @param filter Either a string flename pattern or a callable function which takes a single parameter x which is an instance of a DataFile and evaluates True or False
        @return The current DataFolder object with the files in the file list filtered."""
        return self.filter(filter, invert=True)

    def filter(self, filter=None,  invert=False):
        """Filter the current set of files by some criterion
        @param filter Either a string flename pattern or a callable function which takes a single parameter x which is an instance of a DataFile and evaluates True or False
        @param invert Invert the sense of the filter (done by doing an XOR whith the filter condition
        @return The current DataFolder object"""

        files=[]
        if isinstance(filter, str):
            for f in self.files:
                if fnmatch.fnmatch(f, filter)  ^ invert:
                    files.append(f)
        if isinstance(filter, re._pattern_type):
            for f in self.files:
                if filter.search(f) is not None:
                    files.append(f)
        elif filter is None:
            raise ValueError("A filter must be defined !")
        else:
            for f in self.files:
                x=self.type(f)
                if filter(x)  ^ invert:
                    files.append(f)
        self.files=files
        return self

    def sort(self, key=None, reverse=False):
        """Sort the files by some key
        @param key Either a string or a callable function. If a string then this is interpreted as a
        metadata key, if callable then it is assumed that this is a a function of one paramater x
        that is a DataFile object and that returns a key value. If key is not specified, then a sort is performed on the filename
        @param reverse Optionally sort in reverse order
        @return A copy of the current DataFolder object"""
        if isinstance(key, str):
            self.files=sorted(self.files, cmp=lambda x, y:cmp(self[x].get(key), self[y].get(key)), reverse=reverse)
        elif key is None:
            self.files.sort(reverse=reverse)
        else:
            self.files=sorted(self.files,cmp=lambda x, y:cmp(key(self[x]), key(self[y])), reverse=reverse)
        return self

    def group(self, key):
        """Take the files and sort them into a series of separate DataFolder objects according to the value of the key
        @param key Either a simple string or callable function or a list. If a string then it is interpreted as an item of metadata in each file. If a callable function then
        takes a single argument x which should be an instance of a DataFile and returns some vale. If key is a list then the grouping is done recursively for each element
        in key.
        @return A copy of the current DataFolder object in which the groups attribute is a dictionary of DataFolder objects with sub lists of files

        If ne of the grouping metadata keys does not exist in one file then no exception is raised - rather the fiiles will be returned into the group with key None. Metadata keys that
        are generated from the filename are supported."""
        self.groups={}
        if isinstance(key, list):
            next_keys=key[1:]
            key=key[0]
        else:
            next_keys=[]
        if isinstance(key, str):
            k=key
            key=lambda x:x.get(k)
        for f in self.files:
            x=self[f]
            v=key(x)
            if not v in self.groups:
                self.groups[v]=DataFolder(self.directory, type=self.type, pattern=self.pattern, nolist=True)
            self.groups[v].files.append(f)
        if len(next_keys)>0:
            for g in self.groups:
                self.groups[g].group(next_keys)
        return self

    def zipp_groups(self, groups):
        """Return a list of tuples of DataFiles drawn from the specified groups
        @param groups A list of keys of groups in the DataFolder
        @return A list of tuples of groups of files: [(grp_1_file_1,grp_2_file_1....grp_n_files_1),(grp_1_file_2,grp_2_file_2....grp_n_file_2)....(grp_1_file_m,grp_2_file_m...grp_n_file_m)]
        """
        if not isinstance(groups, list):
            raise SyntaxError("groups must be a list of groups")
        grps=[[y for y in self.groups[x]] for x in groups]
        return zip(*grps)




