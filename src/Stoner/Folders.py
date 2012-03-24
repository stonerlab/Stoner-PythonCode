#############################################
#
# Classes for working directories of datafiles
#
# $Id: Folders.py,v 1.1 2012/03/24 00:36:04 cvs Exp $
#
# $Log: Folders.py,v $
# Revision 1.1  2012/03/24 00:36:04  cvs
# Add a new DataFolder class with methods for sorting and grouping data files
#
#

import os
import os.path as path
import fnmatch

from .Core import DataFile

class DataFolder:
    """Implements a class that manages lists of data files (e.g. the contents of a directory) and can sort and group them in arbitary ways

    This class is intended to help process large groups of datasets in a natural and convenient way."""

    type=DataFile
    pattern="*.*"
    directory="."
    files=[]
    groups={}

    def __init__(self, *args, **kargs):
        """Constructor of DataFolder.
        @param args Non keyword arguments
        @param kargs keyword arguments
        @return The newly constructed instance

        Handles constructors like:

        DataFolder('directory',type=DataFile,pattern='*.*')"""
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
        if len(args)>0:
            if isinstance(args[0], str):
                self.directory=args[0]
        else:
            self.directory=os.getcwd()

    def __iter__(self):
        """Returns the files iterator object
        @return self.files.__iter__"""
        for f in self.files:
            yield self.type(f)


    def getlist(self, recursive=True):
        """Scans the current directory, optionally recursively to build a list of filenames
        @param recursive Do a walk through all the directories for files
        @return A copy of the current DataFoder directory with the files stored in the files attribute"""
        self.files=[]
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
            for f in fnmatch.filter(files, self.pattern):
                self.files.append(path.join(root, f))
        return self

    def filter(self, filter=None,  invert=False):
        """Filter the current set of files by some criterion
        @param filter Either a string flename pattern or a callable function which takes a single parameter x which is an instance of a DataFile and evaluates True or False
        @param invert Invert the sense of the filter (done by doing an XOR whith the filter condition
        @return The current DataFolder object"""

        files=[]
        if isinstance(filter, str):
            for f in self.files:
                x=self.type(f)
                if ffnmatch.fnmatch(f, filter)  ^ invert:
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
            self.files.sort(cmp=lambda x, y:cmp(self.type(x)[key], self.type(y)[key]), reverse=True)
        elif key is None:
            self.files.sort(reverse=reverse)
        else:
            self.files.sort(cmp=lambda x, y:cmp(key(self.type(x)), key(self.type(y))), reverse=reverse)
        return self

    def group(self, key):
        """Take the files and sort them into a series of separate DataFolder objects according to the value of the key
        @param key Either a simple string or callable function or a list. If a string then it is interpreted as an item of metadata in each file. If a callable function then
        takes a single argument x which should be an instance of a DataFile and returns some vale. If key is a list then the grouping is done recursively for each element
        in key.
        @return A copy of the current DataFolder object in which the groups attribute is a dictionary of DataFolder objects with sub lists of files"""
        self.groups={}
        if isinstance(key, list):
            next_keys=key[1:]
            key=key[0]
        else:
            next_keys=[]
        if isinstance(key, str):
            k=key
            key=lambda x:x[k]
        for f in self.files:
            x=self.type(f)
            v=key(x)
            if not v in self.groups:
                self.groups[v]=DataFolder(self.directory, type=self.type, pattern=self.pattern)
            self.groups[v].files.append(f)
        if len(next_keys)>0:
            for g in self.groups:
                self.groups[g].group(next_keys)
        return self




