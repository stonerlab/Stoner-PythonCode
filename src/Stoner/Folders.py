"""
 FStoner.Folders : Classes for working collections of data files

 Classes:
     :py:class:`DataFolder` - manages a list of individual data files (e.g. from a directory tree)
"""


import os
import re
import os.path as path
import fnmatch
import numpy
from copy import copy
import unicodedata
import string

from .Core import DataFile

class DataFolder(object):
    """Implements a class that manages lists of data files (e.g. the contents of a directory) and can sort and group them in arbitary ways

    Attributes:
        directgory (string): Root directory of the files handled by the DataFolder
        files (list): List of filenames or loaded :py:class:`Stoner.DataFile` instances
        groups (dict of :py:class:`DataFolder`) Represent a heirarchy of Folders
        read_means (bool): If tru, create memtadata when reading each file that is the mean of each column
        pattern (string or re): Matches which files in the  directory tree are included. If pattern is a compiled
            reular expression with named groups then the named groups generare used to generate metadata in the :py:class:`Stoner.DataFile`
            object.
        basenames (list of string): Returns the list of files after passing through os.path.basename()
        ls (list of strings): Returns a list of filenames (either the matched filename patterns, or
            :py;attr:`Stoner.Core.DataFile.filename` if DataFolder.files contains DataFile objects
        lsgrp (list of string): Returns a list of the group keys (equivalent to DataFolder.groups.keys()

    Args:
        directory (string or :py:class:`DataFolder` instance): Where to get the data files from

    Keyword Arguments:
        type (class): An subclass of :py:class:`Stoner.Core.DataFolder` that will be used to construct the individual DataFile objects in the folder
        pattern (string or re): A filename pattern - either globbing string or regular expression
        nolist (bool): Delay doing a directory scan to get data files

    Returns:
        The newly constructed instance

    Note:
        All other keywords are set as attributes of the DataFolder.

    Todo:
        Handle :pyth:meth:`__init__(DataFolder)` with subclasses

    """

    def __init__(self, *args, **kargs):
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
            elif isinstance(args[0],DataFolder):
                other=args[0]
                for k in other.__dict__:
                    self.__dict__[k]=other.__dict__[k]
                return None
        else:
            self.directory=os.getcwd()
        recursive=True
        for v in kargs:
            self.__setattr__(v,kargs[v])
        if not nolist:
            self.getlist(recursive=recursive)

    def __read__(self,f):
        """Reads a single filename in and creates an instance of DataFile.

        Args:
            f(string or :py:class:`Stoner.Core.DataFile`): A filename or DataFile object

        Returns:
            A DataFile object

        Note:
             If self.pattern is a regular expression then use any named groups in it to create matadata from the
            filename. If self.read_means is true then create metadata from the mean of the data columns.
        """
        if isinstance(f,DataFile):
            return f
        tmp= self.type(f)
        if isinstance(self.pattern,re._pattern_type ):
            m=self.pattern.search(f)
            for k in m.groupdict():
                tmp.metadata[k]=tmp.metadata.string_to_type(m.group(k))
        if self.read_means:
            if len(tmp)==0:
                pass
            elif len(tmp)==1:
                for h in tmp.column_headers:
                    tmp[h]=tmp.column(h)[0]
            else:
                for h in tmp.column_headers:
                    tmp[h]=numpy.mean(tmp.column(h))
        tmp['Loaded from']=f
        return tmp


    def _dialog(self, message="Select Folder",  new_directory=True):
        """Creates a directory dialog box for working with

        Keyword Arguments:
            message (string): Message to display in dialog
            new_directory (bool): True if allowed to create new directory

        Returns:
            A directory to be used for the file operation."""
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

        Returns:
            self.files.__iter__"""
        for f in self.files:
            tmp=self.__read__(f)
            yield tmp

    def __len__(self):
        """Pass through to return the length of the files array

        Returns:
            len(self.files)"""
        return len(self.files)

    def __getitem__(self, i):
        """Load and returen DataFile type objects based on the filenames in self.files

        Args:
            i(int or slice): The index(eces) of the files to return Can also be a string in which case it is interpreted as one of self.files

        Returns:
            One or more instances of DataFile objects

        This is the canonical method for producing a DataFile from a DataFolder. Various substitutions are done as the file is created:
        1.  Firstly, the filename is inserted into the metadata key "Loaded From"
        2.  Secondly, if the pattern was given as a regular exression then any named matching groups are
            interpreted as metadata key-value pairs (where the key will be the name of the capturing
            group and the value will be the match to the group. This allows metadata to be imported
            from the filename as it is loaded."""""
        if isinstance(i, str): # Ok we've done a DataFolder['filename']
            try:
                i=self.ls.index(i)
            except ValueError:
                try:
                    i=self.basenames.index(i)
                except ValueError:
                    return self.groups[i]
        files=self.files[i]
        tmp=self.__read__(files)
        return tmp

    def __getattr__(self, item):
        """Handles some special case attributes that provide alternative views of the DataFolder

        Args:
            item (string): The attribute name being requested

        Returns:
            Depends on the attribute

        Attributes:
            basenames (list of string): Returns the list of files after passing through os.path.basename()
            ls (list of strings): Returns a list of filenames (either the matched filename patterns, or
                :py;attr:`Stoner.Core.DataFile.filename` if DataFolder.files contains DataFile objects
            lsgrp (list of string): Returns a list of the group keys (equivalent to DataFolder.groups.keys()

        """
        if item=="basenames":
            ret=[]
            for x in self.files:
                if isinstance(x,DataFile):
                    ret.append(path.basename(x.filename))
                elif isinstance(x,str):
                    ret.append(path.basename(x))
            return ret
        elif item=="lsgrp":
            return self.groups.keys()
        elif item=="ls":
            ret=[]
            for f in self.files:
                if isinstance(f,str):
                    ret.append(f)
                elif isinstance(f,DataFile):
                    ret.append(f.filename)
            return ret


    def __repr__(self):
        """Prints a summary of the DataFolder structure

        Returns:
            A string representation of the current DataFolder object"""
        return "DataFolder("+str(self.directory)+") with pattern "+str(self.pattern)+" has "+str(len(self.files))+" files in "+str(len(self.groups))+" groups\n"+str(self.groups)

    def __delitem__(self,item):
        """Deelte and item or a group from the DataFolder

        Args:
            item(string or int): the Item to be deleted.
                If item is an int, then assume that it is a file index
                otherwise it is assumed to be a group key
        """
        if isinstance(item, str) and item in self.groups:
            del self.groups[item]
        elif isinstance(item, int):
            del self.files[item]
        else:
            raise NotImplemented

    def getlist(self, recursive=True, directory=None):
        """Scans the current directory, optionally recursively to build a list of filenames

        Keyword Arguments:
            recursive (bool): Do a walk through all the directories for files
            directory (string or False): Either a string path ot a new directory or False to open a dialog box or not set in which case existing director is used.

        Returns:
            A copy of the current DataFoder directory with the files stored in the files attribute"""
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
                        newfiles.append(path.join(root, f))
                self.files=newfiles
        return self

    def filterout(self, filter):
        """Synonym for self.filter(filter,invert=True)

        Args:
        filter (string or callable): Either a string flename pattern or a callable function which takes a single parameter x which is an instance of a DataFile and evaluates True or False

        Returns:
            The current DataFolder object with the files in the file list filtered."""
        return self.filter(filter, invert=True)

    def filter(self, filter=None,  invert=False):
        """Filter the current set of files by some criterion

        Args:
            filter (string or callable): Either a string flename pattern or a callable function which takes a single parameter x which is an instance of a DataFile and evaluates True or False
            invert (bool): Invert the sense of the filter (done by doing an XOR whith the filter condition
        Returns:
            The current DataFolder object"""

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
            for i in range(len(self.files)):
                x=self[i]
                if filter(x)  ^ invert:
                    files.append(self.files[i])
        self.files=files
        return self

    def sort(self, key=None, reverse=False):
        """Sort the files by some key

        Keyword Arguments:
            key (string, callable or None): Either a string or a callable function. If a string then this is interpreted as a
                metadata key, if callable then it is assumed that this is a a function of one paramater x
                that is a :py:class:`Stoner.Core.DataFile` object and that returns a key value.
                If key is not specified (default), then a sort is performed on the filename
        reverse (bool): Optionally sort in reverse order

        Returns:
            A copy of the current DataFolder object"""
        if isinstance(key, str):
            self.files=sorted(self.files, cmp=lambda x, y:cmp(self[x].get(key), self[y].get(key)), reverse=reverse)
        elif key is None:
            fnames=self.ls
            fnames.sort(reverse=reverse)
            self.files=[self[f] for f in fnames]
        elif isinstance(key,re._pattern_type):
            self.files=sorted(self.files,cmp=lambda x, y:cmp(key.match(x).groups(),key.match(y).groups()), reverse=reverse)
        else:
            self.files=sorted(self.files,cmp=lambda x, y:cmp(key(self[x]), key(self[y])), reverse=reverse)
        return self

    def add_group(self,key):
        """Add a new group to the current Folder with the given key.

        Args:
            key(string): A hashable value to be used as the dictionary key in the groups dictionary
        Returns:
            A copy of the DataFolder

        Note:
            If key already exists in the groups dictionary then no action is taken.

        Todo:
            Propagate any extra attributes into the groups.
        """
        if key in self.groups: # do nothing here
            pass
        else:
            self.groups[key]=self.__class__(self.directory, type=self.type, pattern=self.pattern, read_means=self.read_means, nolist=True)
            self.groups[key].key=key
        return self

    def group(self, key):
        """Take the files and sort them into a series of separate DataFolder objects according to the value of the key

        Args:
            key (string or callable or list): Either a simple string or callable function or a list. If a string then it is interpreted as an item of metadata in each file. If a callable function then
                takes a single argument x which should be an instance of a DataFile and returns some vale. If key is a list then the grouping is done recursively for each element
                in key.
        Returns:
            A copy of the current DataFolder object in which the groups attribute is a dictionary of DataFolder objects with sub lists of files

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
            self.add_group(v)
            self.groups[v].files.append(f)
        if len(next_keys)>0:
            for g in self.groups:
                self.groups[g].group(next_keys)
        return self

    def zip_groups(self, groups):
        """Return a list of tuples of DataFiles drawn from the specified groups

        Args:
            groups(list of strings): A list of keys of groups in the Lpy:class:`DataFolder`

        ReturnsL
            A list of tuples of groups of files: [(grp_1_file_1,grp_2_file_1....grp_n_files_1),(grp_1_file_2,grp_2_file_2....grp_n_file_2)....(grp_1_file_m,grp_2_file_m...grp_n_file_m)]
        """
        if not isinstance(groups, list):
            raise SyntaxError("groups must be a list of groups")
        grps=[[y for y in self.groups[x]] for x in groups]
        return zip(*grps)

    def walk_groups(self, walker, group=False, replace_terminal=False,walker_args={}):
        """Walks through a heirarchy of groups and calls walker for each file.

        Args:
            walker (callable): a callable object that takes either a DataFile instance or a DataFolder instance.

        Keyword Arguments:
            group (bool): (default False) determines whether the wealker function will expect to be given the DataFolder
                representing the lowest level group or individual DataFile objects from the lowest level group
            replace_terminal (bool): if group is True and the walker function returns an instance of DataFile then the return value is appended
                to the files and the group is removed from the current DataFolder. This will unwind the group heirarchy by one level.
            walker_args (dict): a dictionary of static arguments for the walker function.

        Notes:
            The walker function should have a prototype of the form:
                walker(f,list_of_group_names,**walker_args)
                where f is either a DataFolder or DataFile."""
	return self.__walk_groups(walker,group=group,replace_terminal=replace_terminal,walker_args=walker_args,breadcrumb=[])

    def _removeDisallowedFilenameChars(filename):
        """Utility method to clean characters in filenames
        @param filename String filename
        """
        validFilenameChars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        cleanedFilename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore')
        return ''.join(c for c in cleanedFilename if c in validFilenameChars)


    def _save(self,grp,trail,root=None):
        """Save a group of files to disc by calling the save() method on each file. This internal method is called by walk_groups in turn
        called from the public save() method. The trail of group keys is used to create a directory tree.

        Args:
            grp (:py:class:`DataFolder` or :py:calss:`Stoner.DataFile`): A group or file to save
            trail (list of strings): the trail of paths used to get here
            root (string or None): a replacement root directory

        Returns:
            Saved Path
        """

        trail=[self._removeDisallowedFilenameChars(t) for t in trail]
        grp.filename=self._removeDisallowedFilenameChars(grp.filename)
        if root is None:
            root=self.directory

        pth=path.join(root,*trail)
        os.makesdirs(pth)
        grp.save(path.join(pth,grp.filename))
        return grp.filename

    def save(self,root=None):
        """Save the entire data folder out to disc using the groups as a directory tree,
        calling the save method for each file in turn.

        Args:
            root (string): The root directory to start creating files and subdirectories under. If set to None or not specified, the current folder's
                diretory attribute will be used.
        Returns:
            A list of the saved files
        """
        return self.walk_groups(self._save,walker_args={"root",root})


    def __walk_groups(self,walker,group=False,replace_terminal=False,walker_args={},breadcrumb=[]):
  	""""Actually implements the walk_groups method,m but adds the breadcrumb list of groups that we've already visited.

        Args:
            walker (callable): a callable object that takes either a DataFile instance or a DataFolder instance.

        Keyword Arguments:
            group (bool): (default False) determines whether the wealker function will expect to be given the DataFolder
                representing the lowest level group or individual DataFile objects from the lowest level group
            replace_terminal (bool): if group is True and the walker function returns an instance of DataFile then the return value is appended
                to the files and the group is removed from the current DataFolder. This will unwind the group heirarchy by one level.
            walker_args (dict): a dictionary of static arguments for the walker function.
            bbreadcrumb (list of strings): a list of the group names or key values that we've walked through

        Notes:
            The walker function should have a prototype of the form:
                walker(f,list_of_group_names,**walker_args)
                where f is either a DataFolder or DataFile."""
	if (len(self.groups)>0):
            ret=[]
            removeGroups=[]
            if replace_terminal:
                self.files=[]
            for g in self.groups:
                bcumb=copy(breadcrumb)
                bcumb.append(g)
                tmp=self.groups[g].__walk_groups(walker,group=group,replace_terminal=replace_terminal,walker_args=walker_args,breadcrumb=bcumb)
                if group and  replace_terminal and isinstance (tmp, DataFile):
                    removeGroups.append(g)
                    tmp.filename=g
                    self.files.append(tmp)
                    ret.append(tmp)
            for g in removeGroups:
                del(self.groups[g])
            return ret
	else:
	   if group:
	       return walker(self,breadcrumb,**walker_args)
	   else:
	       ret=[]
	       for f in self:
	           ret.append(walker(f,breadcrumb,**walker_args))
	       return ret
