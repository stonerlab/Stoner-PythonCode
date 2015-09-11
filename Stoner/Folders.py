"""
 FStoner.Folders : Classes for working collections of data files

 Classes:
     :py:class:`DataFolder` - manages a list of individual data files (e.g. from a directory tree)
"""

from Stoner.compat import *
import os
import re
import os.path as path
import fnmatch
import numpy as _np_
from copy import copy
import unicodedata
import string
from collections import Iterable
from inspect import ismethod
import matplotlib.pyplot as plt

from .Core import DataFile

class DataFolder(object):
    """Implements a class that manages lists of data files (e.g. the contents of a directory) and can sort and group them in arbitary ways

    Attributes:
        basenames (list of str): Returns the list of files after passing through os.path.basename()
        depth (int): Maximum number of levels of groups below this :py:class:`DataFolder`.
        directory (string): Root directory of the files handled by the DataFolder
        extra_args (dict): Extra Arguments to pass to the constructors of the :py:class:`Stoner.Core.DataFile`
           objects.
        files (list): List of filenames or loaded :py:class:`Stoner.DataFile` instances
        groups (dict of :py:class:`DataFolder`) Represent a heirarchy of Folders
        ls (list of str): Returns a list of filenames (either the matched filename patterns, or
            :py;attr:`Stoner.Core.DataFile.filename` if DataFolder.files contains DataFile objects
        loaded (list of bool): Inidicates which fiels are loaded in memory for the :py:class:`DataFolder`
        lsgrp (list of str): Returns a list of the group keys (equivalent to DataFolder.groups.keys()
        mindepth (int): Minimum number of levels of groups from this :py:class:`DataFolder` to a terminal group.
        multifile (bool): if True a multi-file dialog box is used to load several files from the same folder
        pattern (string or re or sequence of strings and re): Matches which files in the  directory tree are included.
            If pattern is a compiled reular expression with named groups then the named groups are used to
            generate metadata in the :py:class:`Stoner.DataFile` object. If pattern is a list, then the set of
            files included in the py:class:`DataFolder` is the union of files that match any single pattern.
        read_means (bool): If true, create metadata when reading each file that is the mean of each column
        setas (list or string): Sets the default value of the :py:attr:`Stoner.Core.DataFile.setas` attribute for each
            :py:class:`Stoner.Core.DataFile` in the folder.
        type (DataFile): The type of the members of the :py:class:`DataFolder`. Can be either a subclass of 
            :py:class:`Stoner.Core.DataFile` or an instance of one (in which ase the class of the instance is used).

    Args:
        directory (string or :py:class:`DataFolder` instance): Where to get the data files from. If False will bring up a dialog for selecting the directory

    Keyword Arguments:
        type (class): An subclass of :py:class:`Stoner.Core.DataFolder` that will be used to construct the individual DataFile objects in the folder
        pattern (string or re): A filename pattern - either globbing string or regular expression
        nolist (bool): Delay doing a directory scan to get data files
        multifile (bool): if True brings up a dialog for selecting files from a directory.
        read_means (bool): Calculate the average value of each column and add it as metadata entries to the file.

    Returns:
        The newly constructed instance

    Note:
        All other keywords are set as attributes of the DataFolder.

    Todo:
        Handle :py:meth:`__init__(DataFolder)` with subclasses

    """
    
    _type=DataFile # class attribute to keep things happy  
    _pattern=None
    
    def __init__(self, *args, **kargs):
        self.directory=None
        self.files=[]
        self.flat=False
        self.read_means=False
        self.recursive=True
        self.groups={}
        self._file_attrs=dict()
        if not "type" in kargs:
            self.type=DataFile
        else:
            self.type=kargs["type"]
            del kargs["type"]
        self.pattern=kargs.pop("pattern","*.*")
        if not "nolist" in kargs:
            self.nolist=(len(args)==0)
        else:
            self.nolist=kargs["nolist"]
            del kargs["nolist"]
        if not "multifile" in kargs:
            self.multifile = False
        elif isinstance(kargs["multifile"], bool):
            self.multifile=kargs["multifile"]
            del kargs["multifile"]
        else:
            raise ValueError("multifile argument must be boolean")
        if "extra_args" in kargs:
            self.extra_args=kargs["extra_args"]
            del kargs["extra_args"]
        else:
            self.extra_args=dict()
        for v in kargs:
            self.__setattr__(v,kargs[v])
        if self.directory is None:
            self.directory=os.getcwd()
        if len(args)>0:
            if isinstance(args[0], string_types):
                self.directory=args[0]
                if not self.nolist:
                    self.getlist()
            elif isinstance(args[0],bool) and not args[0]:
                self.directory=False
                if not self.nolist:
                    self.getlist()
            elif isinstance(args[0],DataFolder):
                other=args[0]
                for k in other.__dict__:
                    self.__dict__[k]=other.__dict__[k]
            else:
                if not self.nolist:
                    self.getlist()
        else:
            if not self.nolist:
                self.getlist()

    ################################################################################
    ####### Property Methods #######################################################
    ################################################################################

    @property
    def basenames(self):
        """Returns a list of just the filename parts of the DataFolder."""
        ret=[]
        for x in self.files:
            if isinstance(x,DataFile):
                ret.append(path.basename(x.filename))
            elif isinstance(x,string_types):
                ret.append(path.basename(x))
        return ret
        
    @property
    def depth(self):
        """Gives the maximum number of levels of group below the current DataFolder."""
        if len(self.groups)==0:
            r=0
        else:
            r=1
            for g in self.groups:
                r=max(r,self.groups[g].depth+1)
        return r

    @property
    def loaded(self):
        """An iterator that indicates wether the contents of the :py:class:`Stoner.Folders.DataFolder` has been
        loaded into memory."""
        for f in self.files:
            yield isinstance(f,DataFile)

    @property    
    def lsgrp(self):
        """Returns a list of the groups as a generator."""
        for k in self.groups.keys():
            yield k

    @property
    def ls(self):
        ret=[]
        for f in self.files:
            if isinstance(f,string_types):
                ret.append(f)
            elif isinstance(f,DataFile):
                ret.append(f.filename)
        return ret

    @property
    def mindepth(self):
        """Gives the minimum number of levels of group below the current DataFolder."""
        if len(self.groups)==0:
            r=0
        else:
            r=1E6
            for g in self.groups:
                r=min(r,self.groups[g].depth+1)
        return r
        
    @property
    def pattern(self):
        return self._pattern
        
    @pattern.setter
    def pattern(self,value):
        """Sets the filename searching pattern(s) for the :py:class:`Stoner.Core.DataFile`s."""
        if isinstance(value,string_types):
            self._pattern=(value,)
        elif isinstance(value,re._pattern_type):
            self._pattern=(value,)
        elif isinstance(value,Iterable):
            self._pattern=[x for x in value]
        else:
            raise ValueError("pattern should be a string, regular expression or iterable object not a {}".format(type(value)))

    
    @property
    def type(self):
        """Defines the (sub)class of the :py:class:`Stoner.Core.DataFile` instances."""
        return self._type
        
    @type.setter
    def type(self,value):
        """Ensures that type is a subclass of DataFile."""
        if issubclass(value,DataFile):
            self._type=value
        elif isinstance(value,DataFile):
            self._type=value.__class__
        else:
            raise TypeError("{} os neither a subclass nor instance of DataFile".format(type(value)))

    #########################################################
    ######## Special Methods ################################
    #########################################################

    def __add__(self,other):
        """Implement the addition operator for DataFolder and DataFiles."""
        result=copy(self)
        if isinstance(other,DataFolder):
            result.files.extend([self.type(f) for f in other.files])
            result.groups.update(other.groups)
        elif isinstance(other,DataFile):
            result.files.append(self.type(other))
        else:
            result=NotImplemented
        return result

    def __delitem__(self,item):
        """Deelte and item or a group from the DataFolder

        Args:
            item(string or int): the Item to be deleted.
                If item is an int, then assume that it is a file index
                otherwise it is assumed to be a group key
        """
        if isinstance(item, string_types) and item in self.groups:
            del self.groups[item]
        elif isinstance(item, int):
            del self.files[item]
        elif isinstance(item, slice):
            indices = item.indices(len(self))
            for i in reversed(range(*indices)):
                del self.files[i]
        else:
            return NotImplemented

    def __dir__(self):
        """Returns the attributes of the current object by augmenting the keys of self.__dict__ with the attributes that __getattr__ will handle.
        """
        attr=dir(type(self))
        attr.extend(list(self.__dict__.keys()))
        attr.extend(dir(self.type))
        attr=list(set(attr))
        return attr


    def __get_file_attr__(self,item):
        if item in self._file_attrs:
            return self._file_attrs[item]
        else:
            raise KeyError()


    def __getattr__(self, item):
        """Handles some special case attributes that provide alternative views of the DataFolder

        Args:
            item (string): The attribute name being requested

        Returns:
            Depends on the attribute

        """
        if not item.startswith("_"):
            if item in dir(self._type): #Something is in our DataFile type
                if ismethod(getattr(self._type,item)): # It's a method
                    ret=self.__getattr_proxy(item)
                else: # It's a static attribute
                    ret=self.__get_file_attr__(item)
            else: # Ok, pass back
                ret=super(DataFolder,self).__getattribute__(item)
        else: # We dpon't intercept private or special methods
            ret=super(DataFolder,self).__getattribute__(item)
        return ret

    def __getattr_proxy(self,item):
        """Make a prpoxy call to access a method of the DataFile like types.

        Args:
            item (string): Name of method of DataFile class to be called

        Returns:
            Either a modifed copy of this DataFolder or a list of return values
            from evaluating the method for each file in the Folder.
        """
        def _wrapper_(*args,**kargs):
            """Wraps a call to the DataFile type for magic method calling.
            Note:
                This relies on being defined inside the enclosure of the DataFolder method
                so we have access to self and item"""
            retvals=[]
            for ix,f in enumerate(self):
                meth=f.__getattribute__(item)
                ret=meth(*args,**kargs)
                if ret is not f: # method did not returned a modified version of the DataFile
                    retvals.append(ret)
            if len(retvals)==0: # If we haven't got anything to retun, return a copy of our DataFolder
                retvals=self
            return retvals
        #Ok that;s the wrapper function, now return  it for the user to mess around with.
        return _wrapper_

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
        if isinstance(i,int):
            files=self.files[i]
            tmp=self.__read__(files)
            self.files[i]=tmp
            return tmp
        elif isinstance(i, string_types): # Ok we've done a DataFolder['filename']
            try:
                i=self.ls.index(i)
                return self.__read__(self.files[i])
            except ValueError:
                try:
                    i=self.basenames.index(i)
                except ValueError:
                    return self.groups[i]
        elif isinstance(i, slice):
            indices = i.indices(len(self))
            return [self[i] for i in range(*indices)]
        elif isinstance(i,tuple):
            g=self
            for ix in i:
                g=g[ix]
            return g
        else:
            return self.groups[i]

    def __len__(self):
        """Pass through to return the length of the files array

        Returns:
            len(self.files)"""
        return len(self.files)


    def __next__(self):
        for i in range(len(self.files)):
            yield self[i]

    def next(elf):
        for i in range(len(self.files)):
            yield self[i]

    def __repr__(self):
        """Prints a summary of the DataFolder structure

        Returns:
            A string representation of the current DataFolder object"""
        s="DataFolder({}) with pattern {} has {} files and {} groups\n".format(self.directory,self.pattern,len(self.files),len(self.groups))
        for g in self.groups: # iterate over groups
            r=self.groups[g].__repr__()
            for l in r.split("\n"): # indent each line by one tab
                s+="\t"+l+"\n"
        return s.strip()

    def __setattr__(self,name,value):
        """Pass through to set the sample attributes."""
        if name.startswith("_"): # pass ddirectly through for private attributes
            super(DataFolder,self).__setattr__(name,value)            
        elif name in self.__dict__ and not callable(self.__getattr__(name)):
            super(DataFolder,self).__setattr__(name,value)
        elif name in dir(self._type()):
            self._file_attrs[name]=value
        else:
            super(DataFolder,self).__setattr__(name,value)            
            
            
    def __setitem__(self,name,value):
        """Set a DataFile or DataFolder backinto the DataFolder.

        Args:
            name (int or string): The index of the DataFile or Folder to be replaced.
            value (DataFile or DataFolder): The data to be stored

        Returns:
            None

        The method operates in two modes, depending on whether the supplied value is a :py:class:`Stoner.Core.DataFile` or :py:class:`DataFolder`.

        If the value is a :py:class:`Stoner.Core.DataFile`, then the corresponding entry in the files attriobute
        is written. The name in this case may be either a string or an integer. In the former case, the string is compared
        to the :py:attr:`DataFolder.ls`  list of filenames and then to the :py:attr:`DataFolder.basenames` attroibute to
        determine which entry should be replaced. If there is no match, then the new DataFile is imply appended after its
        :py:attr:`Stopner.Core.DataFile.filename` attribute is et to the name parameter. If name is an integer then it is used
        simply as a numerioc index into the :py:attr:`DataFolder.files` atttribute.

        If the value is a :py:class:`Stoner.Core.DataFolder`, then the name must be a string and is used to index into the
        :py:attr:`DataFolder.groups`.
        """
        if not isinstance(value,(DataFolder,DataFile)):
            raise TypError("Can only store DataFile like objects and DataFolders in a DataFolder")
        if isinstance(value,DataFile):
            if isinstance(name,int):
                self.files[name]=value
            elif isinstance(name,string_types):
                if name in self.ls:
                    self.files[self.ls.index(name)]
                elif name in self.basenames:
                    self.files[self.basenames.index(name)]
                else:
                    value.filename=name
                    self.files.append(value)
            else:
                raise KeyError("Cannot workout how to use {} as a key".format(name))
        elif isinstance(value,DataFolder):
            if isinstance(name,string_types):
                self.groups[name]=value
            else:
                raise KeyError("Cannot use {} to index a group".format(name))

    def __sub__(self,other):
        """Implements a subtraction operator."""
        result=copy(self)
        to_del=list()
        if isinstance(other,DataFolder):
            for f in other.ls:
                if f in result.ls:
                    to_del.append(result.ls.index(f))
            for i in to_del.sort(reverse=True):
                del result[i]
        elif isinstance(other,DataFile) and other.filename in result.ls:
            del result[result.ls.index(other.filename)]
        elif isinstance(other,string_types) and other in result.ls:
            del result[result.ls.index(other)]
        else:
            result=NotImplemented
        return result

    #######################################################################
    ###################### Private Methods ################################
    #######################################################################

    def _dialog(self, message="Select Folder",  new_directory=True):
        """Creates a directory dialog box for working with

        Keyword Arguments:
            message (string): Message to display in dialog
            new_directory (bool): True if allowed to create new directory

        Returns:
            A directory to be used for the file operation."""
        # Wildcard pattern to be used in file dialogs.
        if isinstance(self.directory, string_types):
            dirname = self.directory
        else:
            dirname = os.getcwd()
        if not self.multifile:
            mode="directory"
        else:
            mode="files"
        dlg = get_filedialog(what=mode)
        if len(dlg)!=0:
            if not self.multifile:
                self.directory = dlg
                ret=self.directory
            else:
                ret=None
        else:
            self.pattern=[path.basename(name) for name in dlg]
            self.directory = path.commonprefix(dlg)
            ret = self.directory
        return ret

    def _pathsplit(self,pathstr, maxsplit=1):
        """split relative path into list"""
        path = [pathstr]
        while True:
            oldpath = path[:]
            path[:1] = list(os.path.split(path[0]))
            if path[0] == '':
                path = path[1:]
            elif path[1] == '':
                path = path[:1] + path[2:]
            if path == oldpath:
                return path
            if maxsplit is not None and len(path) > maxsplit:
                return path

    def _pruner_(self,grp,breadcrumb):
        """Removes any empty groups fromthe DataFolder tree."""
        if len(grp)==0:
            self._pruneable.append(breadcrumb)
            ret=True
        else:
            ret=False
        return ret

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
        tmp= self.type(f,**self.extra_args)
        if not isinstance(tmp.filename,string_types):
            tmp.filename=path.basename(f)
        for p in self.pattern:
            if isinstance(p,re._pattern_type) and (p.search(tmp.filename) is not None):
                m=p.search(tmp.filename)
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
                    tmp[h]=_np_.mean(tmp.column(h))
        tmp['Loaded from']=tmp.filename
        for k in self._file_attrs:
            tmp.__setattr__(k,self._file_attrs[k])
        return tmp

    def _removeDisallowedFilenameChars(filename):
        """Utility method to clean characters in filenames

        Args:
            filename (string): filename to cleanse

        Returns:
            A filename with non ASCII characters stripped out
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
                    tmp.filename="{}-{}".format(g,tmp.filename)
                    self.files.append(tmp)
                    ret.append(tmp)
            for g in removeGroups:
                del(self.groups[g])
            return ret
        else:
            if group:
                return walker(self,breadcrumb,**walker_args)
            else:
                return [walker(f,breadcrumb,**walker_args) for f in self]

    ##################################################################################
    ############# Public Methods #####################################################
    ##################################################################################

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
            for k in self.__dict__:
                if k not in ["files","groups"]:
                    self.groups[key].__dict__[k]=self.__dict__[k]
            self.groups[key].key=key
        return self


    def concatentate(self,sort=None,reverse=False):
        """Concatentates all the files in a DataFolder into a single DataFile like object.

        Keyword Arguments:
            sort (column index, None or bool, or clallable function): Sort the resultant DataFile by this column (if a column index),
                or by the *x* column if None or True, or not at all if False. *sort* is passed directly to the eponymous method as the
                *order* paramter.
            reverse (bool): Reverse the order of the sort (defaults to False)

        Returns:
            The current DataFolder with only one DataFile item containing all the data.
        """
        for d in self[1:]:
            self[0]+=d
        del self[1:]

        if not isinstance(sort,bool) or sort:
            if isinstance(sort, bool) or sort is None:
                sort=self[0].setas["x"]
            self[0].sort(order=sort,reverse=True)

        return self

    def extract(self,metadata):
        """Walks through the terminal group and gets the listed metadata from each file and constructsa replacement DataFile.

        Args:
            metadata (list): List of metadata indices that should be used to construct the new data file.

        Returns:
            An instance of a DataFile like object.
        """

        def _extractor(group,trail,metadata):

            results=group.type()
            results.metadata=group[0].metadata

            ok_data=list
            for m in metadata: # Sanity check the metadata to include
                try:
                    test=_np_.array(results[m])
                except:
                    continue
                else:
                    ok_data.append(m)
                    results.column_headers.extend([m]*len(test))

            row=_np_.array([])
            for d in group:
                for m in ok_data:
                    row=_np_.append(row,_np_array(d[m]))
                results+=row

            return results

        return self.walk_groups(_extractor,group=True,replace_terminal=True,walker_args={"metadata":metadata})

    def filter(self, filter=None,  invert=False):
        """Filter the current set of files by some criterion

        Args:
            filter (string or callable): Either a string flename pattern or a callable function which takes a single parameter x which is an instance of a DataFile and evaluates True or False
            invert (bool): Invert the sense of the filter (done by doing an XOR whith the filter condition
        Returns:
            The current DataFolder object"""

        files=[]
        if isinstance(filter, string_types):
            for f in self.files:
                if fnmatch.fnmatch(f, filter)  ^ invert:
                    files.append(f)
        elif isinstance(filter, re._pattern_type):
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


    def filterout(self, filter):
        """Synonym for self.filter(filter,invert=True)

        Args:
        filter (string or callable): Either a string flename pattern or a callable function which takes a single parameter x which is an instance of a DataFile and evaluates True or False

        Returns:
            The current DataFolder object with the files in the file list filtered."""
        return self.filter(filter, invert=True)


    def flatten(self, depth=None):
        """Compresses all the groups and sub-groups iunto a single flat file list.
        
        Keyword Arguments:
            depth )(int or None): Only flatten ub-=groups that are within (*depth* of the deepest level.

        Returns:
            A copy of the now flattened DatFolder"""
        if isinstance(depth,int):
            if self.depth<=depth:
                self.flatten()
            else:
                for g in self.groups:
                    self.groups[g].flatten(depth)
        else:
            for g in self.groups:
                self.groups[g].flatten()
                self.files.extend(self.groups[g].files)
            self.groups={}
        return self


    def gather(self,xcol=None,ycol=None):
        """Collects xy and y columns from the subfiles in the final group in the tree and builds iunto a :py:class:`Stoner.Core.DataFile`

        Keyword Arguments:
            xcol (index or None): Column in each file that has x data. if None, then the setas settings are used
            ycol (index or None): Column(s) in each filwe that contain the y data. If none, then the setas settings are used.

        Notes:
            This is a wrapper around walk_groups that assembles the data into a single file for further analysis/plotting.

        """
        def _gatherer(group,trail,xcol=None,ycol=None):
            yerr=None
            xerr=None
            if xcol is None and ycol is None:
                lookup=True
                cols=group[0]._get_cols()
                xcol=cols["xcol"]
                ycol=cols["ycol"]
                if  cols["has_xerr"]:
                    xerr=cols["xerr"]
                if cols["has_yerr"]:
                    yerr=cols["yerr"]
            else:
                lookup=False

            xcol=group[0].find_col(xcol)
            ycol=group[0].find_col(ycol)

            results=group.type()
            results.metadata=group[0].metadata
            xbase=group[0].column(xcol)
            xtitle=group[0].column_headers[xcol]
            results&=xbase
            results.column_headers[0]=xtitle
            setas=["x"]
            if cols["has_xerr"]:
                xerrdata=group[0].column(xerr)
                results&=xerrdata
                results.column_headers[-1]="Error in {}".format(xtitle)
                setas.append("d")
            for f in group:
                if lookup:
                    cols=f._get_cols()
                    xcol=cols["xcol"]
                    ycol=cols["ycol"]
                    zcol=cols["zcol"]
                xdata=f.column(xcol)
                ydata=f.column(ycol)
                if _np_.any(xdata!=xbase):
                    results&=xdata
                    results.column_headers[-1]="{}:{}".format(path.basename(f.filename),f.column_headers[xcol])
                    xbase=xdata
                    setas.append("x")
                    if cols["has_xerr"]:
                        xerr=cols["xerr"]
                        if _np_.any(f.column(xerr)!=xerrdata):
                            xerrdata=f.column(xerr)
                            results&=xerrdata
                            results.column_headers[-1]="{}:{}".format(path.basename(f.filename),f.column_headers[xerr])
                            setas.append("d")
                for i in range(len(ycol)):
                    results&=ydata[:,i]
                    setas.append("y")
                    results.column_headers[-1]="{}:{}".format(path.basename(f.filename),f.column_headers[ycol[i]])
                    if cols["has_yerr"]:
                        yerr=cols["yerr"][i]
                        results&=f.column(yerr)
                        results.column_headers[-1]="{}:{}".format(path.basename(f.filename),f.column_headers[yerr])
                        setas.append("e")
                if len(zcol)>0:
                    zdata=f.column(zcol)
                    for i in range(len(zcol)):
                        results&=zdata[:,i]
                        setas.append("z")
                        results.column_headers[-1]="{}:{}".format(path.basename(f.filename),f.column_headers[zcol[i]])
                        if cols["has_zerr"]:
                            yerr=cols["zerr"][i]
                            results&=f.column(zerr)
                            results.column_headers[-1]="{}:{}".format(path.basename(f.filename),f.column_headers[zerr])
                            setas.append("f")
            results.setas=setas
            return results

        return self.walk_groups(_gatherer,group=True,replace_terminal=True,walker_args={"xcol":xcol,"ycol":ycol})


    def getlist(self, recursive=None, directory=None,flatten=None):
        """Scans the current directory, optionally recursively to build a list of filenames

        Keyword Arguments:
            recursive (bool): Do a walk through all the directories for files
            directory (string or False): Either a string path to a new directory or False to open a dialog box or not set in which case existing directory is used.
            flatten (bool): After scanning the directory tree, flaten all the subgroupos to make a flat file list. (this is the previous behaviour of
            :py:meth:`DataFolder.getlist()`)

        Returns:
            A copy of the current DataFoder directory with the files stored in the files attribute

        getlist() scans a directory tree finding files that match the pattern. By default it will recurse through the entire
        directory tree finding sub directories and creating groups in the data folder for each sub directory.
        """
        self.files=[]
        if recursive is None:
            recursive=self.recursive
        if flatten is None:
            flatten=self.flat
        if isinstance(directory,  bool) and not directory:
            self._dialog()
        elif isinstance(directory, string_types):
            self.directory=directory
            if self.multifile:
                self._dialog()
        if isinstance(self.directory, bool) and not self.directory:
            self._dialog()
        elif self.directory is None:
            self.directory=os.getcwd()
        root=self.directory
        dirs=[]
        files=[]
        for f in os.listdir(root):
            if path.isdir(path.join(root, f)):
                dirs.append(f)
            elif path.isfile(path.join(root, f)):
                files.append(f)
        for p in self.pattern: # pattern is a list of strings and regeps
            if isinstance(p,string_types):
                for f in fnmatch.filter(files, p):
                    self.files.append(path.join(root, f))
                    # Now delete the matched file from the list of candidates
                    #This stops us double adding fles that match multiple patterns
                    del(files[files.index(f)])
            if isinstance(p,re._pattern_type):
                matched=[]
                # For reg expts we iterate over all files, but we can't delete matched
                # files as we go as we're iterating over them - so we store the
                # indices and delete them later.
                for f in files:
                    if p.search(f):
                        self.files.append(path.join(root,f))
                        matched.append(files.index(f))
                matched.sort(reverse=True)
                for i in matched: # reverse sort the matching indices to safely delete
                    del(files[i])
        if recursive:
            for d in dirs:
                self.add_group(d)
                self.groups[d].directory=path.join(root,d)
                self.groups[d].getlist(recursive=recursive,flatten=flatten)
        if flatten:
            self.flatten()
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
        if isinstance(key, string_types):
            k=key
            key=lambda x:x[k]
        for f in self.ls:
            x=self[f]
            v=key(x)
            self.add_group(v)
            self.groups[v].files.append(x)
        self.files=[]
        if len(next_keys)>0:
            for g in self.groups:
                self.groups[g].group(next_keys)
        return self

    def keys(self):
        """An alias for self.lsgrp as a gwenerator."""
        for g in self.lsgrp:
            yield g


    def prune(self):
        """Remove any groups from the DataFolder (and subgroups).

        Returns:
            A copy of thte pruned DataFolder."""
        self._pruneable=[] # slightly ugly to avoid modifying whilst iterating
        self.walk_groups(self._pruner_,group=True)
        while len(self._pruneable)!=0:
            for p in self._pruneable:
                pth=tuple(p[:-1])
                item=p[-1]
                if len(pth)==0:
                    del self[item]
                else:
                    grp=self[pth]
                    del grp[item]
            self._pruneable=[]
            self.walk_groups(self._pruner_,group=True)
        del self._pruneable
        return self



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


    def select(self,*args,**kargs):
        """A generator that can be used to select particular data files from the DataFolder

        Args:
            args (various): A single positional argument if present is interpreted as follows:

            * If a callable function is given, the entire DataFile is presented to it.
                If it evaluates True then that DataFile is selected. This allows arbitary select operations
            * If a dict is given, then it and the kargs dictionary are merged and used to select the DataFiles

        Keyword Arguments:
            kargs (varuous): Arbitary keyword arguments are interpreted as requestion matches against the corresponding
                metadata values. The value of the argument is used as follows:

            * if is a scalar, then an equality test is carried out
            * If is a list then a membership test is carried out
            * if it is a tuple of numbers then it is interpreted as a bounds test (t1<=x<t2)

        Yields:
            A DataFile that matches the select requirements
        """
        if len(args)!=0:
            arg=args[0]
            if callable(arg):
                mode="call"
            elif isinstance(arg,dict):
                kargs.update(arg)
                mode="dict"
            else:
                raise RuntimeError("Bad select specification")
        else:
            mode="dict"
        for f in self:
            if mode=="call":
                result=arg(f)
            elif mode=="dict":
                result=True
                for k in kargs:
                    v=kargs[k]
                    if isinstance(v,tuple) and len(v)==2:
                        l1=v[0]
                        l2=v[1]
                        result&=l1<=f[k]<l2
                    elif isinstance(v,tuple) and len(v)==1:
                        v=v[0]
                    if isinstance(v,list):
                        result&=f[k] in v
                    else:
                        result&=f[k]==v
            else:
                raise RuntimeError("oops what happened here?")
            if result:
                yield f


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
        if isinstance(key, string_types):
            k=[(self[i].get(key),i) for i in range(len(self.files))]
            k=sorted(k,reverse=reverse)
            self.files=[self.files[y] for (x,y) in k]
        elif key is None:
            fnames=self.ls
            fnames.sort(reverse=reverse)
            self.files=[self[f] for f in fnames]
        elif isinstance(key,re._pattern_type):
            self.files=sorted(self.files,cmp=lambda x, y:cmp(key.match(x).groups(),key.match(y).groups()), reverse=reverse)
        else:
            self.files=sorted(self.files,cmp=lambda x, y:cmp(key(self[x]), key(self[y])), reverse=reverse)
        return self

    def unflatten(self):
        """Takes a file list an unflattens them according to the file paths.

        Returns:
            A copy of the DataFolder
        """
        self.directory=path.commonprefix(self.ls)
        if self.directory[-1]!=path.sep:
            self.directory=path.dirname(self.directory)
        relpaths=[path.relpath(f,self.directory) for f in self.ls]
        dels=list()
        for i,f in enumerate(relpaths):
            grp=path.split(f)[0]
            if grp!=f and grp!="":
                self.add_group(grp)
                self.groups[grp]+=self[i]
                dels.append(i)
        for i in sorted(dels,reverse=True):
            del self[i]
        for g in self.groups:
            self.groups[g].unflatten()

    def walk_groups(self, walker, group=False, replace_terminal=False,walker_args={}):
        """Walks through a heirarchy of groups and calls walker for each file.

        Args:
            walker (callable): a callable object that takes either a DataFile instance or a DataFolder instance.

        Keyword Arguments:
            group (bool): (default False) determines whether the walker function will expect to be given the DataFolder
                representing the lowest level group or individual DataFile objects from the lowest level group
            replace_terminal (bool): if group is True and the walker function returns an instance of DataFile then the return value is appended
                to the files and the group is removed from the current DataFolder. This will unwind the group heirarchy by one level.
            walker_args (dict): a dictionary of static arguments for the walker function.

        Notes:
            The walker function should have a prototype of the form:
                walker(f,list_of_group_names,**walker_args)
                where f is either a DataFolder or DataFile."""
        return self.__walk_groups(walker,group=group,replace_terminal=replace_terminal,walker_args=walker_args,breadcrumb=[])


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

class PlotFolder(DataFolder):
    """A subclass of :py:class:`DataFolder` with extra methods for plotting lots of files."""
    
    def plot(self,*args,**kargs):
        """Call the plot method for each DataFile, but switching to a subplot each time.
        
        Args:
            args: Positional arguments to pass through to the :py:meth:`Stoner.Plot.PlotFile.plot` call.
            kargs: Keyword arguments to pass through to the :py:meth:`Stoner.Plot.PlotFile.plot` call.
            
        Returns:
            A list of :py:class:`matplotlib.pyplot.Axes` instances.
            
        Notes:
            If the underlying type of the :py:class:`Stoner.Core.DataFile` instances in the :py:class:`PlotFolder`
            lacks a **plot** method, then the instances are converted to :py:class:`Stoner.Util.Data`.
        
            Each plot is generated as sub-plot on a page. The number of rows and columns of subplots is computed
            from the aspect ratio of the figure and the number of files in the :py:class:`PlotFolder`.
        """        
        plts=len(self)
        
        if not hasattr(self.type,"plot"): # switch the objects to being Stoner.Data instances
            from Stoner import Data
            for i,d in enumerate(self):
                self[i]=Data(d)
        
        fig_num=kargs.pop("figure",None)
        fig_args={}
        for arg in ("figsize", "dpi", "facecolor", "edgecolor", "frameon", "FigureClass"):
            if arg in kargs:
                fig_args[arg]=kargs.pop(arg)
        if fig_num is None:
            fig=plt.figure(**fig_args)
        else:
            fig=plt.figure(fig_num,**fig_args)
        w,h=fig.get_size_inches()
        plt_x=_np_.floor(_np_.sqrt(plts*w/h))
        plt_y=_np_.ceil(plts/plt_x)

        kargs["figure"]=fig
        ret=[]
        for i,d in enumerate(self):
            ax=plt.subplot(plt_y,plt_x,i+1)
            ret.append(d.plot(*args,**kargs))
        plt.tight_layout()
        return ret