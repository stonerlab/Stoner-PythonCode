"""
 FStoner.Folders : Classes for working collections of data files

 Classes:
     :py:class:`objectFolder` - manages a list of individual data files (e.g. from a directory tree)
"""

__all__ = ["objectFolder","DataFolder","PlotFolder"]
from .compat import *
import os
import re
import os.path as path
import fnmatch
import numpy as _np_
from copy import copy
import unicodedata
import string
from collections import Iterable,MutableSequence,MutableMapping,OrderedDict
from inspect import ismethod
import matplotlib.pyplot as plt
from .Core import metadataObject,DataFile


regexp_type=(re._pattern_type,)

class regexpDict(OrderedDict):
    """An ordered dictionary that permits looks up by regular expression."""
    def __init__(self,*args,**kargs):
        super(regexpDict,self).__init__(*args,**kargs)

    def __lookup__(self,name):
        """Lookup name and find a matching key or raise KeyError.

        Parameters:
            name (str, re._pattern_type): The name to be searched for

        Returns:
            Canonical key matching the specified name.

        Raises:
            KeyError: if no key matches name.
        """
        if super(regexpDict,self).__contains__(name):
            return name
        if isinstance(name,string_types):
            nm=re.compile(name)
        elif isinstance(name,int_types): #We can do this because we're an OrderedDict!
            return list(self.keys())[name]
        else:
            nm=name
        if isinstance(nm,re._pattern_type):
            for n in self.keys():
                if nm.match(n):
                        return n

        raise KeyError("{} is not a match to any key.".format(name))


    def __getitem__(self,name):
        """Adds a lookup via regular expression when retrieving items."""
        return super(regexpDict,self).__getitem__(self.__lookup__(name))

    def __setitem__(self,name,value):
        """Overwrites any matching key, or if not found adds a new key."""
        try:
            key=self.__lookup__(name)
        except KeyError:
            if not isinstance(name,string_types):
                raise KeyError("{} is not a match to any key.".format(name))
            key=name
        OrderedDict.__setitem__(self, key, value)

    def __delitem__(self,name):
        """Deletes keys that match by regular expression as well as exact matches"""
        super(regexpDict,self).__delitem__(self.__lookup__(name))

    def __contains__(self,name):
        """Returns True if name either is an exact key or matches when interpreted as a regular experssion."""
        try:
            name=self.__lookup__(name)
            return True
        except KeyError:
            return False

class baseFolder(MutableSequence):
    """A base class for objectFolders that supports both a sequence of objects and a mapping of instances of itself.

    Attributes:
        groups(regexpDict): A dictionary of similar baseFolder instances
        objects(regexptDict): A dictionary of metadataObjects
        _index(list): An index of the keys associated with objects
    """
    _mro_list=None

    def __init__(self,*args,**kargs):
        """Initialise the baseFolder.

        Notes:
            - Creates empty groups and objects stres
            - Sets all keyword arguments as attributes unless otherwise overwriting an existing attribute
            - stores other arguments in self.args
            - iterates over the multuiple inheritance tree and eplaces any interface methods with ones from
                the mixin classes
            - calls the mixin init methods.
            """
        self.debug=kargs.get("debug",False)
        self.groups=regexpDict()
        self.objects=regexpDict()
        self._type=metadataObject
        self._iface={}
        self._object_attrs=dict()
        self.args=copy(args)
        self.kargs=copy(kargs)
        #List of routines that define the interface for manipulating the objects stored in the folder
        interface_routines=["__init__","__clone__","__getter__","__setter__","__deleter__","__lookup__","__names__","__clear__"]
        for k in list(self.kargs.keys()): # Store keyword parameters as attributes
            if not hasattr(self,k) or k in ["type","kargs","args"]:
                value=kargs.pop(k,None)
                self.__setattr__(k,value)
                if self.debug: print("Setting self.{} to {}".format(k,value))
        for c in self._mro: # Iterate over the multiple inheritance  run order
            if self.debug: print("Looking at {}".format(c.__name__))
            if c is baseFolder:
                continue
            for method in interface_routines: # Look for methods implemented in a mixin
                if self.debug: print("Examining {} {}".format(c.__name__,method))
                if method in c.__dict__:
                    if self.debug: print("Ok, need a routine to set...")
                    if method not in self._iface or not isinstance(self._iface[method],list):
                        lst=[]
                    else:
                        lst=self._iface[method]
                    lst.append(getattr(c,method))
                    self._iface[method]=lst
                    if self.debug: print("{} is now {}".format(method,self._iface[method]))
            #Now call the init method of the mixin classes
            if c.__module__.startswith("Stoner") and not issubclass(c,baseFolder):
                if self.debug: print("Initing {}".format(c))
                c.__init__(self)

    ###########################################################################
    ################### Properties of baseFolder ##############################

    @classproperty
    def _mro(self):
        _mro_list = []
        for c in self.__mro__[1:]:
            if c not in _mro_list:
                _mro_list.append(c)
        return _mro_list

    @property
    def depth(self):
        """Gives the maximum number of levels of group below the current objectFolder."""
        if len(self.groups)==0:
            r=0
        else:
            r=1
            for g in self.groups:
                r=max(r,self.groups[g].depth+1)
        return r

    @property
    def files(self):
        """Return an iterator of potentially unloaded named objects."""
        return [self.__getter__(i,instantiate=False) for i in range(len(self))]

    @files.setter
    def files(self,value):
        """Just a wrapper to clear and then set the objects."""
        if isinstance(value,Iterable):
            self.__clear__()
            for i,v in enumerate(value):
                self.insert(i,v)

    @property
    def loaded(self):
        """An iterator that indicates wether the contents of the :py:class:`Stoner.Folders.objectFolder` has been
        loaded into memory."""
        for f in self.__names__():
            yield isinstance(self.__getter__(f,instantiate=False),metadataObject)

    @property
    def ls(self):
        return self.__names__()

    @property
    def lsgrp(self):
        """Returns a list of the groups as a generator."""
        for k in self.groups.keys():
            yield k

    @property
    def mindepth(self):
        """Gives the minimum number of levels of group below the current objectFolder."""
        if len(self.groups)==0:
            r=0
        else:
            r=1E6
            for g in self.groups:
                r=min(r,self.groups[g].depth+1)
        return r

    @property
    def not_empty(self):
        """An iterator for objectFolder that checks whether the loaded metadataObject objects have any data.

        Returns the next non-empty DatFile member of the objectFolder.

        Note:
            not_empty will also silently skip over any cases where loading the metadataObject object will raise
            and exception."""
        for i in range(len(self)):
            try:
                d=self[i]
            except:
                continue
            if len(d)==0:
                continue
            yield(d)

    @property
    def type(self):
        """Defines the (sub)class of the :py:class:`Stoner.Core.metadataObject` instances."""
        return self._type

    @type.setter
    def type(self,value):
        """Ensures that type is a subclass of metadataObject."""
        if issubclass(value,metadataObject):
            self._type=value
        elif isinstance(value,metadataObject):
            self._type=value.__class__
        else:
            raise TypeError("{} os neither a subclass nor instance of metadataObject".format(type(value)))

    ################### Methods for subclasses to override to handle storage #####
    def __lookup__(self,name):
        """Stub for other classes to implement.
        Parameters:
            name(str): Name of an object

        Returns:
            A key in whatever form the :py:meth:`baseFolder.__getter__` will accept.
        """
        for method in self._iface.get("__lookup__",[]):
            try:
                return method(self,name)
            except NotImplemented:
                continue
        return name

    def __names__(self):
        """Stub method to return a list of names of all objects that can be indexed for __getter__."""
        for method in self._iface.get("__names__",[]):
            try:
                return method(self)
            except NotImplemented:
                continue

        return list(self.objects.keys())

    def __getter__(self,name,instantiate=True):
        """Stub method to do whatever is needed to transform a key to a metadataObject.

        Parameters:
            name (key type): The canonical mapping key to get the dataObject. By default
                the baseFolder class uses a :py:class:`regexpDict` to store objects in.

        Keyword Arguments:
            instatiate (bool): IF True (default) then always return a metadataObject. If False,
                the __getter__ method may return a key that can be used by it later to actually get the
                metadataObject.

        Returns:
            (metadataObject): The metadataObject
        """
        for method in self._iface.get("__getter__",[]):
            try:
                return method(self,name,instantiate=instantiate)
            except NotImplementedError:
                continue
        return self.objects[name]

    def __setter__(self,name,value):
        """Stub to setting routine to store a metadataObject.
        Parameters:
            name (string) the named object to write - may be an existing or new name
            value (metadataObject) the value to store."""
        for method in self._iface.get("__setter__",[]):
            try:
                method(self,name,value)
                break
            except NotImplemented:
                continue
        else:
            self.objects[name]=value

    def __deleter__(self,ix):
        """Deletes an object from the baseFolder.

        Parameters:
            ix(str): Index to delete, should be within +- the lengthe length of the folder.
        """
        for method in self._iface.get("__deleter__",[]):
            try:
                method(self,name)
                break
            except NotImplemented:
                continue
        else:
            del self.objects[ix]

    def __clear__(self):
        for method in self._iface.get("__clear__",[]):
            try:
                method(self)
                break
            except NotImplemented:
                continue
        else:
            for n in self.__names__():
                self.__deleter__(self.__lookup__(n))

    def __clone__(self,other=None):
        """Do whatever is necessary to copy attributes from self to other."""
        if other is None:
            other=self.__class__()
        other.args=self.args
        other.kargs=self.kargs
        other.type=self.type
        for k in self.kargs:
            if not hasattr(other,k):
                setattr(other,k,self.kargs[k])
        for method in self._iface.get("__clone__",[]):
            method(self,other)
        return other



    ###########################################################################
    ######## Methods to implement the MutableMapping abstract methods #########
    ######## And to provide a mapping interface that mainly access groups #####

    def __getitem__(self,name):
        """Try to get either a group or an object.

        Parameters:
            name(str, int): If name is a string then it is checked first against the groups
                and then against the objects dictionaries - both will fall back to a regular
                expression if necessary. If name is an int, then the _index attribute is used to
                find a matching object key.

        Returns:
            Either a baseFolder instance or a metadataObject instance or raises KeyError
        """
        if isinstance(name,string_types+regexp_type):
            if name in self.groups:
                return self.groups[name]
            elif name in self.objects:
                name=self.__lookup__(name)
                return self.__getter__(name)
            else:
                raise KeyError("{} is neither a group name nor object name.".format(name))
        elif isinstance(name,int_types):
            if -len(self)<name<len(self):
                return self.__getter__(self.__lookup__(self.__names__()[name]))
            else:
                raise IndexError("{} is out of range.".format(name))
        else:
            raise KeyError("Can't index the baseFolder with {}",format(name))

    def __setitem__(self,name,value):
        """Attempts to store a value in either the groups or objects.

        Parameters:
            name(str or int): If the name is a string and the value is a baseFolder, then assumes we're accessing a group.
                if name is an integer, then it must be a metadataObject.
            value (baseFolder,metadataObject,str): The value to be storred.
        """
        if isinstance(name,string_types):
            if isinstance(value,baseFolder):
                self.groups[name]=value
        else:
            self.__setter__(self.__lookup__(name),value)
        if isinstance(name,int_types):
            if -len(self)<name<len(self):
                self.__setter__(self.__lookup__(self.__names__()[name]),value)
            else:
                raise IndexError("{} is out of range".format(name))
        else:
            raise KeyError("{} is not a valid key for baseFolder".format(name))

    def __delitem__(self,name):
        """Attempt to delete an item from either a group or list of files.

        Parameters:
            name(str,int): IF name is a string, then it is checked first against the groups and then
                against the objects. If name is an int then it s checked against the _index.
        """
        if isinstance(name,string_types):
            if name in self.groups:
                del self.groups[name]
            elif name in self.objects:
                self.__deleter__(self.__lookup__(name))
            else:
                raise KeyError("{} doesn't match either a group or object.".format(name))
        elif isinstance(name,int_types):
            if -len(self)<name<=len(self):
                self.__deleter__(self.__lookup__(self.__names__()[name]))
            else:
                raise IndexError("{} is out of range.".format(name))
        else:
            raise KeyError("Can't use {} as a key to delete from baseFolder.".format(name))

    def __contains__(self,name):
        """Check whether name is in a list of groups or in the list of names"""
        return name in self.groups or name in self.__names__()

    def __len__(self):
        return len(self.__names__())

    ###########################################################################
    ###################### Standard Special Methods ###########################


    def __add_core__(self,result,other):
        if isinstance(other,baseFolder):
            if issubclass(other.type,self.type):
                result.extend([f for f in other.files])
                result.groups.update(other.groups)
            else:
                raise RuntimeError("Incompatible types ({} must be a subclass of {}) in the two folders.".format(other.type,result.type))
        elif isinstance(other,result.type):
            result.append(self.type(other))
        else:
            result=NotImplemented
        return result

    def __add__(self,other):
        """Implement the addition operator for baseFolder and metadataObjects."""
        result=copy(self)
        result=self.__add_core__(result,other)
        return result

    def __iadd__(self,other):
        """Implement the addition operator for baseFolder and metadataObjects."""
        result=self
        result=self.__add_core__(result,other)
        return result
        

    def __getattr__(self, item):
        """Handles some special case attributes that provide alternative views of the objectFolder

        Args:
            item (string): The attribute name being requested

        Returns:
            Depends on the attribute

        """
        if hasattr(self,item) or item.startswith("_"): # bypass for local attrs and private attrs
            ret=getattr(super(baseFolder,self),item)
        else:
            instance=self._type()
            if hasattr(instance,item): #Something is in our metadataObject type
                if callable(getattr(instance,item,None)): # It's a method
                    ret=self.__getattr_proxy(item)
                else: # It's a static attribute
                    if item in self._object_attrs:
                        ret=self._object_attrs[item]
                    else:
                        ret=getattr(self[0],item,None)
            else: # Ok, pass back
                ret=getattr(super(baseFolder,self),item)
        return ret

    def __getattr_proxy(self,item):
        """Make a prpoxy call to access a method of the metadataObject like types.

        Args:
            item (string): Name of method of metadataObject class to be called

        Returns:
            Either a modifed copy of this objectFolder or a list of return values
            from evaluating the method for each file in the Folder.
        """
        meth=getattr(self._type(),item,None)
        def _wrapper_(*args,**kargs):
            """Wraps a call to the metadataObject type for magic method calling.
            Note:
                This relies on being defined inside the enclosure of the objectFolder method
                so we have access to self and item"""
            retvals=[]
            for ix,f in enumerate(self):
                meth=getattr(f,item,None)
                ret=meth(*args,**kargs)
                if ret is not f: # method did not returned a modified version of the metadataObject
                    retvals.append(ret)
                if isinstance(ret,self._type):
                    self[ix]=ret
            if len(retvals)==0: # If we haven't got anything to retun, return a copy of our objectFolder
                retvals=self
            return retvals
        #Ok that's the wrapper function, now return  it for the user to mess around with.
        _wrapper_.__doc__=meth.__doc__
        _wrapper_.__name__=meth.__name__
        return _wrapper_



    def __repr__(self):
        """Prints a summary of the objectFolder structure

        Returns:
            A string representation of the current objectFolder object"""
        s="objectFolder({}) with pattern {} has {} files and {} groups\n".format(self.directory,self.pattern,len(self),len(self.groups))
        for g in self.groups: # iterate over groups
            r=self.groups[g].__repr__()
            for l in r.split("\n"): # indent each line by one tab
                s+="\t"+l+"\n"
        return s.strip()

    def __setattr__(self,name,value):
        """Pass through to set the sample attributes."""
        if name.startswith("_"): # pass ddirectly through for private attributes
            super(baseFolder,self).__setattr__(name,value)
        elif hasattr(self,name) and not callable(getattr(self,name,None)):
            super(baseFolder,self).__setattr__(name,value)
        elif hasattr(self,"_type") and name in dir(self._type()):
            self._object_attrs[name]=value
        else:
            super(baseFolder,self).__setattr__(name,value)



    ###########################################################################
    ###################### Private Methods ####################################

    def _pruner_(self,grp,breadcrumb):
        """Removes any empty groups fromthe objectFolder tree."""
        if len(grp)==0:
            self._pruneable.append(breadcrumb)
            ret=True
        else:
            ret=False
        return ret

    def __walk_groups(self,walker,group=False,replace_terminal=False,only_terminal=True,walker_args={},breadcrumb=[]):
        """"Actually implements the walk_groups method,m but adds the breadcrumb list of groups that we've already visited.

        Args:
            walker (callable): a callable object that takes either a metadataObject instance or a objectFolder instance.

        Keyword Arguments:
            group (bool): (default False) determines whether the wealker function will expect to be given the objectFolder
                representing the lowest level group or individual metadataObject objects from the lowest level group
            replace_terminal (bool): if group is True and the walker function returns an instance of metadataObject then the return value is appended
                to the files and the group is removed from the current objectFolder. This will unwind the group heirarchy by one level.
            only_terminal (bool): Only iterate over the files in the group if the group has no sub-groups.
            walker_args (dict): a dictionary of static arguments for the walker function.
            bbreadcrumb (list of strings): a list of the group names or key values that we've walked through

        Notes:
            The walker function should have a prototype of the form:
                walker(f,list_of_group_names,**walker_args)
                where f is either a objectFolder or metadataObject."""
        if (len(self.groups)>0):
            ret=[]
            removeGroups=[]
            if replace_terminal:
                self.__clear__()
            for g in self.groups:
                bcumb=copy(breadcrumb)
                bcumb.append(g)
                tmp=self.groups[g].__walk_groups(walker,group=group,replace_terminal=replace_terminal,walker_args=walker_args,breadcrumb=bcumb)
                if group and  replace_terminal and isinstance (tmp, metadataObject):
                    removeGroups.append(g)
                    tmp.filename="{}-{}".format(g,tmp.filename)
                    self.append(tmp)
                    ret.append(tmp)
            for g in removeGroups:
                del(self.groups[g])
        elif len(self.groups)==0 or not only_terminal:
            if group:
                ret=walker(self,breadcrumb,**walker_args)
            else:
                ret=[walker(f,breadcrumb,**walker_args) for f in self]
        return ret


    ###########################################################################
    ############# Normal Methods ##############################################

    def add_group(self,key):
        """Add a new group to the current baseFolder with the given key.

        Args:
            key(string): A hashable value to be used as the dictionary key in the groups dictionary
        Returns:
            A copy of the objectFolder

        Note:
            If key already exists in the groups dictionary then no action is taken.

        Todo:
            Propagate any extra attributes into the groups.
        """
        if key in self.groups: # do nothing here
            pass
        else:
            new_group=self.__clone__()
            self.groups[key]=new_group
            self.groups[key].key=key
        return self

    def clear(self):
        """Clear the subgroups."""
        self.groups.clear()
        self.__clear__()

    def get(self,name,default=None):
        """Return either a sub-group or named object from this folder."""
        try:
            ret=self[name]
        except (KeyError,IndexError):
            ret=default
        return ret

    def filter(self, filter=None,  invert=False):
        """Filter the current set of files by some criterion

        Args:
            filter (string or callable): Either a string flename pattern or a callable function which takes a single parameter x which is an instance of a metadataObject and evaluates True or False
            invert (bool): Invert the sense of the filter (done by doing an XOR whith the filter condition
        Returns:
            The current objectFolder object"""

        names=[]
        if isinstance(filter, string_types):
            for f in self.__names__():
                if fnmatch.fnmatch(f, filter)  ^ invert:
                    names.append(f)
        elif isinstance(filter, re._pattern_type):
            for f in self.__names__():
                if filter.search(f) is not None:
                    names.append(f)
        elif filter is None:
            raise ValueError("A filter must be defined !")
        else:
            for i,x in enumerate(self):
                if filter(x)  ^ invert:
                    names.append(self.files[i])
        self.files=files
        return self

    def filterout(self, filter):
        """Synonym for self.filter(filter,invert=True)

        Args:
        filter (string or callable): Either a string flename pattern or a callable function which takes a single parameter x which is an instance of a metadataObject and evaluates True or False

        Returns:
            The current objectFolder object with the files in the file list filtered."""
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
                self.extend([
                    self.groups[g].__getter__(self.groups[g].__lookup__(n),instantiate=False)
                    for n in self.groups[g].__names__()])
            self.groups={}
        return self

    def group(self, key):
        """Take the files and sort them into a series of separate objectFolder objects according to the value of the key

        Args:
            key (string or callable or list): Either a simple string or callable function or a list. If a string then it is interpreted as an item of metadata in each file. If a callable function then
                takes a single argument x which should be an instance of a metadataObject and returns some vale. If key is a list then the grouping is done recursively for each element
                in key.
        Returns:
            A copy of the current objectFolder object in which the groups attribute is a dictionary of objectFolder objects with sub lists of files

        If ne of the grouping metadata keys does not exist in one file then no exception is raised - rather the fiiles will be returned into the group with key None. Metadata keys that
        are generated from the filename are supported."""
        if isinstance(key, list):
            next_keys=key[1:]
            key=key[0]
        else:
            next_keys=[]
        if isinstance(key, string_types):
            k=key
            key=lambda x:x[k]
        for x in self:
            v=key(x)
            self.add_group(v)
            self.groups[v].append(x)
        self.__clear__()
        if len(next_keys)>0:
            for g in self.groups:
                self.groups[g].group(next_keys)
        return self

    def insert(self,ix,value):
        """Implements the insert method with the option to append as well."""
        print(len(self),ix)
        if -len(self)<ix<len(self):
            name=self.__names__()[ix]
            self.__setter__(self.__lookup__(name),value)
        elif ix>=len(self):
            name=value.filename if hasattr(value,"filename") else value if isinstance(value,string_types) else "object {}".format(len(self))
            i=1
            names=self.__names__()
            while name in names: # Since we're adding a new entry, make sure we have a unique name !
                name=value.filename if hasattr(value,"filename") else value if isinstance(value,string_types) else "object {}".format(len(self))
                name,ext=os.path.splitext(name)
                name="{}({}).{}".format(name,i,ext)
                i+=1
            self.__setter__(name,value)

    def items(self):
        """Return the key,value pairs for the subbroups of this folder."""
        return self.groups.items()

    def keys(self):
        """Return the keys used to access the sub-=groups of this folder."""
        return self.groups.keys()

    def pop(self,name=-1,default=None):
        """Return and remove either a subgroup or named object from this folder."""
        try:
            ret=self[name]
            del self[name]
        except (KeyError,IndexError):
            ret=default
        return ret

    def popitem(self):
        """Return the most recent subgroup from this folder."""
        return self.groups.popitem()

    def prune(self):
        """Remove any groups from the objectFolder (and subgroups).

        Returns:
            A copy of thte pruned objectFolder."""

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

    def select(self,*args, **kargs):
        """A generator that can be used to select particular data files from the objectFolder

        Args:
            args (various): A single positional argument if present is interpreted as follows:

            * If a callable function is given, the entire metadataObject is presented to it.
                If it evaluates True then that metadataObject is selected. This allows arbitary select operations
            * If a dict is given, then it and the kargs dictionary are merged and used to select the metadataObjects

        Keyword Arguments:
            recurse (bool): Also recursively slect through the sub groups
            kargs (varuous): Arbitary keyword arguments are interpreted as requestion matches against the corresponding
                metadata values. The keyword argument may have an additional *__operator** appended to it which is interpreted
                as follows:

                - *eq* metadata value equals argument value (this is the default test for scalar argument)
                - *ne* metadata value doe not equal argument value
                - *gt* metadata value doe greater than argument value
                - *lt* metadata value doe less than argument value
                - *ge* metadata value doe greater than or equal to argument value
                - *le* metadata value doe less than or equal to argument value
                - *contains* metadata value contains argument value (this is the default test for non-tuple iterable arguments)
                - *startswith* metadata value startswith argument value
                - *endswith* metadata value endwith argument value
                - *icontains*,*istartswith*,*iendswith* as above but case insensitive
                - *between* metadata value lies beween the minimum and maximum values of the arguement (the default test for 2-length tuple arguments)
                - *ibetween*,*ilbetween*,*iubetween* as above but include both,lower or upper values

            The syntax is inspired by the Django project for selecting, but is not quite as rich.

        Returns:
            (baseFGolder): a new baseFolder instance that contains just the matching metadataObjects.

        Note:
            If any of the tests is True, then the metadataObject will be selected, so the effect is a logical OR. To
            achieve a logical AND, you can chain two selects together::

                d.select(temp__le=4.2,vti_temp__lt=4.2).select(field_gt=3.0)

            will select metadata objects that have either temp or vti_temp metadata values below 4.2 AND field metadata values greater than 3.

            If you need to select on a aparameter called *recurse*, pass a dictionary of {"recurse":value} as the sole
            positional argument. If you need to select on a metadata value that ends in an operator word, then append
            *__eq* in the keyword name to force the equality test. If the metadata keys to select on are not valid python identifiers,
            then pass them via the first positional dictionary value.
        """
        recurse=kargs.pop("recurse",False)
        if len(args)==1:
            if callable(args[0]):
                kargs["__"]=args[0]
            elif isinstance(args[0],dict):
                kargs.update(args[0])
        operator={
            "eq":lambda k,v:k==v,
            "ne":lambda k,v:k!=v,
            "contains":lambda k,v: k in v,
            "icontains":lambda k,v: k.upper() in str(v).upper(),
            "lt":lambda k,v:k<v,
            "le":lambda k,v:k<=v,
            "gt":lambda k,v:k>v,
            "ge":lambda k,v:k>=v,
            "between":lambda k,v: min(v)<k<max(v),
            "ibetween":lambda k,v: min(v)<=k<=max(v),
            "ilbetween":lambda k,v: min(v)<=k<max(v),
            "iubetween":lambda k,v: min(v)<k<=max(v),
            "startswith":lambda k,v:str(v).startswith(k),
            "istartswith":lambda k,v:str(v).upper().startswith(k.upper()),
            "endsswith":lambda k,v:str(v).endswith(k),
            "iendsswith":lambda k,v:str(v).upper().endswith(k.upper()),
        }
        result=self.__clone__
        if recurse:
            gkargs={}
            gkargs.update(kargs)
            gkargs["recurse"]=True
            for g in self.groups:
                result.groups[g]=self.groups[g].select(*args,**gkargs)
        for f in self:
            for arg in kargs:
                if callable(kargs[arg]) and kargs[arg](f):
                    break
                elif isinstance(arg,string_types):
                    parts=arg.split("__")
                    if parts[-1] in operator and len(parts)>1:
                        arg="__".join(parts[:-1])
                        op=parts[-1]
                    else:
                        if isinstance(kargs[arg],tuple) and len(kargs[arg]==2):
                            op="between" #Assume two length tuples are testing for range
                        elif not isinstance(kargs[arg],string_types) and isinstance(kargs[arg],Iterable):
                            op="contains" # Assume other iterables are testing for memebership
                        else: #Everything else is exact matches
                            op="eq"
                    func=operator[op]
                    if arg in f and func(kargs[arg],f[arg]):
                        break
            else: # No tests matched - contineu to next line
                continue
            #Something matched, so append to result
            result.append(f)
        return result

    def setdefault(self,k,d=None):
        """Return or set a subgroup or named object."""
        self[k]=self.get(k,d)
        return self[k]

    def sort(self, key=None, reverse=False):
        """Sort the files by some key

        Keyword Arguments:
            key (string, callable or None): Either a string or a callable function. If a string then this is interpreted as a
                metadata key, if callable then it is assumed that this is a a function of one paramater x
                that is a :py:class:`Stoner.Core.metadataObject` object and that returns a key value.
                If key is not specified (default), then a sort is performed on the filename

        reverse (bool): Optionally sort in reverse order

        Returns:
            A copy of the current objectFolder object"""
        if isinstance(key, string_types):
            k=[(x.get(key),i) for x,i in enumerate(self)]
            k=sorted(k,reverse=reverse)
            new_order=[self[i] for x,i in k]
        elif key is None:
            fnames=self.__names__()
            fnames.sort(reverse=reverse)
            new_order=[self.__getter__(name,instantiate=False) for name in fnames]
        elif isinstance(key,re._pattern_type):
            new_order=sorted(self,cmp=lambda x, y:cmp(key.match(x).groups(),key.match(y).groups()), reverse=reverse)
        else:
            new_order=sorted(self,cmp=lambda x, y:cmp(key(self[x]), key(self[y])), reverse=reverse)
        self.__clear__()
        self.extend(new_order)
        return self


    def update(self,other):
        """Update this folder with a dictionary or another folder."""
        if isinstance(other,dict):
            for k in other:
                self[k]=other[k]
        elif isinstance(other,baseFolder):
            for k in other.groups:
                self.groups[k]=other.groups[k]
            for k in self.__names__():
                self.__setter__(self.__lookup__(k),other.__getter__(other.__lookup__(k)))

    def values(self):
        """Return the sub-groups of this folder."""
        return self.groups.values()

    def walk_groups(self, walker, group=False, replace_terminal=False,only_terminal=True,walker_args={}):
        """Walks through a heirarchy of groups and calls walker for each file.

        Args:
            walker (callable): a callable object that takes either a metadataObject instance or a objectFolder instance.

        Keyword Arguments:
            group (bool): (default False) determines whether the walker function will expect to be given the objectFolder
                representing the lowest level group or individual metadataObject objects from the lowest level group
            replace_terminal (bool): if group is True and the walker function returns an instance of metadataObject then the return value is appended
                to the files and the group is removed from the current objectFolder. This will unwind the group heirarchy by one level.
            obly_terminal(bool): Only execute the walker function on groups that have no sub-groups inside them (i.e. are terminal groups)
            walker_args (dict): a dictionary of static arguments for the walker function.

        Notes:
            The walker function should have a prototype of the form:
                walker(f,list_of_group_names,**walker_args)
                where f is either a objectFolder or metadataObject."""

        return self.__walk_groups(walker,group=group,replace_terminal=replace_terminal,only_terminal=only_terminal,walker_args=walker_args,breadcrumb=[])

    def zip_groups(self, groups):
        """Return a list of tuples of metadataObjects drawn from the specified groups

        Args:
            groups(list of strings): A list of keys of groups in the Lpy:class:`objectFolder`

        ReturnsL
            A list of tuples of groups of files: [(grp_1_file_1,grp_2_file_1....grp_n_files_1),(grp_1_file_2,grp_2_file_2....grp_n_file_2)....(grp_1_file_m,grp_2_file_m...grp_n_file_m)]
        """
        if not isinstance(groups, list):
            raise SyntaxError("groups must be a list of groups")
        grps=[[y for y in self.groups[x]] for x in groups]
        return zip(*grps)

class DiskBssedFolder(object):
    """A Mixin class that implmenets reading metadataObjects from disc."""

    def __init__(self,*args,**kargs):
        from Stoner import Data
        defaults={"type":Data,
                  "extra_args":dict(),
                  "pattern":["*.*"],
                  "read_means":False,
                  "recursive":True,
                  "flattern":False,
                  "directory":os.getcwd(),
                  "multiple":False,
                  "readlist":True,
                  }
        for k in defaults:
            setattr(self,k,self.kargs.get(k,defaults[k]))
        if self.readlist:
            pass

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
            grp (:py:class:`objectFolder` or :py:calss:`Stoner.metadataObject`): A group or file to save
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

    def __getter__(self,name,instantiate=True):
        """Stub method to do whatever is needed to transform a key to a metadataObject.

        Parameters:
            name (key type): The canonical mapping key to get the dataObject. By default
                the baseFolder class uses a :py:class:`regexpDict` to store objects in.

        Keyword Arguments:
            instatiate (bool): IF True (default) then always return a metadataObject. If False,
                the __getter__ method may return a key that can be used by it later to actually get the
                metadataObject.

        Returns:
            (metadataObject): The metadataObject
        """
        if not instantiate or not path.exists(name): #If we're not try to instantiate this object then let the parent do the work
            raise NotImplementedError()
        if isinstance(self.objects[name],metadataObject):
            return self.objects[name]
        tmp= self.type(name,**self.extra_args)
        if not hasattr(tmp,"filename") or not isinstance(tmp.filename,string_types):
            tmp.filename=path.basename(name)
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
        for k in self._object_attrs:
            tmp.__setattr__(k,self._object_attrs[k])
        self.__setter__(name,tmp)
        return tmp

    @property
    def basenames(self):
        """Returns a list of just the filename parts of the objectFolder."""
        ret=[]
        for x in self.__names__():
            ret.append(path.basename(x))
        return ret

    @property
    def pattern(self):
        return self._pattern

    @pattern.setter
    def pattern(self,value):
        """Sets the filename searching pattern(s) for the :py:class:`Stoner.Core.metadataObject`s."""
        if isinstance(value,string_types):
            self._pattern=(value,)
        elif isinstance(value,re._pattern_type):
            self._pattern=(value,)
        elif isinstance(value,Iterable):
            self._pattern=[x for x in value]
        else:
            raise ValueError("pattern should be a string, regular expression or iterable object not a {}".format(type(value)))


    def getlist(self, recursive=None, directory=None,flatten=None):
        """Scans the current directory, optionally recursively to build a list of filenames

        Keyword Arguments:
            recursive (bool): Do a walk through all the directories for files
            directory (string or False): Either a string path to a new directory or False to open a dialog box or not set in which case existing directory is used.
            flatten (bool): After scanning the directory tree, flaten all the subgroupos to make a flat file list. (this is the previous behaviour of
            :py:meth:`objectFolder.getlist()`)

        Returns:
            A copy of the current DataFoder directory with the files stored in the files attribute

        getlist() scans a directory tree finding files that match the pattern. By default it will recurse through the entire
        directory tree finding sub directories and creating groups in the data folder for each sub directory.
        """
        self.__clear__()
        if recursive is None:
            recursive=self.recursive
        if flatten is None:
            flatten=self.flatten
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
                    self.append(path.join(root, f))
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
                        self.__setter__(path.join(root,f),path.join(root,f))
                        matched.append(files.index(f))
                matched.sort(reverse=True)
                for i in matched: # reverse sort the matching indices to safely delete
                    del(files[i])
        if recursive:
            for d in dirs:
                if self.debug: print("Entering directory {}".format(d))
                self.add_group(d)
                self.groups[d].directory=path.join(root,d)
                self.groups[d].getlist(recursive=recursive,flatten=flatten)
        if flatten:
            self.flatten()
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

    def unflatten(self):
        """Takes a file list an unflattens them according to the file paths.

        Returns:
            A copy of the objectFolder
        """
        self.directory=path.commonprefix(self.__names__())
        if self.directory[-1]!=path.sep:
            self.directory=path.dirname(self.directory)
        relpaths=[path.relpath(f,self.directory) for f in self.__names__()]
        dels=list()
        for i,f in enumerate(relpaths):
            grp=path.split(f)[0]
            if grp!=f and grp!="":
                self.add_group(grp)
                self.groups[grp].append([i])
                dels.append(i)
        for i in sorted(dels,reverse=True):
            del self[i]
        for g in self.groups:
            self.groups[g].unflatten()



class objectFolder(baseFolder,DiskBssedFolder):
    """Implements a class that manages lists of data files (e.g. the contents of a directory) and can sort and group them in arbitary ways

    Attributes:
        basenames (list of str): Returns the list of files after passing through os.path.basename()
        depth (int): Maximum number of levels of groups below this :py:class:`objectFolder`.
        directory (string): Root directory of the files handled by the objectFolder
        extra_args (dict): Extra Arguments to pass to the constructors of the :py:class:`Stoner.Core.metadataObject`
           objects.
        files (list): List of filenames or loaded :py:class:`Stoner.metadataObject` instances
        groups (dict of :py:class:`objectFolder`) Represent a heirarchy of Folders
        ls (list of str): Returns a list of filenames (either the matched filename patterns, or
            :py;attr:`Stoner.Core.metadataObject.filename` if objectFolder.files contains metadataObject objects
        loaded (list of bool): Inidicates which fiels are loaded in memory for the :py:class:`objectFolder`
        lsgrp (list of str): Returns a list of the group keys (equivalent to objectFolder.groups.keys()
        mindepth (int): Minimum number of levels of groups from this :py:class:`objectFolder` to a terminal group.
        multifile (bool): if True a multi-file dialog box is used to load several files from the same folder
        pattern (string or re or sequence of strings and re): Matches which files in the  directory tree are included.
            If pattern is a compiled reular expression with named groups then the named groups are used to
            generate metadata in the :py:class:`Stoner.metadataObject` object. If pattern is a list, then the set of
            files included in the py:class:`objectFolder` is the union of files that match any single pattern.
        read_means (bool): If true, create metadata when reading each file that is the mean of each column
        setas (list or string): Sets the default value of the :py:attr:`Stoner.Core.metadataObject.setas` attribute for each
            :py:class:`Stoner.Core.metadataObject` in the folder.
        skip_empty (bool): Controls whether iterating over the Folder will skip over empty files. Defaults to False.
        type (metadataObject): The type of the members of the :py:class:`objectFolder`. Can be either a subclass of
            :py:class:`Stoner.Core.metadataObject` or an instance of one (in which ase the class of the instance is used).

    Args:
        directory (string or :py:class:`objectFolder` instance): Where to get the data files from. If False will bring up a dialog for selecting the directory

    Keyword Arguments:
        type (class): An subclass of :py:class:`Stoner.Core.objectFolder` that will be used to construct the individual metadataObject objects in the folder
        pattern (string or re): A filename pattern - either globbing string or regular expression
        nolist (bool): Delay doing a directory scan to get data files
        multifile (bool): if True brings up a dialog for selecting files from a directory.
        read_means (bool): Calculate the average value of each column and add it as metadata entries to the file.

    Returns:
        The newly constructed instance

    Note:
        All other keywords are set as attributes of the objectFolder.

    Todo:
        Handle :py:meth:`__init__(objectFolder)` with subclasses

    """
#
#    _type=metadataObject # class attribute to keep things happy
#    _pattern=None
#    _file_attrs=dict()
#    flat=False
#
#    def __init__(self, *args, **kargs):
#        self.directory=None
#        self.files=[]
#        self.flat=False
#        self.recursive=True
#        self.groups={}
#        self._file_attrs=dict()
#        self.skip_empty=kargs.pop("skip_empty",False)
#        self.pattern=kargs.pop("pattern","*.*")
#        self.nolist=kargs.pop("nolist",len(args)==0)
#        self.multifile=kargs.pop("multifile",False)
#        self.extra_args=kargs.pop("extra_args",{})
#        for v in kargs:
#            self.__setattr__(v,kargs[v])
#        if self.directory is None:
#            self.directory=os.getcwd()
#        if len(args)>0:
#            if isinstance(args[0], string_types):
#                self.directory=args[0]
#                if not self.nolist:
#                    self.getlist()
#            elif isinstance(args[0],bool) and not args[0]:
#                self.directory=False
#                if not self.nolist:
#                    self.getlist()
#            elif isinstance(args[0],objectFolder):
#                other=args[0]
#                for k in other.__dict__:
#                    self.__dict__[k]=other.__dict__[k]
#            else:
#                if not self.nolist:
#                    self.getlist()
#        else:
#            if not self.nolist:
#                self.getlist()
#
#    ################################################################################
#    ####### Property Methods #######################################################
#    ################################################################################
#
#    @property
#    def basenames(self):
#        """Returns a list of just the filename parts of the objectFolder."""
#        ret=[]
#        for x in self.files:
#            if isinstance(x,metadataObject):
#                ret.append(path.basename(x.filename))
#            elif isinstance(x,string_types):
#                ret.append(path.basename(x))
#        return ret
#
#    @property
#    def depth(self):
#        """Gives the maximum number of levels of group below the current objectFolder."""
#        if len(self.groups)==0:
#            r=0
#        else:
#            r=1
#            for g in self.groups:
#                r=max(r,self.groups[g].depth+1)
#        return r
#
#    @property
#    def loaded(self):
#        """An iterator that indicates wether the contents of the :py:class:`Stoner.Folders.objectFolder` has been
#        loaded into memory."""
#        for f in self.files:
#            yield isinstance(f,metadataObject)
#
#    @property
#    def lsgrp(self):
#        """Returns a list of the groups as a generator."""
#        for k in self.groups.keys():
#            yield k
#
#    @property
#    def ls(self):
#        ret=[]
#        for f in self.files:
#            if isinstance(f,string_types):
#                ret.append(f)
#            elif isinstance(f,metadataObject):
#                ret.append(f.filename)
#        return ret
#
#    @property
#    def mindepth(self):
#        """Gives the minimum number of levels of group below the current objectFolder."""
#        if len(self.groups)==0:
#            r=0
#        else:
#            r=1E6
#            for g in self.groups:
#                r=min(r,self.groups[g].depth+1)
#        return r
#
#    @property
#    def pattern(self):
#        return self._pattern
#
#    @pattern.setter
#    def pattern(self,value):
#        """Sets the filename searching pattern(s) for the :py:class:`Stoner.Core.metadataObject`s."""
#        if isinstance(value,string_types):
#            self._pattern=(value,)
#        elif isinstance(value,re._pattern_type):
#            self._pattern=(value,)
#        elif isinstance(value,Iterable):
#            self._pattern=[x for x in value]
#        else:
#            raise ValueError("pattern should be a string, regular expression or iterable object not a {}".format(type(value)))
#
#
#    @property
#    def type(self):
#        """Defines the (sub)class of the :py:class:`Stoner.Core.metadataObject` instances."""
#        return self._type
#
#    @type.setter
#    def type(self,value):
#        """Ensures that type is a subclass of metadataObject."""
#        if issubclass(value,metadataObject):
#            self._type=value
#        elif isinstance(value,metadataObject):
#            self._type=value.__class__
#        else:
#            raise TypeError("{} os neither a subclass nor instance of metadataObject".format(type(value)))
#
#    #########################################################
#    ######## Special Methods ################################
#    #########################################################
#
#    def __add__(self,other):
#        """Implement the addition operator for objectFolder and metadataObjects."""
#        result=copy(self)
#        if isinstance(other,objectFolder):
#            result.files.extend([self.type(f) for f in other.files])
#            result.groups.update(other.groups)
#        elif isinstance(other,metadataObject):
#            result.files.append(self.type(other))
#        else:
#            result=NotImplemented
#        return result
#
#    def __delitem__(self,item):
#        """Deelte and item or a group from the objectFolder
#
#        Args:
#            item(string or int): the Item to be deleted.
#                If item is an int, then assume that it is a file index
#                otherwise it is assumed to be a group key
#        """
#        if isinstance(item, string_types) and item in self.groups:
#            del self.groups[item]
#        elif isinstance(item, int):
#            del self.files[item]
#        elif isinstance(item, slice):
#            indices = item.indices(len(self))
#            for i in reversed(range(*indices)):
#                del self.files[i]
#        else:
#            return NotImplemented
#
#    def __dir__(self):
#        """Returns the attributes of the current object by augmenting the keys of self.__dict__ with the attributes that __getattr__ will handle.
#        """
#        attr=dir(type(self))
#        attr.extend(list(self.__dict__.keys()))
#        attr.extend(dir(self._type))
#        attr=list(set(attr))
#        return attr
#
#
#    def __get_file_attr__(self,item):
#        if item in self._file_attrs:
#            return self._file_attrs[item]
#        else:
#            return super(objectFolder,self).__getattribute__(item)
#
#
#    def __getattr__(self, item):
#        """Handles some special case attributes that provide alternative views of the objectFolder
#
#        Args:
#            item (string): The attribute name being requested
#
#        Returns:
#            Depends on the attribute
#
#        """
#        if not item.startswith("_"):
#            instance=self._type()
#            if item in dir(instance): #Something is in our metadataObject type
#                if callable(getattr(instance,item)): # It's a method
#                    ret=self.__getattr_proxy(item)
#                else: # It's a static attribute
#                    ret=self.__get_file_attr__(item)
#            else: # Ok, pass back
#                ret=super(objectFolder,self).__getattribute__(item)
#        else: # We dpon't intercept private or special methods
#            ret=super(objectFolder,self).__getattribute__(item)
#        return ret
#
#    def __getattr_proxy(self,item):
#        """Make a prpoxy call to access a method of the metadataObject like types.
#
#        Args:
#            item (string): Name of method of metadataObject class to be called
#
#        Returns:
#            Either a modifed copy of this objectFolder or a list of return values
#            from evaluating the method for each file in the Folder.
#        """
#        meth=getattr(self._type(),item)
#        def _wrapper_(*args,**kargs):
#            """Wraps a call to the metadataObject type for magic method calling.
#            Note:
#                This relies on being defined inside the enclosure of the objectFolder method
#                so we have access to self and item"""
#            retvals=[]
#            for ix,f in enumerate(self):
#                meth=getattr(f,item)
#                ret=meth(*args,**kargs)
#                if ret is not f: # method did not returned a modified version of the metadataObject
#                    retvals.append(ret)
#                if isinstance(ret,self._type):
#                    self[ix]=ret
#            if len(retvals)==0: # If we haven't got anything to retun, return a copy of our objectFolder
#                retvals=self
#            return retvals
#        #Ok that's the wrapper function, now return  it for the user to mess around with.
#        _wrapper_.__doc__=meth.__doc__
#        _wrapper_.__name__=meth.__name__
#        return _wrapper_
#
#    def __getitem__(self, i):
#        """Load and returen metadataObject type objects based on the filenames in self.files
#
#        Args:
#            i(int or slice): The index(eces) of the files to return Can also be a string in which case it is interpreted as one of self.files
#
#        Returns:
#            One or more instances of metadataObject objects
#
#        This is the canonical method for producing a metadataObject from a objectFolder. Various substitutions are done as the file is created:
#        1.  Firstly, the filename is inserted into the metadata key "Loaded From"
#        2.  Secondly, if the pattern was given as a regular exression then any named matching groups are
#            interpreted as metadata key-value pairs (where the key will be the name of the capturing
#            group and the value will be the match to the group. This allows metadata to be imported
#            from the filename as it is loaded."""""
#        if isinstance(i,int):
#            files=self.files[i]
#            tmp=self.__read__(files)
#            self.files[i]=tmp
#            return tmp
#        elif isinstance(i, string_types): # Ok we've done a objectFolder['filename']
#            try:
#                i=self.ls.index(i)
#                return self.__read__(self.files[i])
#            except ValueError:
#                try:
#                    i=self.basenames.index(i)
#                except ValueError:
#                    return self.groups[i]
#        elif isinstance(i, slice):
#            indices = i.indices(len(self))
#            return [self[i] for i in range(*indices)]
#        elif isinstance(i,tuple):
#            g=self
#            for ix in i:
#                g=g[ix]
#            return g
#        else:
#            return self.groups[i]
#
#    def __len__(self):
#        """Pass through to return the length of the files array
#
#        Returns:
#            len(self.files)"""
#        return len(self.files)
#
#
#    def __next__(self):
#        """Iterates over contents of objectFolder.
#
#        If :py:attr:`objectFolder.skip_empty` is True, then any members that
#        either faile to load or have zero length are skipped over."""
#        for i in range(len(self.files)):
#            try:
#                ret=self[i]
#                if self.skip_empty and len(ret)==0:
#                    continue
#            except StonerLoadError:
#                if self.skip_empty:
#                    continue
#            else:
#                yield ret
#
#    def next(elf):
#        for i in range(len(self.files)):
#            yield self[i]
#
#    def __repr__(self):
#        """Prints a summary of the objectFolder structure
#
#        Returns:
#            A string representation of the current objectFolder object"""
#        s="objectFolder({}) with pattern {} has {} files and {} groups\n".format(self.directory,self.pattern,len(self.files),len(self.groups))
#        for g in self.groups: # iterate over groups
#            r=self.groups[g].__repr__()
#            for l in r.split("\n"): # indent each line by one tab
#                s+="\t"+l+"\n"
#        return s.strip()
#
#    def __setattr__(self,name,value):
#        """Pass through to set the sample attributes."""
#        if name.startswith("_"): # pass ddirectly through for private attributes
#            super(objectFolder,self).__setattr__(name,value)
#        elif name in self.__dict__ and not callable(getattr(self,name,None)):
#            super(objectFolder,self).__setattr__(name,value)
#        elif name in dir(self._type()):
#            self._file_attrs[name]=value
#        else:
#            super(objectFolder,self).__setattr__(name,value)
#
#
#    def __setitem__(self,name,value):
#        """Set a metadataObject or objectFolder backinto the objectFolder.
#
#        Args:
#            name (int or string): The index of the metadataObject or Folder to be replaced.
#            value (metadataObject or objectFolder): The data to be stored
#
#        Returns:
#            None
#
#        The method operates in two modes, depending on whether the supplied value is a :py:class:`Stoner.Core.metadataObject` or :py:class:`objectFolder`.
#
#        If the value is a :py:class:`Stoner.Core.metadataObject`, then the corresponding entry in the files attriobute
#        is written. The name in this case may be either a string or an integer. In the former case, the string is compared
#        to the :py:attr:`objectFolder.ls`  list of filenames and then to the :py:attr:`objectFolder.basenames` attroibute to
#        determine which entry should be replaced. If there is no match, then the new metadataObject is imply appended after its
#        :py:attr:`Stopner.Core.metadataObject.filename` attribute is et to the name parameter. If name is an integer then it is used
#        simply as a numerioc index into the :py:attr:`objectFolder.files` atttribute.
#
#        If the value is a :py:class:`Stoner.Core.objectFolder`, then the name must be a string and is used to index into the
#        :py:attr:`objectFolder.groups`.
#        """
#        if not isinstance(value,(objectFolder,metadataObject)):
#            raise TypeError("Can only store metadataObject like objects and objectFolders in a objectFolder")
#        if isinstance(value,metadataObject):
#            if isinstance(name,int):
#                self.files[name]=value
#            elif isinstance(name,string_types):
#                if name in self.ls:
#                    self.files[self.ls.index(name)]
#                elif name in self.basenames:
#                    self.files[self.basenames.index(name)]
#                else:
#                    value.filename=name
#                    self.files.append(value)
#            else:
#                raise KeyError("Cannot workout how to use {} as a key".format(name))
#        elif isinstance(value,objectFolder):
#            if isinstance(name,string_types):
#                self.groups[name]=value
#            else:
#                raise KeyError("Cannot use {} to index a group".format(name))
#
#    def __sub__(self,other):
#        """Implements a subtraction operator."""
#        result=copy(self)
#        to_del=list()
#        if isinstance(other,objectFolder):
#            for f in other.ls:
#                if f in result.ls:
#                    to_del.append(result.ls.index(f))
#            for i in to_del.sort(reverse=True):
#                del result[i]
#        elif isinstance(other,metadataObject) and other.filename in result.ls:
#            del result[result.ls.index(other.filename)]
#        elif isinstance(other,string_types) and other in result.ls:
#            del result[result.ls.index(other)]
#        else:
#            result=NotImplemented
#        return result
#
#    #######################################################################
#    ###################### Private Methods ################################
#    #######################################################################
#
#    def _dialog(self, message="Select Folder",  new_directory=True):
#        """Creates a directory dialog box for working with
#
#        Keyword Arguments:
#            message (string): Message to display in dialog
#            new_directory (bool): True if allowed to create new directory
#
#        Returns:
#            A directory to be used for the file operation."""
#        # Wildcard pattern to be used in file dialogs.
#        if isinstance(self.directory, string_types):
#            dirname = self.directory
#        else:
#            dirname = os.getcwd()
#        if not self.multifile:
#            mode="directory"
#        else:
#            mode="files"
#        dlg = get_filedialog(what=mode)
#        if len(dlg)!=0:
#            if not self.multifile:
#                self.directory = dlg
#                ret=self.directory
#            else:
#                ret=None
#        else:
#            self.pattern=[path.basename(name) for name in dlg]
#            self.directory = path.commonprefix(dlg)
#            ret = self.directory
#        return ret
#
#    def _pathsplit(self,pathstr, maxsplit=1):
#        """split relative path into list"""
#        path = [pathstr]
#        while True:
#            oldpath = path[:]
#            path[:1] = list(os.path.split(path[0]))
#            if path[0] == '':
#                path = path[1:]
#            elif path[1] == '':
#                path = path[:1] + path[2:]
#            if path == oldpath:
#                return path
#            if maxsplit is not None and len(path) > maxsplit:
#                return path
#
#    def _pruner_(self,grp,breadcrumb):
#        """Removes any empty groups fromthe objectFolder tree."""
#        if len(grp)==0:
#            self._pruneable.append(breadcrumb)
#            ret=True
#        else:
#            ret=False
#        return ret
#
#    def __read__(self,f):
#        """Reads a single filename in and creates an instance of metadataObject.
#
#        Args:
#            f(string or :py:class:`Stoner.Core.metadataObject`): A filename or metadataObject object
#
#        Returns:
#            A metadataObject object
#
#        Note:
#             If self.pattern is a regular expression then use any named groups in it to create matadata from the
#            filename. If self.read_means is true then create metadata from the mean of the data columns.
#        """
#        if isinstance(f,metadataObject):
#            return f
#        tmp= self.type(f,**self.extra_args)
#        if not isinstance(tmp.filename,string_types):
#            tmp.filename=path.basename(f)
#        for p in self.pattern:
#            if isinstance(p,re._pattern_type) and (p.search(tmp.filename) is not None):
#                m=p.search(tmp.filename)
#                for k in m.groupdict():
#                    tmp.metadata[k]=tmp.metadata.string_to_type(m.group(k))
#        tmp['Loaded from']=tmp.filename
#        for k in self._file_attrs:
#            tmp.__setattr__(k,self._file_attrs[k])
#        return tmp
#
#    def _removeDisallowedFilenameChars(filename):
#        """Utility method to clean characters in filenames
#
#        Args:
#            filename (string): filename to cleanse
#
#        Returns:
#            A filename with non ASCII characters stripped out
#        """
#        validFilenameChars = "-_.() %s%s" % (string.ascii_letters, string.digits)
#        cleanedFilename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore')
#        return ''.join(c for c in cleanedFilename if c in validFilenameChars)
#
#
#    def _save(self,grp,trail,root=None):
#        """Save a group of files to disc by calling the save() method on each file. This internal method is called by walk_groups in turn
#        called from the public save() method. The trail of group keys is used to create a directory tree.
#
#        Args:
#            grp (:py:class:`objectFolder` or :py:calss:`Stoner.metadataObject`): A group or file to save
#            trail (list of strings): the trail of paths used to get here
#            root (string or None): a replacement root directory
#
#        Returns:
#            Saved Path
#        """
#
#        trail=[self._removeDisallowedFilenameChars(t) for t in trail]
#        grp.filename=self._removeDisallowedFilenameChars(grp.filename)
#        if root is None:
#            root=self.directory
#
#        pth=path.join(root,*trail)
#        os.makesdirs(pth)
#        grp.save(path.join(pth,grp.filename))
#        return grp.filename
#
#    def __walk_groups(self,walker,group=False,replace_terminal=False,only_terminal=True,walker_args={},breadcrumb=[]):
#        """"Actually implements the walk_groups method,m but adds the breadcrumb list of groups that we've already visited.
#
#        Args:
#            walker (callable): a callable object that takes either a metadataObject instance or a objectFolder instance.
#
#        Keyword Arguments:
#            group (bool): (default False) determines whether the wealker function will expect to be given the objectFolder
#                representing the lowest level group or individual metadataObject objects from the lowest level group
#            replace_terminal (bool): if group is True and the walker function returns an instance of metadataObject then the return value is appended
#                to the files and the group is removed from the current objectFolder. This will unwind the group heirarchy by one level.
#            only_terminal (bool): Only iterate over the files in the group if the group has no sub-groups.
#            walker_args (dict): a dictionary of static arguments for the walker function.
#            bbreadcrumb (list of strings): a list of the group names or key values that we've walked through
#
#        Notes:
#            The walker function should have a prototype of the form:
#                walker(f,list_of_group_names,**walker_args)
#                where f is either a objectFolder or metadataObject."""
#        if (len(self.groups)>0):
#            ret=[]
#            removeGroups=[]
#            if replace_terminal:
#                self.files=[]
#            for g in self.groups:
#                bcumb=copy(breadcrumb)
#                bcumb.append(g)
#                tmp=self.groups[g].__walk_groups(walker,group=group,replace_terminal=replace_terminal,walker_args=walker_args,breadcrumb=bcumb)
#                if group and  replace_terminal and isinstance (tmp, metadataObject):
#                    removeGroups.append(g)
#                    tmp.filename="{}-{}".format(g,tmp.filename)
#                    self.files.append(tmp)
#                    ret.append(tmp)
#            for g in removeGroups:
#                del(self.groups[g])
#        elif len(self.groups)==0 or not terminal_only:
#            if group:
#                ret=walker(self,breadcrumb,**walker_args)
#            else:
#                ret=[walker(f,breadcrumb,**walker_args) for f in self]
#        return ret
#
#    ##################################################################################
#    ############# Public Methods #####################################################
#    ##################################################################################
#
#    def add_group(self,key):
#        """Add a new group to the current Folder with the given key.
#
#        Args:
#            key(string): A hashable value to be used as the dictionary key in the groups dictionary
#        Returns:
#            A copy of the objectFolder
#
#        Note:
#            If key already exists in the groups dictionary then no action is taken.
#
#        Todo:
#            Propagate any extra attributes into the groups.
#        """
#        if key in self.groups: # do nothing here
#            pass
#        else:
#            self.groups[key]=self.__class__(self.directory, type=self.type, pattern=self.pattern, read_means=self.read_means, nolist=True)
#            for k in self.__dict__:
#                if k not in ["files","groups"]:
#                    self.groups[key].__dict__[k]=self.__dict__[k]
#            self.groups[key].key=key
#        return self
#
#    def filter(self, filter=None,  invert=False):
#        """Filter the current set of files by some criterion
#
#        Args:
#            filter (string or callable): Either a string flename pattern or a callable function which takes a single parameter x which is an instance of a metadataObject and evaluates True or False
#            invert (bool): Invert the sense of the filter (done by doing an XOR whith the filter condition
#        Returns:
#            The current objectFolder object"""
#
#        files=[]
#        if isinstance(filter, string_types):
#            for f in self.files:
#                if fnmatch.fnmatch(f, filter)  ^ invert:
#                    files.append(f)
#        elif isinstance(filter, re._pattern_type):
#            for f in self.files:
#                if filter.search(f) is not None:
#                    files.append(f)
#        elif filter is None:
#            raise ValueError("A filter must be defined !")
#        else:
#            for i in range(len(self.files)):
#                x=self[i]
#                if filter(x)  ^ invert:
#                    files.append(self.files[i])
#        self.files=files
#        return self
#
#
#    def filterout(self, filter):
#        """Synonym for self.filter(filter,invert=True)
#
#        Args:
#        filter (string or callable): Either a string flename pattern or a callable function which takes a single parameter x which is an instance of a metadataObject and evaluates True or False
#
#        Returns:
#            The current objectFolder object with the files in the file list filtered."""
#        return self.filter(filter, invert=True)
#
#
#    def flatten(self, depth=None):
#        """Compresses all the groups and sub-groups iunto a single flat file list.
#
#        Keyword Arguments:
#            depth )(int or None): Only flatten ub-=groups that are within (*depth* of the deepest level.
#
#        Returns:
#            A copy of the now flattened DatFolder"""
#        if isinstance(depth,int):
#            if self.depth<=depth:
#                self.flatten()
#            else:
#                for g in self.groups:
#                    self.groups[g].flatten(depth)
#        else:
#            for g in self.groups:
#                self.groups[g].flatten()
#                self.files.extend(self.groups[g].files)
#            self.groups={}
#        return self
#
#
#    def getlist(self, recursive=None, directory=None,flatten=None):
#        """Scans the current directory, optionally recursively to build a list of filenames
#
#        Keyword Arguments:
#            recursive (bool): Do a walk through all the directories for files
#            directory (string or False): Either a string path to a new directory or False to open a dialog box or not set in which case existing directory is used.
#            flatten (bool): After scanning the directory tree, flaten all the subgroupos to make a flat file list. (this is the previous behaviour of
#            :py:meth:`objectFolder.getlist()`)
#
#        Returns:
#            A copy of the current DataFoder directory with the files stored in the files attribute
#
#        getlist() scans a directory tree finding files that match the pattern. By default it will recurse through the entire
#        directory tree finding sub directories and creating groups in the data folder for each sub directory.
#        """
#        self.files=[]
#        if recursive is None:
#            recursive=self.recursive
#        if flatten is None:
#            flatten=self.flat
#        if isinstance(directory,  bool) and not directory:
#            self._dialog()
#        elif isinstance(directory, string_types):
#            self.directory=directory
#            if self.multifile:
#                self._dialog()
#        if isinstance(self.directory, bool) and not self.directory:
#            self._dialog()
#        elif self.directory is None:
#            self.directory=os.getcwd()
#        root=self.directory
#        dirs=[]
#        files=[]
#        for f in os.listdir(root):
#            if path.isdir(path.join(root, f)):
#                dirs.append(f)
#            elif path.isfile(path.join(root, f)):
#                files.append(f)
#        for p in self.pattern: # pattern is a list of strings and regeps
#            if isinstance(p,string_types):
#                for f in fnmatch.filter(files, p):
#                    self.files.append(path.join(root, f))
#                    # Now delete the matched file from the list of candidates
#                    #This stops us double adding fles that match multiple patterns
#                    del(files[files.index(f)])
#            if isinstance(p,re._pattern_type):
#                matched=[]
#                # For reg expts we iterate over all files, but we can't delete matched
#                # files as we go as we're iterating over them - so we store the
#                # indices and delete them later.
#                for f in files:
#                    if p.search(f):
#                        self.files.append(path.join(root,f))
#                        matched.append(files.index(f))
#                matched.sort(reverse=True)
#                for i in matched: # reverse sort the matching indices to safely delete
#                    del(files[i])
#        if recursive:
#            for d in dirs:
#                self.add_group(d)
#                self.groups[d].directory=path.join(root,d)
#                self.groups[d].getlist(recursive=recursive,flatten=flatten)
#        if flatten:
#            self.flatten()
#        return self
#
#
#    def group(self, key):
#        """Take the files and sort them into a series of separate objectFolder objects according to the value of the key
#
#        Args:
#            key (string or callable or list): Either a simple string or callable function or a list. If a string then it is interpreted as an item of metadata in each file. If a callable function then
#                takes a single argument x which should be an instance of a metadataObject and returns some vale. If key is a list then the grouping is done recursively for each element
#                in key.
#        Returns:
#            A copy of the current objectFolder object in which the groups attribute is a dictionary of objectFolder objects with sub lists of files
#
#        If ne of the grouping metadata keys does not exist in one file then no exception is raised - rather the fiiles will be returned into the group with key None. Metadata keys that
#        are generated from the filename are supported."""
#        self.groups={}
#        if isinstance(key, list):
#            next_keys=key[1:]
#            key=key[0]
#        else:
#            next_keys=[]
#        if isinstance(key, string_types):
#            k=key
#            key=lambda x:x[k]
#        for f in self.ls:
#            x=self[f]
#            v=key(x)
#            self.add_group(v)
#            self.groups[v].files.append(x)
#        self.files=[]
#        if len(next_keys)>0:
#            for g in self.groups:
#                self.groups[g].group(next_keys)
#        return self
#
#    def insert(self,index,value):
#        """Implements the insert method to support MutableSequence.
#
#        Parameters:
#            index (integer): Position before which new value will be inserted.
#            value (metadataObject, or string): New value to be inserted.
#
#        Returns:
#            Modifield objectFolder"""
#        if isinstance(value,string_types) or isinstance(value,metadataObject):
#            self.files.insert(index,value)
#        else:
#            raise TypeError("Can't store a {} in a {}".format(type(value),type(self)))
#        return self
#
#
#
#    def keys(self):
#        """An alias for self.lsgrp as a gwenerator."""
#        for g in self.lsgrp:
#            yield g
#
#
#    def not_empty(self):
#        """An iterator for objectFolder that checks whether the loaded metadataObject objects have any data.
#
#        Returns the next non-empty DatFile member of the objectFolder.
#
#        Note:
#            not_empty will also silently skip over any cases where loading the metadataObject object will raise
#            and exception."""
#        for i in range(len(self)):
#            try:
#                d=self[i]
#            except:
#                continue
#            if len(d)==0:
#                continue
#            yield(d)
#
#    def prune(self):
#        """Remove any groups from the objectFolder (and subgroups).
#
#        Returns:
#            A copy of thte pruned objectFolder."""
#        self._pruneable=[] # slightly ugly to avoid modifying whilst iterating
#        self.walk_groups(self._pruner_,group=True)
#        while len(self._pruneable)!=0:
#            for p in self._pruneable:
#                pth=tuple(p[:-1])
#                item=p[-1]
#                if len(pth)==0:
#                    del self[item]
#                else:
#                    grp=self[pth]
#                    del grp[item]
#            self._pruneable=[]
#            self.walk_groups(self._pruner_,group=True)
#        del self._pruneable
#        return self
#
#
#
#    def save(self,root=None):
#        """Save the entire data folder out to disc using the groups as a directory tree,
#        calling the save method for each file in turn.
#
#        Args:
#            root (string): The root directory to start creating files and subdirectories under. If set to None or not specified, the current folder's
#                diretory attribute will be used.
#        Returns:
#            A list of the saved files
#        """
#        return self.walk_groups(self._save,walker_args={"root",root})
#
#
#    def select(self,*args,**kargs):
#        """A generator that can be used to select particular data files from the objectFolder
#
#        Args:
#            args (various): A single positional argument if present is interpreted as follows:
#
#            * If a callable function is given, the entire metadataObject is presented to it.
#                If it evaluates True then that metadataObject is selected. This allows arbitary select operations
#            * If a dict is given, then it and the kargs dictionary are merged and used to select the metadataObjects
#
#        Keyword Arguments:
#            kargs (varuous): Arbitary keyword arguments are interpreted as requestion matches against the corresponding
#                metadata values. The value of the argument is used as follows:
#
#            * if is a scalar, then an equality test is carried out
#            * If is a list then a membership test is carried out
#            * if it is a tuple of numbers then it is interpreted as a bounds test (t1<=x<t2)
#
#        Yields:
#            A metadataObject that matches the select requirements
#        """
#        if len(args)!=0:
#            arg=args[0]
#            if callable(arg):
#                mode="call"
#            elif isinstance(arg,dict):
#                kargs.update(arg)
#                mode="dict"
#            else:
#                raise RuntimeError("Bad select specification")
#        else:
#            mode="dict"
#        for f in self:
#            if mode=="call":
#                result=arg(f)
#            elif mode=="dict":
#                result=True
#                for k in kargs:
#                    v=kargs[k]
#                    if isinstance(v,tuple) and len(v)==2:
#                        l1=v[0]
#                        l2=v[1]
#                        result&=l1<=f[k]<l2
#                    elif isinstance(v,tuple) and len(v)==1:
#                        v=v[0]
#                    if isinstance(v,list):
#                        result&=f[k] in v
#                    else:
#                        result&=f[k]==v
#            else:
#                raise RuntimeError("oops what happened here?")
#            if result:
#                yield f
#
#
#    def sort(self, key=None, reverse=False):
#        """Sort the files by some key
#
#        Keyword Arguments:
#            key (string, callable or None): Either a string or a callable function. If a string then this is interpreted as a
#                metadata key, if callable then it is assumed that this is a a function of one paramater x
#                that is a :py:class:`Stoner.Core.metadataObject` object and that returns a key value.
#                If key is not specified (default), then a sort is performed on the filename
#
#        reverse (bool): Optionally sort in reverse order
#
#        Returns:
#            A copy of the current objectFolder object"""
#        if isinstance(key, string_types):
#            k=[(self[i].get(key),i) for i in range(len(self.files))]
#            k=sorted(k,reverse=reverse)
#            self.files=[self.files[y] for (x,y) in k]
#        elif key is None:
#            fnames=self.ls
#            fnames.sort(reverse=reverse)
#            self.files=[self[f] for f in fnames]
#        elif isinstance(key,re._pattern_type):
#            self.files=sorted(self.files,cmp=lambda x, y:cmp(key.match(x).groups(),key.match(y).groups()), reverse=reverse)
#        else:
#            self.files=sorted(self.files,cmp=lambda x, y:cmp(key(self[x]), key(self[y])), reverse=reverse)
#        return self
#
#    def unflatten(self):
#        """Takes a file list an unflattens them according to the file paths.
#
#        Returns:
#            A copy of the objectFolder
#        """
#        self.directory=path.commonprefix(self.ls)
#        if self.directory[-1]!=path.sep:
#            self.directory=path.dirname(self.directory)
#        relpaths=[path.relpath(f,self.directory) for f in self.ls]
#        dels=list()
#        for i,f in enumerate(relpaths):
#            grp=path.split(f)[0]
#            if grp!=f and grp!="":
#                self.add_group(grp)
#                self.groups[grp]+=self[i]
#                dels.append(i)
#        for i in sorted(dels,reverse=True):
#            del self[i]
#        for g in self.groups:
#            self.groups[g].unflatten()
#
#    def walk_groups(self, walker, group=False, replace_terminal=False,only_terminal=True,walker_args={}):
#        """Walks through a heirarchy of groups and calls walker for each file.
#
#        Args:
#            walker (callable): a callable object that takes either a metadataObject instance or a objectFolder instance.
#
#        Keyword Arguments:
#            group (bool): (default False) determines whether the walker function will expect to be given the objectFolder
#                representing the lowest level group or individual metadataObject objects from the lowest level group
#            replace_terminal (bool): if group is True and the walker function returns an instance of metadataObject then the return value is appended
#                to the files and the group is removed from the current objectFolder. This will unwind the group heirarchy by one level.
#            obly_terminal(bool): Only execute the walker function on groups that have no sub-groups inside them (i.e. are terminal groups)
#            walker_args (dict): a dictionary of static arguments for the walker function.
#
#        Notes:
#            The walker function should have a prototype of the form:
#                walker(f,list_of_group_names,**walker_args)
#                where f is either a objectFolder or metadataObject."""
#        return self.__walk_groups(walker,group=group,replace_terminal=replace_terminal,only_terminal=only_terminal,walker_args=walker_args,breadcrumb=[])
#
#
#    def zip_groups(self, groups):
#        """Return a list of tuples of metadataObjects drawn from the specified groups
#
#        Args:
#            groups(list of strings): A list of keys of groups in the Lpy:class:`objectFolder`
#
#        ReturnsL
#            A list of tuples of groups of files: [(grp_1_file_1,grp_2_file_1....grp_n_files_1),(grp_1_file_2,grp_2_file_2....grp_n_file_2)....(grp_1_file_m,grp_2_file_m...grp_n_file_m)]
#        """
#        if not isinstance(groups, list):
#            raise SyntaxError("groups must be a list of groups")
#        grps=[[y for y in self.groups[x]] for x in groups]
#        return zip(*grps)

class DataFolder(objectFolder):


    def __init__(self,*args,**kargs):
        from Stoner import Data
        self.type=kargs.pop("type",Data)
        self.read_means=kargs.pop("read_means",False)
        super(DataFolder,self).__init__(*args,**kargs)

    def __read__(self,f):
        """Reads a single filename in and creates an instance of metadataObject.

        Args:
            f(string or :py:class:`Stoner.Core.metadataObject`): A filename or metadataObject object

        Returns:
            A metadataObject object

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


    def concatentate(self,sort=None,reverse=False):
        """Concatentates all the files in a objectFolder into a single metadataObject like object.

        Keyword Arguments:
            sort (column index, None or bool, or clallable function): Sort the resultant metadataObject by this column (if a column index),
                or by the *x* column if None or True, or not at all if False. *sort* is passed directly to the eponymous method as the
                *order* paramter.
            reverse (bool): Reverse the order of the sort (defaults to False)

        Returns:
            The current objectFolder with only one metadataObject item containing all the data.
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
        """Walks through the terminal group and gets the listed metadata from each file and constructsa replacement metadataObject.

        Args:
            metadata (list): List of metadata indices that should be used to construct the new data file.

        Returns:
            An instance of a metadataObject like object.
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

    def gather(self,xcol=None,ycol=None):
        """Collects xy and y columns from the subfiles in the final group in the tree and builds iunto a :py:class:`Stoner.Core.metadataObject`

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


class PlotFolder(DataFolder):
    """A subclass of :py:class:`objectFolder` with extra methods for plotting lots of files."""

    def plot(self,*args,**kargs):
        """Call the plot method for each metadataObject, but switching to a subplot each time.

        Args:
            args: Positional arguments to pass through to the :py:meth:`Stoner.plot.PlotMixin.plot` call.
            kargs: Keyword arguments to pass through to the :py:meth:`Stoner.plot.PlotMixin.plot` call.

        Returns:
            A list of :py:class:`matplotlib.pyplot.Axes` instances.

        Notes:
            If the underlying type of the :py:class:`Stoner.Core.metadataObject` instances in the :py:class:`PlotFolder`
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