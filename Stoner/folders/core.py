# -*- coding: utf-8 -*-
"""Provide the base classes and functions for the :py:class:`Stoner.DataFolder` class."""
__all__ = ["baseFolder"]

from collections.abc import Iterable, MutableSequence
import fnmatch
import re
from itertools import islice
from copy import copy, deepcopy
from inspect import isclass
import os.path as path

import numpy as np

from ..compat import int_types, string_types, commonpath, _pattern_type
from ..tools import operator, isiterable, all_type, get_option
from ..core.base import regexpDict, TypeHintedDict
from ..core.base import metadataObject

from .utils import pathjoin
from .each import Item as EachItem
from .metadata import MetadataProxy
from .groups import GroupsDict

regexp_type = (_pattern_type,)


def _add_core_(result, other):
    """Implement the core logic of the addition operator.

    Note:
        We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!
    """
    if isinstance(other, baseFolder):
        if isclass(other.type) and issubclass(other.type, result.type):
            result.extend(list(other.files))
            for grp in other.groups:
                if grp in result.groups:
                    result.groups[grp] += other.groups[grp]  # recursely merge groups
                else:
                    result.groups[grp] = copy(other.groups[grp])
        else:
            raise RuntimeError(
                f"Incompatible types ({other.type} must be a subclass of {result.type}) in the two folders."
            )
    elif isinstance(other, result.type):
        result.append(other)
    else:
        result = NotImplemented
    return result


def _div_core_(result, other):
    """Implement the divide operator as a grouping function."""
    if isinstance(other, string_types + (list, tuple)):
        result.group(other)
        return result
    if isinstance(other, int_types):  # Simple decimate
        for i in range(other):
            result.add_group(f"Group {i}")
        for ix in range(len(result)):
            d = result.__getter__(ix, instantiate=None)
            group = ix % other
            result.groups[f"Group {group}"].__setter__(result.__lookup__(ix), d)
        result.__clear__()
        return result
    return NotImplemented


def _sub_core_(result, other):
    """Implement the core logic of the subtraction operator.

    Note:
        We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!
    """
    calls = [
        (int_types, _sub_core_int_),
        (string_types, _sub_core_string_),
        (metadataObject, _sub_core_data_),
        (baseFolder, _sub_core_folder_),
        (Iterable, _sub_core_iterable_),
    ]
    for typ, func in calls:
        if isinstance(other, typ):
            result = func(result, other)
            break
    else:
        result = NotImplemented

    return result


def _sub_core_int_(result, other):
    """Remove indexed file."""
    delname = result.__names__()[other]
    result.__deleter__(delname)
    return result


def _sub_core_string_(result, other):
    """Remove named file."""
    if other in result.__names__():
        result.__deleter__(other)
    else:
        raise RuntimeError(f"{other} is not in the folder.")
    return result


def _sub_core_data_(result, other):
    """Remove a data object."""
    othername = getattr(other, "filename", getattr(other, "title", None))
    if othername in result.__names__():
        result.__deleter__(othername)
    else:
        raise RuntimeError(f"{othername} is not in the folder.")
    return result


def _sub_core_folder_(result, other):
    """Remove a folder."""
    if isclass(other.type) and issubclass(other.type, result.type):
        for othername in other.ls:
            if othername in result:
                result.__deleter__(othername)
        for othergroup in other.groups:
            if othergroup in result.groups:
                result.groups[othergroup] -= other.groups[othergroup]
    else:
        raise RuntimeError(
            f"Incompatible types ({other.type} must be a subclass of {result.type}) in the two folders."
        )
    return result


def _sub_core_iterable_(result, other):
    """Iterate to remove iterables."""
    for c in sorted(other):
        _sub_core_(result, c)
    return result


def _build_select_function(kargs, arg):
    """Build a select function from an a list of keywords and a keyword name.

    Args:
        kargs (dict):
            The keyword arguments passed to the select function.
        arg (str):
            Name of the keyword argument we're considering.

    Returns:
        tuple of:
            Callable function that takes two arguments and returns a boolean if the two arguments match.
            str name of key to look up
    """
    parts = arg.split("__")
    negate = kargs.pop("negate", False)
    if parts[-1] in operator and len(parts) > 1:
        if len(parts) > 2 and parts[-2] == "not":
            end = -2
            negate = True
        else:
            end = -1
            negate = False
        arg = "__".join(parts[:end])
        op = parts[-1]
    else:
        if isinstance(kargs[arg], tuple) and len(kargs[arg] == 2):
            op = "between"  # Assume two length tuples are testing for range
        elif not isinstance(kargs[arg], string_types) and isiterable(kargs[arg]):
            op = "in"  # Assume other iterables are testing for membership
        else:  # Everything else is exact matches
            op = "eq"
    func = operator[op]
    if negate:
        func = lambda k, v: not func(k, v)
    return func, arg


class baseFolder(MutableSequence):
    """A base class for objectFolders that supports both a sequence of objects and a mapping of instances of itself.

    Attributes:
        groups(GroupsDict):
            A dictionary of similar baseFolder instances
        objects(regexptDict):
            A dictionary of metadataObjects
        _defaults (dict):
            A dictionary of default balues for the constructor of the class when combined with mixin classes
        _no_defaults (list):
            A list of default parameters to veto when setting the constructor.

    Properties:
        depth (int):
            The maximum number of levels of nested groups in the folder
        files (list of str or metadataObject):
            The individual objects or their names if they are not loaded
        instance (metadataObject):
            An empty instance of the data type stored in the folder
        loaded (generator of (str name, metadataObject value):
            Iterate over only the loaded into memory items of the folder
        ls (list of str):
            The names of the objects in the folder, loaded or not
        lsgrp (list of str):
            The names of all the groups in the folder
        mindepth (int):
            Fhe minimum level of nesting groups in the folder.
        not_empty (iterator of metadaaObject):
            Iterates over all members of the folder that have non-zero length
        shape (tuple):
            A data structure that indicates the structure of the objectFolder - tuple of number of files and
            dictionary of the shape of each group.
        type (subclass of metadtaObject):
            The class of objects sotred in this folder

    Notes:
        A baseFolder is a multable sequence object that should store a mapping of instances of some sort of data
        object (typically a :py:class:`Stoner.Core.metadataobject`) which can be iterated over in a reproducible and
        predicatable way as well as being accessed by a key. The other requirement is that it stores a mapping to
        objects of its own type to allow an in-memory tree object to be constructed.

        Additional functionality is built in by providing mixin classes that override the accessors for the data
        object store. Minimally this should include

        - __lookup__ take a keyname and return a canonical accessor key
        - __names__ returns the ordered list of mapping keys to the object store
        - __getter__ returns a single instance of the data object referenced by a canonical key
        - __setter__ add or overwrite an instance of the object store by canonical key
        - __inserter__ insert an instance into a specific place in the Folder
        - __deleter__ remove an instance of a data object by canonical key
        - __clear__ remove all instance
        - __clone__ create a new copy of the mixin's state kinformation
    """

    # pylint: disable=no-member

    _defaults = (
        {}
    )  # A Dictionary of default values that will be combined with other classes to make a global set of defaults
    _no_defaults = []  # A list of dewfaults to remove becayse they clash with subclass methods etc.

    def __new__(cls, *args, **kargs):
        """Create the underlying storage attributes.

        We do this in __new__ so that the mixin classes can access baseFolders state storage before baseFolder does
        further __init__() work.
        """
        self = super(baseFolder, cls).__new__(cls)
        self._debug = kargs.pop("debug", False)
        self._object_attrs = dict()
        self._last_name = 0
        self._groups = GroupsDict(base=self)
        self._objects = regexpDict()
        self._instance = None
        self._object_attrs = dict()
        self._key = None
        self._type = metadataObject
        self._loader = None
        self._instance_attrs = set()
        self._root = "."
        self._default_store = None
        self.directory = None
        return self

    def __init__(self, *args, **kargs):
        """Initialise the baseFolder.

        Notes:
            - Creates empty groups and objects stres
            - Sets all keyword arguments as attributes unless otherwise overwriting an existing attribute
            - stores other arguments in self.args
            - iterates over the multuiple inheritance tree and eplaces any interface methods with ones from
                the mixin classes
            - calls the mixin init methods.
        """
        for k in self.defaults:
            setattr(self, k, kargs.pop(k, self.defaults[k]))

        if len(args) == 1 and isinstance(args[0], baseFolder):  # Special case for type changing.
            self.args = ()
            self.kargs = {}
            self.__init_from_other(args[0])
        else:
            self.args = copy(args)
            self.kargs = copy(kargs)
            # List of routines that define the interface for manipulating the objects stored in the folder
            for k in list(self.kargs.keys()):  # Store keyword parameters as attributes
                if not hasattr(self, k) or k in ["type", "kargs", "args"]:
                    value = kargs.pop(k, None)
                    self.__setattr__(k, value)
                    if self.debug:
                        print(f"Setting self.{k} to {value}")
        self.directory = getattr(self, "directory", None)  # pointless hack for pylint
        super().__init__()

    ###########################################################################
    ################### Properties of baseFolder ##############################

    @property
    def clone(self):
        """Clone just does a deepcopy as a property for compatibility with :py:class:`Stoner.Core.DataFile`."""
        return self.__clone__()

    @property
    def defaults(self):
        """Build a single list of all of our defaults by iterating over the __mro__, caching the result."""
        if getattr(self, "_default_store", None) is None:
            self._default_store = dict()  # pylint: disable=attribute-defined-outside-init
            for cls in reversed(type(self).__mro__):
                if hasattr(cls, "_defaults"):
                    self._default_store.update(cls._defaults)
            for cls in reversed(type(self).__mro__):
                if hasattr(cls, "_no_defaults"):
                    for k in cls._no_defaults:
                        self._default_store.pop(k, None)
        return self._default_store

    @property
    def debug(self):
        """Just read the local debug value."""
        return self._debug

    @debug.setter
    def debug(self, value):
        """Recursely set the debug value."""
        self._debug = value
        self._object_attrs["debug"] = value
        for _, member in self.loaded:
            member.debug = value
        for grp in self.groups:
            self.groups[grp].debug = value

    @property
    def depth(self):
        """Give the maximum number of levels of group below the current objectFolder."""
        if len(self.groups) == 0:
            r = 0
        else:
            r = 1
            for g in self.groups:
                r = max(r, self.groups[g].depth + 1)
        return r

    @property
    def each(self):
        """Return a :py:class:`Stoner.folders.each.item` proxy object.

        This is for calling attributes of the member type of the folder.
        """
        return EachItem(self)

    @property
    def files(self):
        """Return an iterator of potentially unloaded named objects."""
        return [self.__getter__(i, instantiate=None) for i in range(len(self))]

    @files.setter
    def files(self, value):
        """Just a wrapper to clear and then set the objects."""
        if isiterable(value):
            self.__clear__()
            for i, v in enumerate(value):
                self.insert(i, v)

    @property
    def groups(self):
        """Subfolders are held in an ordered dictionary of groups."""
        self._groups.base = self
        return self._groups

    @groups.setter
    def groups(self, value):
        """Ensure groups gets set as a :py:class:`regexpDict`."""
        if not isinstance(value, GroupsDict):
            self._groups = GroupsDict(deepcopy(value), base=self)
        else:
            self._groups = GroupsDict({g: v.clone for g, v in value.items()})
            self._groups.base = self

    @property
    def instance(self):
        """Return a default instance of the type of object in the folder."""
        if self._instance is None:
            self._instance = self._type()
        return self._instance

    @property
    def is_empty(self):
        """Return True if the folder is empty."""
        return len(self) == 0 and len(self.groups) == 0

    @property
    def key(self):
        """Allow overriding for getting and setting the key in mixins."""
        return self._key

    @key.setter
    def key(self, value):
        """Set the folder's key."""
        self._key = value

    @property
    def layout(self):
        """Return a tuple that describes the number of files and groups in the folder."""
        return (len(self), {k: grp.layout for k, grp in self.groups.items()})

    @property
    def loaded(self):
        """Iterate only over those members of the folder in memory."""
        for f in self.__names__():
            val = self.__getter__(f, instantiate=None)
            if isinstance(val, self.type):
                yield f, val

    @property
    def loader(self):
        """Return a callable that will load the files on demand."""
        if self._loader is None:
            self._loader = self.type
        return self._loader

    @loader.setter
    def loader(self, value):
        """Set the loader class ensuring that it is a metadataObject."""
        if isclass(value) and issubclass(value, metadataObject):
            self._loader = value

    @property
    def ls(self):
        """List just the names of the objects in the folder."""
        for f in self.__names__():
            yield f

    @property
    def lsgrp(self):
        """Return a list of the groups as a generator."""
        for k in self.groups:
            yield k

    @property
    def metadata(self):
        """Return a :py:class:`Stoner.folders.metadata.MetadataProxy` object.

        This allows for operations on combined metadata.
        """
        return MetadataProxy(self)

    @property
    def mindepth(self):
        """Give the minimum number of levels of group below the current objectFolder."""
        if len(self.groups) == 0:
            r = 0
        else:
            r = 1e6
            for g in self.groups:
                r = min(r, self.groups[g].depth + 1)
        return r

    @property
    def not_empty(self):
        """Iterate over the objectFolder that checks whether the loaded metadataObject objects have any data.

        Returns the next non-empty DatFile member of the objectFolder.

        Note:
            not_empty will also silently skip over any cases where loading the metadataObject object will raise
            and exception.
        """
        for d in self:
            if len(d) == 0:
                continue
            yield (d)

    @property
    def objects(self):
        """Return the objects in the folder are stored in a :py:class:`regexpDict`."""
        return self._objects

    @objects.setter
    def objects(self, value):
        """Ensure we keep the objects in a :py:class:`regexpDict`."""
        if not isinstance(value, regexpDict):
            self._objects = regexpDict(value)
        else:
            self._objects = value

    @property
    def setas(self):
        """Return the proxy for the setas attribute for each object in the folder."""
        return self.each.setas

    @setas.setter
    def setas(self, value):
        """Set a value to the proxy setas object for each item in the folder."""
        self.each.setas = value

    @property
    def shape(self):
        """Return a data structure that is characteristic of the objectFolder's shape."""
        grp_shape = {k: self[k].shape for k in self.groups}
        return (len(self), grp_shape)

    @property
    def root(self):
        """Return the real folder root."""
        return self._root

    @root.setter
    def root(self, value):
        """Set the folder root."""
        self._root = value

    @property
    def trunkdepth(self):
        """Return the number of levels of group before a group with files is found."""
        if self.files:
            return 0
        return min([self.groups[g].trunkdepth for g in self.groups]) + 1

    @property
    def type(self):
        """Return the (sub)class of the :py:class:`Stoner.Core.metadataObject` instances."""
        return self._type

    @type.setter
    def type(self, value):
        """Ensure that type is a subclass of metadataObject."""
        if isclass(value) and issubclass(value, metadataObject):
            self._type = value
        elif isinstance(value, metadataObject):
            self._type = type(value)
        else:
            raise TypeError(f"{type(value)} os neither a subclass nor instance of metadataObject")
        self._instance = None  # Reset the instance cache

    ################### Methods for subclasses to override to handle storage #####
    def __lookup__(self, name):
        """Stub for other classes to implement.

        Parameters:
            name(str):
                Name of an object

        Returns:
            A key in whatever form the :py:meth:`baseFolder.__getter__` will accept.

        Note:
            We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!
        """
        if isinstance(name, int_types):
            name = self.__names__()[name]
        elif name not in self.__names__():
            name = self._objects.__lookup__(name)
        return name

    def __names__(self):
        """Stub method to return a list of names of all objects that can be indexed for __getter__.

        Note:
            We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!
        """
        return list(self.objects.keys())

    def __getter__(self, name, instantiate=True):
        """Stub method to do whatever is needed to transform a key to a metadataObject.

        Parameters:
            name (key type):
                The canonical mapping key to get the dataObject. By default
                the baseFolder class uses a :py:class:`regexpDict` to store objects in.

        Keyword Arguments:
            instantiate (bool):
                If True (default) then always return a metadataObject. If False, the __getter__ method may return a
                key that can be used by it later to actually get the metadataObject. If None, then will return
                whatever is held in the object cache, either instance
                or name.

        Returns:
            (metadataObject):
                The metadataObject

        Note:
            We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!
        """
        name = self.__lookup__(name)
        if instantiate is None:
            return self.objects[name]
        if not instantiate:
            return name
        name = self.objects[name]
        if not isinstance(name, self._type):
            raise KeyError(f"{name} is not a valid {self._type}")
        return self._update_from_object_attrs(name)

    def __setter__(self, name, value, force_insert=False):
        """Stub to setting routine to store a metadataObject.

        Parameters:
            name (string)
            the named object to write - may be an existing or new name
            value (metadataObject):
                the value to store.

        Keyword Parameters:
            force_insert (bool):
                Ensures the new item is always inserted as a new item and does not replace and existing one.

        Note:
            We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!
        """
        if name is None:
            name = self.make_name()
        if force_insert:
            self.objects.update({name: value})
        else:
            self.objects[name] = value

    def __inserter__(self, ix, name, value):
        """Insert the element into a specific place in our data folder.

        Parameters:
            ix (int):
                the index value to insert at, must be 0 to len(self)-1
            name (str):
                the string name to add as a key
            value (self.type):
                the value to be inserted.

        Note:
            This is written in a way to be generic, but might be better implemented if storage is customised.
        """
        names = list(self.__names__())
        values = [self.__getter__(n, instantiate=None) for n in names]
        names.insert(ix, name)
        values.insert(ix, value)
        self.__clear__()
        for n, v in zip(names, values):
            self.__setter__(n, v)

    def __deleter__(self, ix):
        """Delete an object from the baseFolder.

        Parameters:
            ix(str):
                Index to delete, should be within +- the lengthe length of the folder.

        Note:
            We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!

        """
        del self.objects[ix]

    def __clear__(self):
        """Clear all stored :py:class:`Stoner.Core.metadataObject` instances stored.

        Note:
            We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!

        """
        for n in self.__names__():
            self.__deleter__(self.__lookup__(n))

    def __clone__(self, other=None, attrs_only=False):
        """Do whatever is necessary to copy attributes from self to other.

        Note:
            We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!


        """
        if other is None and not attrs_only:
            return deepcopy(self)
        if other is None:
            other = type(self)()
        for arg in self.defaults:
            if hasattr(self, arg):
                setattr(other, arg, getattr(self, arg))
        other.key = self.key
        other.args = self.args
        other.kargs = self.kargs
        other.type = self.type
        other.debug = self.debug
        for k in self.kargs:
            if not hasattr(other, k):
                setattr(other, k, self.kargs[k])
        for k in self._instance_attrs:
            setattr(other, k, getattr(self, k))
        if not attrs_only:
            for g in self.groups:
                other.groups[g] = self.groups[g].__clone__(other=type(other)(), attrs_only=attrs_only)
            for k in self.__names__():
                other.__setter__(k, self.__getter__(k, instantiate=None))
        return other

    ###########################################################################
    ######## Methods to implement the MutableMapping abstract methods #########
    ######## And to provide a mapping interface that mainly access groups #####

    def __getitem__(self, name):
        """Try to get either a group or an object.

        Parameters:
            name(str, int,slice):
                Which objects to return from the folder.

        Returns:
            Either a baseFolder instance or a metadataObject instance or raises KeyError

        How the indexing works depends on the data type of the parameter *name*:

            - str, regexp
                Then it is checked first against the groups and then against the objects
                dictionaries - both will fall back to a regular expression if necessary.

            - int
                Then the _index attribute is used to find a matching object key.

            - slice
                Then a new :py:class:`baseFolder` is constructed by cloning he current one, but without
                any groups or files. The new :py:class:`baseFolder` is populated with entries
                from the current folder according tot he usual slice definition. This has the advantage
                of not loading the objects in the folder into memory if a :py:class:`DiskBasedFolderMixin` is
                used.
        """
        if name in self.groups and not isinstance(name, int_types):
            return self.groups[name]
        if isinstance(name, string_types + regexp_type):
            if name in self.objects:
                name = self.__lookup__(name)
                return self.__getter__(name)
            name = self.__lookup__(name)
            return self.__getter__(name)
        if isinstance(name, int_types):
            if -len(self) < name < len(self):
                return self.__getter__(self.__lookup__(name), instantiate=True)
            raise IndexError(f"{name} is out of range.")
        if isinstance(name, slice):  # Possibly ought to return another Folder?
            other = self.__clone__(attrs_only=True)
            for iname in islice(self.__names__(), name.start, name.stop, name.step):
                item = self.__getter__(iname)
                if hasattr(item, "filename"):
                    item.filename = iname
                other.append(item)
            return other
        if isinstance(name, tuple):  # recurse indexing through tree with a tuple
            item = self[name[0]]
            if len(name) > 2:
                name = tuple(name[1:])
            elif len(name) == 1:
                return item
            else:
                name = name[1]
            if isinstance(item, self._type):
                return item[name]
            if isinstance(item, type(self)):
                if all_type(name, (int_types, slice)):  # Looks like we're accessing data arrays
                    test = (len(item),) + item[0].data[name].shape
                    output = np.array([]).view(item[0].data.__class__)
                    for data in item:
                        append = data[name]
                        if not isinstance(append, np.ndarray):
                            append = append.asarray()
                        output = np.append(output, append)
                    output = output.reshape(test)
                    return output
                try:
                    return item[name]
                except KeyError:
                    if name in item.metadata.common_keys:
                        return item.metadata.slice(name, output="Data")
                    if self.debug:
                        print(name)
                    raise
            raise KeyError(f"Can't index the baseFolder with {name}")

        raise KeyError(f"Can't index the baseFolder with {name}")

    def __setitem__(self, name, value):
        """Attempt to store a value in either the groups or objects.

        Parameters:
            name(str or int):
                If the name is a string and the value is a baseFolder, then assumes we're accessing
                a group. if name is an integer, then it must be a metadataObject.
        value (baseFolder,metadataObject,str):
            The value to be storred.
        """
        if isinstance(name, string_types):
            if isinstance(value, baseFolder):
                self.groups[name] = value
            else:
                self.__setter__(self.__lookup__(name), value)
        elif isinstance(name, int_types):
            if -len(self) < name < len(self):
                self.__setter__(self.__lookup__(name), value)
            else:
                raise IndexError(f"{name} is out of range")
        else:
            raise KeyError(f"{name} is not a valid key for baseFolder")

    def __delitem__(self, name):
        """Attempt to delete an item from either a group or list of files.

        Parameters:
            name(str,int):
                IF name is a string, then it is checked first against the groups and then
                against the objects. If name is an int then it s checked against the _index.
        """
        if isinstance(name, string_types):
            if name in self.groups:
                del self.groups[name]
            elif name in self.objects:
                self.__deleter__(self.__lookup__(name))
            else:
                raise KeyError(f"Can't use {name} as a key to delete in baseFolder. ({self.__names__()})")
        elif isinstance(name, int_types):
            if -len(self) < name <= len(self):
                self.__deleter__(self.__lookup__(name))
            else:
                raise IndexError(f"{name} is out of range.")
        elif isinstance(name, slice):
            indices = name.indices(len(self))
            name = range(*indices)
            for ix in sorted(name, reverse=True):
                del self[ix]
        else:
            raise KeyError(f"Can't use {name} as a key to delete in baseFolder. ({repr(self.__names__())})")

    def __contains__(self, name):
        """Check whether name is in a list of groups or in the list of names."""
        return name in self.groups or name in self.__names__()

    def __len__(self):
        """Allow len(:py:class:`baseFolder`) works as expected."""
        return len(self.__names__())

    ###########################################################################
    ###################### Standard Special Methods ###########################

    def __add__(self, other):
        """Implement the addition operator for baseFolder and metadataObjects."""
        result = deepcopy(self)
        result = _add_core_(result, other)
        return result

    def __iadd__(self, other):
        """Implement the addition operator for baseFolder and metadataObjects."""
        result = self
        result = _add_core_(result, other)
        return result

    def __truediv__(self, other):
        """Implement the divide operator as a grouping function for a :py:class:`baseFolder`."""
        result = deepcopy(self)
        return _div_core_(result, other)

    def __itruediv__(self, other):
        """Implement the divide operator as an in-place a grouping function for a :py:class:`baseFolder`."""
        result = self
        return _div_core_(result, other)

    def __eq__(self, other):
        """Test whether two objectFolders are the same."""
        if not isinstance(other, baseFolder):
            return False
        if other.shape != self.shape:
            return False
        for mine, theirs in zip(self.groups, other.groups):
            if mine != theirs:
                return False
            if self.groups[mine] != other.groups[theirs]:
                return False
        for mine, theirs in zip(sorted(self.ls), sorted(other.ls)):
            if self[mine] != other[theirs]:
                return False
        return True

    def __invert__(self):
        """For a :py:class:`naseFolder`, inverting means either flattening or unflattening the folder.

        If we have no sub-groups then we assume we are unflattening the Folder and that the object names have
        embedded path separators.
        If we have sub-groups then we assume that we need to flatten the data..
        """
        result = deepcopy(self)
        if len(result.groups) == 0:
            result.unflatten()
        else:
            result.flatten()
        return result

    def __iter__(self):
        """Iterate over objects."""
        return self.__next__()

    def __next__(self):
        """Python 3.x style iterator function."""
        for n in self.__names__():
            member = self.__getter__(n, instantiate=True)
            if member is None:
                continue
            yield member

    def __rmatmul__(self, other):
        """Implement callable@DataFolder as a generic iterate a function over DataFolder members.

        Returns:
            An object that supports __call__ and knows about this DataFolder.
        """
        if not callable(other):
            return NotImplemented
        return self.each.__rmatmul__(other)  # Just bounce it onto the each object

    def __sub__(self, other):
        """Implement the addition operator for baseFolder and metadataObjects."""
        result = deepcopy(self)
        result = _sub_core_(result, other)
        return result

    def __isub__(self, other):
        """Implement the addition operator for baseFolder and metadataObjects."""
        result = self
        result = _sub_core_(result, other)
        return result

    def __deepcopy__(self, memo):
        """Provide support for copy.deepcopy to work."""
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            try:
                setattr(result, k, deepcopy(v, memo))
            except (TypeError, ValueError, RecursionError):
                try:
                    setattr(result, k, copy(v))
                except (TypeError, ValueError, RecursionError):
                    setattr(result, k, v)  # Fallback to just assign the original value if no copy possible
        return result

    def __repr__(self):
        """Print a summary of the objectFolder structure.

        Returns:
            A string representation of the current objectFolder object
        """
        short = get_option("short_folder_rrepr")
        cls = type(self).__name__
        pth = self.key
        pattern = getattr(self, "pattern", "")
        string = f"{cls}({pth}) with pattern {pattern} has {len(self)} files and {len(self.groups)} groups\n"
        if not short:
            for row in self.ls:
                string += "\t" + row + "\n"
        for g in self.groups:  # iterate over groups
            r = self.groups[g].__repr__()
            for line in r.split("\n"):  # indent each line by one tab
                string += "\t" + line + "\n"
        return string.strip()

    def __reversed__(self):
        """Create an iterator function that runs backwards through the stored objects."""
        for n in reversed(self.__names__()):
            member = self.__getter__(n, instantiate=True)
            if member is None:
                continue
            yield member

    def __delattr__(self, name):
        """Handle removing an attribute from the folder, including proxied attributes."""
        if name.startswith("_") or name in [
            "debug",
            "groups",
            "args",
            "kargs",
            "objects",
            "key",
        ]:  # pass ddirectly through for private attributes
            raise AttributeError(f"{name} is a protected attribute and may not be deleted!")
        super().__delattr__(name)

    ###########################################################################
    ###################### Private Methods ####################################

    def __init_from_other(self, other):
        other.__clone__(other=self)

    def _marshall(self, layout=None, data=None):
        """Return the baseFolder as a list of the members including the groups.

        Keyword Arguments:
            layout (tuple):
                number of entries and a dictionary of the groups as generated by :py:property:`baseFolder.layout`
            data (list):
                list of entries to be marshalled if *layout* is defined.

        Returns:
            (list or self):
                If *layout* is defined then returns a copy of the baseFolder with the entries moved around as
                defined in the *layout*. If *layout* is None, then moves the contents into a flat list.
        """
        if layout is None:
            output = []
            for d, name in zip(self, self.__names__()):
                d.filename = name
                output.append(d)
            for grp in self.groups.values():
                output.extend(grp._marshall())
            return output
        count, groups = layout
        if len(data) < count:
            raise ValueError("Insufficient entries in the data argument given the layout supplied.")
        self.extend(data[:count])
        del data[:count]
        for grp in groups:
            self.add_group(grp)
            self.groups[grp]._marshall(layout=groups[grp], data=data)
        return self

    def _update_from_object_attrs(self, obj):
        """Update an object from object_attrs store."""
        for k in self.kargs:  # Set from keyword arguments
            if hasattr(obj, k):
                setattr(obj, k, self.kargs[k])
        if hasattr(self, "_object_attrs") and isinstance(self._object_attrs, dict):
            for k in self._object_attrs:
                try:
                    setattr(obj, k, self._object_attrs[k])
                except AttributeError as err:
                    raise AttributeError(f"Can't set attribute {k} to {self._object_attrs[k]}") from err
        return obj

    def __walk_groups(self, walker, **kargs):
        """Implement the walk_groups method with vreadcrumb trail.

        Args:
            walker (callable):
                A callable object that takes either a metadataObject instance or a objectFolder instance.

        Keyword Arguments:
            group (bool):
                (default False) determines whether the wealker function will expect to be given the objectFolder
                representing the lowest level group or individual metadataObject objects from the lowest level group
            replace_terminal (bool):
                If group is True and the walker function returns an instance of metadataObject then the return value
                is appended to the files and the group is removed from the current objectFolder. This will unwind
                the group hierarchy by one level.
            only_terminal (bool):
                Only iterate over the files in the group if the group has no sub-groups.
            walker_args (dict):
                A dictionary of static arguments for the walker function.
            bbreadcrumb (list of strings):
                A list of the group names or key values that we've walked through

        Notes:
            The walker function should have a prototype of the form: walker(f,list_of_group_names,**walker_args)
            where f is either a objectFolder or metadataObject.
        """
        group = kargs.pop("group", False)
        replace_terminal = kargs.pop("replace_terminal", False)
        only_terminal = kargs.pop("only_terminal", True)
        walker_args = kargs.pop("walker_args", dict())
        breadcrumb = kargs.pop("breadcrumb", dict())
        if len(self.groups) > 0:
            ret = []
            removeGroups = []
            if replace_terminal:
                self.__clear__()
            for g in self.groups:
                bcumb = copy(breadcrumb)
                bcumb.append(g)
                tmp = self.groups[g].__walk_groups(
                    walker, group=group, replace_terminal=replace_terminal, walker_args=walker_args, breadcrumb=bcumb
                )
                if group and replace_terminal and isinstance(tmp, metadataObject):
                    removeGroups.append(g)
                    tmp.filename = f"{g}-{tmp.filename}"
                    self.append(tmp)
                    ret.append(tmp)
            for g in removeGroups:
                del self.groups[g]
        elif len(self.groups) == 0 or not only_terminal:
            if group:
                ret = walker(self, breadcrumb, **walker_args)
            else:
                ret = [walker(f, breadcrumb, **walker_args) for f in self]
        return ret

    ###########################################################################
    ############# Normal Methods ##############################################

    def add_group(self, key):
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
        if key in self.groups:  # do nothing here
            pass
        else:
            new_group = self.__clone__(attrs_only=True)
            self.groups[key] = new_group
            self.groups[key].key = key
            self.groups[key].root = path.join(self.root, str(key))
        return self

    def all(self):
        """Iterate over all the files in the Folder and all it's sub Folders recursely.

        Yields:
            (path/filename,file)
        """
        for g in self.groups.values():
            for p, d in g.all():
                p = path.join(self.key, p)
                yield p, d
        for d in self:
            yield d.filename, d

    def clear(self):
        """Clear the subgroups."""
        self.groups.clear()
        self.__clear__()

    def compress(self, base=None, key=".", keep_terminal=False):
        """Compresses all empty groups from the root up until the first non-empty group is located.

        Returns:
            A copy of the now flattened DatFolder
        """
        return self.groups.compress(base=base, key=key, keep_terminal=keep_terminal)

    def count(self, value):  # pylint:  disable=arguments-differ
        """Provide a count method like a sequence.

        Args:
            value(str, regexp, or :py:class:`Stoner.Core.metadataObject`): The thing to count matches for.

        Returns:
            (int): The number of matching metadataObject instances.

        Notes:
            If *name* is a string, then matching is based on either exact matches of the name, or if it includes a
            * or ? then the basis of a globbing match. *name* may also be a regular expressiuon, in which case
            matches are made on the basis of  the match with the name of the metadataObject. Finally, if *name*
            is a metadataObject, then  it matches for an equyality test.
        """
        if isinstance(value, string_types):
            if "*" in value or "?" in value:  # globbing pattern
                return len(fnmatch.filter(self.__names__(), value))
            return self.__names__().count(self.__lookup__(value))
        if isinstance(value, _pattern_type):
            match = [1 for n in self.__names__() if value.search(n)]
            return len(match)
        if isinstance(value, metadataObject):
            match = [1 for d in self if d == value]
            return len(match)
        raise TypeError(f"Failed to count as value was a {type(value)} which we couldn't use.")

    def fetch(self):
        """Preload the contents of the baseFolder.

        In the base  class this is a NOP because the objects are all in memory anyway.
        """
        return self

    def file(self, name, value, create=True, pathsplit=None):
        """recursely add groups in order to put the named value into a virtual tree of :py:class:`baseFolder`.

        Args:
            name(str):
                A name (which may be a nested path) of the object to file.
            value(metadataObject):
                The object to be filed - it should be an instance of :py:attr:`baseFolder.type`.

        Keyword Aprameters:
            create(bool):
                Whether to create missing groups or to raise an error (default True to create groups).
            pathsplit(str or None):
                Character to use to split the name into path components. Defaults to using os.path.split()

        Returns:
            (baseFolder):
                A reference to the group where the value was eventually filed

        """
        if pathsplit is None:
            pathsplit = r"[\\/]+"
        pathsplit = re.compile(pathsplit)
        pth = pathsplit.split(name)
        tmp = self
        for ix, section in enumerate(pth):
            if ix == len(pth) - 1:
                existing = tmp.__getter__(section, instantiate=None) if section in tmp.__names__() else None
                if (
                    existing is None
                    or (isinstance(value, self.type) and id(existing) != id(value))
                    or (isinstance(existing, string_types) and existing != value)
                ):  # skip if this is a nul op
                    if hasattr(value, "filename"):
                        value.filename = section
                    tmp.__setter__(section, value)
                else:
                    return False  # Return False if we didn't need to move the filing.
                break

            if section not in tmp.groups and create:
                tmp.add_group(section)

            if section in tmp.groups:
                tmp = tmp.groups[section]
            else:
                raise KeyError(f"No group {section} exists and not creating groups.")
        return tmp

    def filter(
        self, filter=None, invert=False, copy=False, recurse=False, prune=True
    ):  # pylint: disable=redefined-builtin
        r"""Filter the current set of files by some criterion.

        Args:
            filter (string or callable):
                Either a string flename pattern or a callable function which takes a single parameter x which is an
                instance of a metadataObject and evaluates True or False

        Keyword Arguments:
            invert (bool):
                Invert the sense of the filter (done by doing an XOR with the filter condition
            copy (bool):
                If set True then the :py:class:`DataFolder` is copied before being filtered. \Default is False -
                work in place.
            recurse (bool):
                If True, apply the filter recursely to all groups. Default False
            prune (bool):
                If True, execute a :py:meth:`baseFolder.prune` to remove empty groups after filering

        Returns:
            The current objectFolder object
        """
        names = []
        if copy:
            result = deepcopy(self)
        else:
            result = self
        if isinstance(filter, string_types):
            for f in result.__names__():
                if fnmatch.fnmatch(f, filter) ^ invert:
                    names.append(result.__getter__(f))
        elif isinstance(filter, _pattern_type):
            for f in result.__names__():
                if (filter.search(f) is not None) ^ invert:
                    names.append(result.__getter__(f))
        elif filter is None:
            raise ValueError("A filter must be defined !")
        else:
            for x in result:
                if filter(x) ^ invert:
                    names.append(x)
        result.__clear__()
        result.extend(names)
        if recurse:
            for g in result.groups.values():
                g.filter(filter=filter, invert=invert, copy=False, recurse=True)
        if prune:
            result.prune()
        return result

    def filterout(self, filter, copy=False, recurse=False, prune=True):  # pylint: disable=redefined-builtin
        """Synonym for self.filter(filter,invert=True).

        Args:
            filter (string or callable):
                Either a string flename pattern or a callable function which takes a single parameter x which is an
                instance of a metadataObject and evaluates True or False

        Keyword Arguments:
            copy (bool):
                If set True then the :py:class:`DataFolder` is copied before being filtered. Default is False -
                work in place.
            recurse (bool):
                If True, apply the filter recursely to all groups. Default False
            prune (bool):
                If True, execute a :py:meth:`baseFolder.prune` to remove empty groups after filering

        Returns:
            The current objectFolder object with the files in the file list filtered.
        """
        return self.filter(filter, invert=True, copy=copy, recurse=recurse, prune=prune)

    def flatten(self, depth=None):
        """Compresses all the groups and sub-groups iunto a single flat file list.

        Keyword Arguments:
            depth )(int or None):
            Only flatten ub-=groups that are within (*depth* of the deepest level.

        Returns:
            A copy of the now flattened DatFolder
        """
        if isinstance(depth, int_types):
            if self.depth <= depth:
                return self.flatten()
            for g in self.groups:
                self.groups[g].flatten(depth)
            return self

        for g in self.groups:
            if self.debug:
                print(f"{self.key}->{self.groups[g].key}")
            self.groups[g].flatten()
            for n in self.groups[g].__names__():
                value = self.groups[g].__getter__(n, instantiate=None)
                old_name = pathjoin(self.groups[g].root, n)
                new_name = path.relpath(old_name, start=self.root)
                if self.debug:
                    print(f"\t{g}::{old_name}=>{new_name}")

                if hasattr(value, "filename"):
                    value.filename = new_name

                if isinstance(
                    value, string_types
                ):  # We haven't loaded this yet, in which case change value to new_name
                    value = new_name
                self.__setter__(new_name, value)
            self.groups[g].__clear__()
        self.groups = {}
        return self

    def get(self, name, default=None):
        """Return either a sub-group or named object from this folder."""
        try:
            ret = self[name]
        except (KeyError, IndexError):
            ret = default
        return ret

    def group(self, key):
        """Sort Files into a series of objectFolders according to the value of the key.

        Args:
            key (string or callable or list):
                Either a simple string or callable function or a list. If a string then it is interpreted as an item
                of metadata in each file. If a callable function then takes a single argument x which should be an
                instance of a metadataObject and returns some vale. If key is a list then the grouping is
                done recursely for each element in key.

        Returns:
            A copy of the current objectFolder object in which the groups attribute is a dictionary of objectFolder
            objects with sub lists of files

        Notes:
            If ne of the grouping metadata keys does not exist in one file then no exception is raised - rather the
            fiiles will be returned into the grou with key None. Metadata keys that are generated from the filename
            are supported.
        """
        if isinstance(key, list):
            next_keys = key[1:]
            key = key[0]
        else:
            next_keys = []
        if isinstance(key, string_types):
            k = key
            key = lambda x: x.get(k, "None")
        for x in self:
            v = key(x)
            if v not in self.groups:
                self.add_group(v)
            self.groups[v].append(x)
        self.__clear__()
        if len(next_keys) > 0:
            for g in self.groups:
                self.groups[g].group(next_keys)
        return self

    def index(self, value, start=None, stop=None):
        """Provide an index method like a sequence.

        Args:
            value(str, regexp, or :py:class:`Stoner.Core.metadataObject`):
                The thing to search for.

        Keyword Arguments:
            start,end(int):
                Limit the index search to a sub-range as per Python 3.5+ list.index

        Returns:
            (int):
                The index of the first matching metadataObject instances.

        Notes:
            If *name* is a string, then matching is based on either exact matches of the name, or if it includes a
            * or ? then the basis of a globbing match. *name* may also be a regular expressiuon, in which case
            matches are made on the basis of  the match with the name of the metadataObject. Finally, if *name*
            is a metadataObject, then it matches for an equyality test.
        """
        if start is None:
            start = 0
        if stop is None:
            stop = len(self)
        search = self.__names__()[start:stop]
        if isinstance(value, string_types):
            if "*" in value or "?" in value:  # globbing pattern
                m = fnmatch.filter(search, value)
                if len(m) > 0:
                    return search.index(m[0]) + start
                raise ValueError(f"{value} is not a name of a metadataObject in this baseFolder.")
            return search.index(self.__lookup__(value)) + start
        if isinstance(value, _pattern_type):
            for i, n in enumerate(search):
                if value.search(n):
                    return i + start
            raise ValueError("No match for any name of a metadataObject in this baseFolder.")
        if isinstance(value, metadataObject):
            for i, n in enumerate(search):
                if value == n:
                    return i + start
            raise ValueError("No match for any name of a metadataObject in this baseFolder.")
        raise TypeError(f"Could not use value of type {type(value)} for index.")

    def insert(self, index, value):  # pylint:  disable=arguments-differ
        """Implement the insert method with the option to append as well."""
        name = self.make_name(value)
        names = self.__names__()
        i = 1
        while name in names:  # Since we're adding a new entry, make sure we have a unique name !
            name, ext = path.splitext(name)
            name = f"{name}({i}).{ext}"
            i += 1
        if -len(self) < index < len(self):
            index = index % len(self)
            self.__inserter__(index, name, value)
            name = self.__names__()[index]
            self.__setter__(self.__lookup__(name), value)
        elif index >= len(self):
            self.__setter__(name, value, force_insert=True)

    def append(self, value):
        """Append an item to the folder object."""
        self.insert(len(self), value)

    def items(self):
        """Return the key,value pairs for the subbroups of this folder."""
        return self.groups.items()

    def keys(self):
        """Return the keys used to access the sub-=groups of this folder."""
        return self.groups.keys()

    def make_name(self, value=None):
        """Construct a name from the value object if possible."""
        if isinstance(value, self.type):
            name = getattr(value, "filename", "")
            if name == "":
                name = f"Untitled-{self._last_name}"
                while name in self:
                    self._last_name += 1
                    name = f"Untitled-{self._last_name}"
            return name
        if isinstance(value, string_types):
            return value
        name = f"Untitled-{self._last_name}"
        while name in self:
            self._last_name += 1
            name = f"Untitled-{self._last_name}"
        return name

    def pop(self, name=-1, default=None):  # pylint: disable=arguments-differ,arguments-renamed
        """Return and remove either a subgroup or named object from this folder."""
        try:
            ret = self[name]
            del self[name]
        except (KeyError, IndexError):
            ret = default
        return ret

    def popitem(self):
        """Return the most recent subgroup from this folder."""
        return self.groups.popitem()

    def prune(self, name=None):
        """Remove any empty groups from the objectFolder (and subgroups).

        Returns:
            A copy of thte pruned objectFolder.
        """
        return self.groups.prune(name=name)

    def select(self, *args, **kargs):
        """Select a subset of the objects in the folder based on flexible search criteria on the metadata.

        Args:
            args (various):
                A single positional argument if present is interpreted as follows:

                *   If a callable function is given, the entire metadataObject is presented to it.
                    If it evaluates True then that metadataObject is selected. This allows arbitrary select operations
                *   If a dict is given, then it and the kargs dictionary are merged and used to select the
                    metadataObjects

        Keyword Arguments:
            recurse (bool):
                Also recursively slect through the sub groups
            kargs (varuous):
                Arbitrary keyword arguments are interpreted as requestion matches against the corresponding
                metadata values. The keyword argument may have an additional **__operator** appended to it which is
                interpreted as follows:

                -   *eq* metadata value equals argument value (this is the default test for scalar argument)
                -   *ne* metadata value doe not equal argument value
                -   *gt* metadata value doe greater than argument value
                -   *lt* metadata value doe less than argument value
                -   *ge* metadata value doe greater than or equal to argument value
                -   *le* metadata value doe less than or equal to argument value
                -   *contains* metadata value contains argument value
                -   *in* metadata value is in the argument value (this is the default test for non-tuple iterable
                                                                arguments)
                -   *startswith* metadata value startswith argument value
                -   *endswith* metadata value endwith argument value
                -   *icontains*,*iin*, *istartswith*,*iendswith* as above but case insensitive
                -   *between* metadata value lies between the minimum and maximum values of the argument
                    (the default test for 2-length tuple arguments)
                -   *ibetween*,*ilbetween*,*iubetween* as above but include both,lower or upper values

            The syntax is inspired by the Django project for selecting, but is not quite as rich.

        Returns:
            (baseFGolder):
                A new baseFolder instance that contains just the matching metadataObjects.

        Note:
            If any of the tests is True, then the metadataObject will be selected, so the effect is a logical OR. To
            achieve a logical AND, you can chain two selects together::

                d.select(temp__le=4.2,vti_temp__lt=4.2).select(field_gt=3.0)

            will select metadata objects that have either temp or vti_temp metadata values below 4.2 AND field
            metadata values greater than 3.

            There are a few cases where special treatment is needed:

            -   If you need to select on a aparameter called *recurse*, pass a dictionary of {"recurse":value} as
                the sole positional argument.
            -   If you need to select on a metadata value that ends in an operator word, then append *__eq* in the
                keyword name to force the equality test.
            -   If the metadata keys to select on are not valid python  identifiers, then pass them via the first
                positional dictionary value.

            If the metadata item being checked exists in a regular expression file pattern for the folder, then
            the files are not loaded and the metadata is evaluated based on the filename. This can speed up operations
            where a file load is not required.
        """
        recurse = kargs.pop("recurse", False)
        negate = kargs.pop("negate", False)
        if len(args) == 1:
            if callable(args[0]):
                kargs["__"] = args[0]
            elif isinstance(args[0], dict):
                kargs.update(args[0])
        result = self.__clone__(attrs_only=True)
        if recurse:
            gkargs = {}
            gkargs.update(kargs)
            gkargs["negate"] = negate
            gkargs["recurse"] = True
            for g in self.groups:
                result.groups[g] = self.groups[g].select(*args, **gkargs)
        if isinstance(self.pattern[0], regexp_type):
            pattern_keys = list(self.pattern[0].groupindex.keys())
            for karg in kargs:
                if karg.split("__")[0] not in pattern_keys:
                    must_read = True
                    break
            else:
                must_read = False
        else:
            must_read = True

        for f in self.objects:
            if must_read and isinstance(f, string_types):
                f = self.__getter__(f, instantiate=True)
            placer = f
            if not must_read:
                match = self.pattern[0].search(f)
                f = TypeHintedDict(match.groupdict())

            for arg in kargs:
                if callable(kargs[arg]) and kargs[arg](f):
                    break
                elif isinstance(arg, string_types):
                    val = kargs[arg]
                    skargs = copy(kargs)
                    skargs["negate"] = negate
                    func, key = _build_select_function(skargs, arg)
                    if key in f and func(f[key], val):
                        break
            else:  # No tests matched - contineu to next line
                continue
            # Something matched, so append to result
            f = placer
            if hasattr(f, "filename"):
                name = f.filename
                result.__setter__(name, f)
            else:
                result.append(f)
        return result

    def setdefault(self, k, d=None):
        """Return or set a subgroup or named object."""
        self[k] = self.get(k, d)
        return self[k]

    def slice_metadata(self, key, output="smart"):
        """Return an array of the metadata values for each item/file in the top level group.

        Args:
            key(str, regexp or list of str): the meta data key(s) to return

        Keyword Parameters:
            output (str):
                Output format - values are
                -   dict: return an array of dictionaries
                -   list: return a list of lists
                -   array: return a numpy array
                -   Data: return a :py:class:`Stoner.Data` object
                -   smart: (default) return either a list if only one key or a list of dictionaries

        Returns:
            (array of metadata):
                If single key is given and is an exact match then returns an array of the matching values.
                If the key results in a regular expression match, then returns an array of dictionaries of all
                matching keys. If key is a list ir other iterable, then return a 2D array where each column
                corresponds to one of the keys.

        Todo:
            Add options to recurse through all groups? Put back RCT's values only functionality?
        """
        return self.metadata.slice(key, output=output)

    def sort(self, key=None, reverse=False, recurse=True):
        """Sort the files by some key.

        Keyword Arguments:
            key (string, callable or None):
                Either a string or a callable function. If a string then this is interpreted as a
                metadata key, if callable then it is assumed that this is a a function of one parameter x
                that is a :py:class:`Stoner.Core.metadataObject` object and that returns a key value.
                If key is not specified (default), then a sort is performed on the filename
            reverse (bool):
                Optionally sort in reverse order
            recurse (bool):
                If True (default) sort the sub-groups as well.

        Returns:
            A copy of the current objectFolder object
        """
        if recurse:
            for grp in self.groups.values():
                grp.sort(key=key, reverse=reverse, recurse=recurse)
        tmp = self.clone
        if isinstance(key, string_types):
            k = [(x.get(key), i) for i, x in enumerate(tmp)]
            k = sorted(k, reverse=reverse)
            new_order = [tmp[i] for x, i in k]
            new_names = [self.__names__()[i] for x, i in k]
        elif key is None:
            fnames = tmp.__names__()
            fnames.sort(reverse=reverse)
            new_order = [tmp.__getter__(name) for name in fnames]
            new_names = fnames
        elif isinstance(key, _pattern_type):
            new_names = sorted(tmp.__names__(), key=lambda x: key.match(x).groups(), reverse=reverse)
            new_order = [tmp.__getter__(x) for x in new_names]
        else:
            order = range(len(tmp))
            new_order = sorted(order, key=lambda x: key(self[x]), reverse=reverse)
            new_order = [tmp.__names__()[i] for i in new_order]
            new_names = new_order
        self.__clear__()
        for obj, k in zip(new_order, new_names):
            self.__setter__(k, obj)

        return self

    def unflatten(self):
        """Take the file list an unflattens them according to the file paths.

        Returns:
            A copy of the objectFolder
        """
        if len(self):
            if len(self) == 1:
                self.directory = path.join(self.directory, path.dirname(self.__names__()[0]))
            else:
                self.directory = commonpath([path.realpath(path.join(self.directory, x)) for x in self.__names__()])
            names = self.__names__()
            relpaths = [path.relpath(path.join(self.directory, f), self.directory) for f in names]
            dels = list()
            for i, f in enumerate(relpaths):
                ret = self.file(f, self.__getter__(names[i], instantiate=None))
                if isinstance(ret, baseFolder):  # filed ok
                    dels.append(i)
            for i in sorted(dels, reverse=True):
                del self[i]
        for g in self.groups:
            self.groups[g].unflatten()
        return self

    def update(self, other):
        """Update this folder with a dictionary or another folder."""
        if isinstance(other, dict):
            for k in other:
                self[k] = other[k]
        elif isinstance(other, baseFolder):
            for k in other.groups:
                if k in self.groups:
                    self.groups[k].update(other.groups[k])
                else:
                    self.groups[k] = other.groups[k].clone
            for k in other.__names__():
                if k in self.__names__():
                    self.__setter__(self.__lookup__(k), other.__getter__(other.__lookup__(k)).clone)
                else:
                    self.append(other.__getter__(other.__lookup__(k)).clone)

    def values(self):
        """Return the sub-groups of this folder."""
        return self.groups.values()

    def walk_groups(self, walker, **kargs):
        """Walk through a hierarchy of groups and calls walker for each file.

        Args:
            walker (callable):
                A callable object that takes either a metadataObject instance or a objectFolder instance.

        Keyword Arguments:
            group (bool):
                (default False) determines whether the walker function will expect to be given the objectFolder
                representing the lowest level group or individual metadataObject objects from the lowest level group
            replace_terminal (bool):
                If group is True and the walker function returns an instance of metadataObject then the return value
                is appended to the files and the group is removed from the current objectFolder. This will unwind
                the group hierarchy by one level.
            obly_terminal(bool):
                Only execute the walker function on groups that have no sub-groups inside them (i.e. are terminal
                groups)
            walker_args (dict):
                A dictionary of static arguments for the walker function.

        Notes:
            The walker function should have a prototype of the form::

                walker(f,list_of_group_names,**walker_args)

            where f is either a objectFolder or metadataObject.
        """
        group = kargs.pop("group", False)
        replace_terminal = kargs.pop("replace_terminal", False)
        only_terminal = kargs.pop("only_terminal", True)
        walker_args = kargs.pop("walker_args", dict())
        walker_args = dict() if walker_args is None else walker_args
        return self.__walk_groups(
            walker,
            group=group,
            replace_terminal=replace_terminal,
            only_terminal=only_terminal,
            walker_args=walker_args,
            breadcrumb=[],
        )

    def zip_groups(self, groups):
        """Return a list of tuples of metadataObjects drawn from the specified groups.

        Args:
            groups(list of strings):
                A list of keys of groups in the Lpy:class:`objectFolder`

        Returns:
            A list of tuples of groups of files:
                [(grp_1_file_1,grp_2_file_1....grp_n_files_1),(grp_1_file_2,
                grp_2_file_2....grp_n_file_2)....(grp_1_file_m,grp_2_file_m...grp_n_file_m)]
        """
        if not isinstance(groups, list):
            raise SyntaxError("groups must be a list of groups")
        grps = [[y for y in self.groups[x]] for x in groups]
        return zip(*grps)
