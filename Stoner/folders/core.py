# -*- coding: utf-8 -*-
"""Provide the base classes and functions for the :py:class:`Stoner.DataFolder` class."""

__all__ = ["BaseFolder"]

from collections.abc import Iterable, MutableSequence
from copy import copy, deepcopy
from inspect import isclass
from itertools import islice
from os import path

import numpy as np

from ..compat import _pattern_type, int_types, string_types
from ..core.base import RegexpDict, metadataObject
from ..tools import all_type, get_option, isiterable
from ..tools.decorators import class_modifier
from . import functions, methods
from .each import Item as EachItem
from .groups import GroupsDict
from .metadata import MetadataProxy

regexp_type = (_pattern_type,)


def _add_core_(result, other):
    """Implement the core logic of the addition operator.

    Note:
        We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!
    """
    resultype = result.type
    match other:
        case BaseFolder() if isclass(other.type) and issubclass(other.type, result.type):
            result.extend(list(other.files))
            for grp in other.groups:
                if grp in result.groups:
                    result.groups[grp] += other.groups[grp]  # recursely merge groups
                    return result
                result.groups[grp] = copy(other.groups[grp])
            return result
        case BaseFolder():
            raise RuntimeError(
                f"Incompatible types ({other.type} must be a subclass of {result.type}) in the two folders."
            )
        case resultype():
            result.append(other)
            return result
        case _:
            return NotImplemented


def _div_core_(result, other):
    """Implement the divide operator as a grouping function."""
    match other:
        case str() | list() | tuple():
            result.group(other)
            return result
        case int():
            for i in range(other):
                result.add_group(f"Group {i}")
            for ix in range(len(result)):
                d = result.__getter__(ix, instantiate=None)
                group = ix % other
                result.groups[f"Group {group}"].__setter__(result.__lookup__(ix), d)
            result.__clear__()
            return result
        case _:
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
        (BaseFolder, _sub_core_folder_),
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
        return result
    raise RuntimeError(f"{other} is not in the folder.")


def _sub_core_data_(result, other):
    """Remove a data object."""
    othername = getattr(other, "filename", getattr(other, "title", None))
    if othername in result.__names__():
        result.__deleter__(othername)
        return result
    raise RuntimeError(f"{othername} is not in the folder.")


def _sub_core_folder_(result, other):
    """Remove a folder."""
    if isclass(other.type) and issubclass(other.type, result.type):
        for othername in other.ls:
            if othername in result:
                result.__deleter__(othername)
        for othergroup in other.groups:
            if othergroup in result.groups:
                result.groups[othergroup] -= other.groups[othergroup]
        return result
    raise RuntimeError(f"Incompatible types ({other.type} must be a subclass of {result.type}) in the two folders.")


def _sub_core_iterable_(result, other):
    """Iterate to remove iterables."""
    for c in sorted(other):
        _sub_core_(result, c)
    return result


@class_modifier([functions, methods], adaptor=None, no_long_names=True, overload=True)
class BaseFolder(MutableSequence):
    """A base class for objectFolders that supports both a sequence of objects and a mapping of instances of itself.

    Attributes:
        groups(GroupsDict):
            A dictionary of similar BaseFolder instances
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
            The class of objects stored in this folder

    Notes:
        A BaseFolder is a multable sequence object that should store a mapping of instances of some sort of data
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

    def __new__(cls, *args, **kwargs):
        """Create the underlying storage attributes.

        We do this in __new__ so that the mixin classes can access BaseFolders state storage before BaseFolder does
        further __init__() work.
        """
        self = super(BaseFolder, cls).__new__(cls)
        self._debug = kwargs.pop("debug", False)
        self._object_attrs = {}
        self._last_name = 0
        self._groups = GroupsDict(base=self)
        self._objects = RegexpDict()
        self._instance = None
        self._object_attrs = {}
        self._key = None
        self._type = metadataObject
        self._loader = None
        self._instance_attrs = set()
        self._root = "."
        self._default_store = None
        self.directory = None
        self.executor = None
        return self

    def __init__(self, *args, **kwargs):
        """Initialise the BaseFolder.

        Notes:
            - Creates empty groups and objects stres
            - Sets all keyword arguments as attributes unless otherwise overwriting an existing attribute
            - stores other arguments in self.args
            - iterates over the multuiple inheritance tree and eplaces any interface methods with ones from
                the mixin classes
            - calls the mixin init methods.
        """
        for k in self.defaults:
            setattr(self, k, kwargs.pop(k, self.defaults[k]))

        if len(args) == 1 and isinstance(args[0], BaseFolder):  # Special case for type changing.
            self.args = ()
            self.kwargs = {}
            self.__init_from_other(args[0])
        else:
            self.args = copy(args)
            self.kwargs = copy(kwargs)
            # List of routines that define the interface for manipulating the objects stored in the folder
            for k in list(self.kwargs.keys()):  # Store keyword parameters as attributes
                if not hasattr(self, k) or k in ["type", "kwargs", "args"]:
                    value = kwargs.pop(k, None)
                    self.__setattr__(k, value)
                    if self.debug:
                        print(f"Setting self.{k} to {value}")
        self.directory = getattr(self, "directory", None)  # pointless hack for pylint
        super().__init__()

    ###########################################################################
    ################### Properties of BaseFolder ##############################

    @property
    def clone(self):
        """Clone just does a deepcopy as a property for compatibility with :py:class:`Stoner.Core.DataFile`."""
        return self.__clone__()

    @property
    def defaults(self):
        """Build a single list of all of our defaults by iterating over the __mro__, caching the result."""
        if getattr(self, "_default_store", None) is None:
            self._default_store = {}  # pylint: disable=attribute-defined-outside-init
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
        for val in self.groups.values():
            val.debug = value

    @property
    def depth(self):
        """Give the maximum number of levels of group below the current objectFolder."""
        if len(self.groups) == 0:
            return 0
        r = 1
        for val in self.groups.values():
            r = max(r, val.depth + 1)
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
        """Ensure groups gets set as a :py:class:`RegexpDict`."""
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
        yield from self.__names__()

    @property
    def lsgrp(self):
        """Return a list of the groups as a generator."""
        yield from self.groups

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
            return 0
        r = 1e6
        for val in self.groups.values():
            r = min(r, val.depth + 1)
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
            yield d

    @property
    def objects(self):
        """Return the objects in the folder are stored in a :py:class:`RegexpDict`."""
        return self._objects

    @objects.setter
    def objects(self, value):
        """Ensure we keep the objects in a :py:class:`RegexpDict`."""
        if not isinstance(value, RegexpDict):
            self._objects = RegexpDict(value)
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
        return min((grp.trunkdepth for grp in self.groups.values())) + 1

    @property
    def type(self):
        """Return the (sub)class of the :py:class:`Stoner.Core.metadataObject` instances."""
        return self._type

    @type.setter
    def type(self, value):
        """Ensure that type is a subclass of metadataObject."""
        match value:
            case type() if issubclass(value, metadataObject):
                self._type = value
            case metadataObject():
                self._type = type(value)
            case _:
                raise TypeError(f"{type(value)} os neither a subclass nor instance of metadataObject")
        self._instance = None  # Reset the instance cache

    ################### Methods for subclasses to override to handle storage #####
    def __lookup__(self, name):
        """Stub for other classes to implement.

        Parameters:
            name(str):
                Name of an object

        Returns:
            A key in whatever form the :py:meth:`BaseFolder.__getter__` will accept.

        Note:
            We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!
        """
        if isinstance(name, int_types):
            return self.__names__()[name]
        if name not in self.__names__():
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

        Args:
            name (key type):
                The canonical mapping key to get the dataObject. By default
                the BaseFolder class uses a :py:class:`RegexpDict` to store objects in.

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
        """Delete an object from the BaseFolder.

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
        other.kwargs = self.kwargs
        other.type = self.type
        other.debug = self.debug
        for k in self.kwargs:
            if not hasattr(other, k):
                setattr(other, k, self.kwargs[k])
        for k in self._instance_attrs:
            setattr(other, k, getattr(self, k))
        if not attrs_only:
            for g, val in self.groups.items():
                other.groups[g] = val.__clone__(other=type(other)(), attrs_only=attrs_only)
            for k in self.__names__():
                other.__setter__(k, self.__getter__(k, instantiate=None))
        return other

    ###########################################################################
    ######## Methods to implement the MutableMapping abstract methods #########
    ######## And to provide a mapping interface that mainly access groups #####

    def __del__(self):
        """Clean up the exececutor if it is defined."""
        if self.executor:
            self.executor.shutdown()

    def __getitem__(self, name):
        """Try to get either a group or an object.

        Parameters:
            name(str, int,slice):
                Which objects to return from the folder.

        Returns:
            Either a BaseFolder instance or a metadataObject instance or raises KeyError

        How the indexing works depends on the data type of the parameter *name*:

            - str, regexp
                Then it is checked first against the groups and then against the objects
                dictionaries - both will fall back to a regular expression if necessary.

            - int
                Then the _index attribute is used to find a matching object key.

            - slice
                Then a new :py:class:`BaseFolder` is constructed by cloning he current one, but without
                any groups or files. The new :py:class:`BaseFolder` is populated with entries
                from the current folder according tot he usual slice definition. This has the advantage
                of not loading the objects in the folder into memory if a :py:class:`DiskBasedFolderMixin` is
                used.
        """
        if name in self.groups and not isinstance(name, int_types):
            return self.groups[name]
        match name:
            case str():
                name = self.__lookup__(name)
                return self.__getter__(name)
            case _ if isinstance(name, regexp_type):
                name = self.__lookup__(name)
                return self.__getter__(name)
            case int():
                if -len(self) < name < len(self):
                    return self.__getter__(self.__lookup__(name), instantiate=True)
                raise IndexError(f"{name} is out of range.")
            case slice():
                other = self.__clone__(attrs_only=True)
                for iname in islice(self.__names__(), name.start, name.stop, name.step):
                    item = self.__getter__(iname)
                    if hasattr(item, "filename"):
                        item.filename = iname
                    other.append(item)
                return other
            case tuple():
                return self._recursive_getitem(name)
            case _:
                raise KeyError(f"Can't index the BaseFolder with {name}")

    def __setitem__(self, name, value):
        """Attempt to store a value in either the groups or objects.

        Parameters:
            name(str or int):
                If the name is a string and the value is a BaseFolder, then assumes we're accessing
                a group. if name is an integer, then it must be a metadataObject.
        value (BaseFolder,metadataObject,str):
            The value to be storred.
        """
        if isinstance(name, string_types):
            if isinstance(value, BaseFolder):
                self.groups[name] = value
            else:
                self.__setter__(self.__lookup__(name), value)
        elif isinstance(name, int_types):
            if -len(self) < name < len(self):
                self.__setter__(self.__lookup__(name), value)
            else:
                raise IndexError(f"{name} is out of range")
        else:
            raise KeyError(f"{name} is not a valid key for BaseFolder")

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
                raise KeyError(f"Can't use {name} as a key to delete in BaseFolder. ({self.__names__()})")
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
            raise KeyError(f"Can't use {name} as a key to delete in BaseFolder. ({repr(self.__names__())})")

    def __contains__(self, name):
        """Check whether name is in a list of groups or in the list of names."""
        return name in self.groups or name in self.__names__()

    def __len__(self):
        """Allow len(:py:class:`BaseFolder`) works as expected."""
        return len(self.__names__())

    ###########################################################################
    ###################### Standard Special Methods ###########################

    def __add__(self, other):
        """Implement the addition operator for BaseFolder and metadataObjects."""
        result = deepcopy(self)
        result = _add_core_(result, other)
        return result

    def __iadd__(self, other):
        """Implement the addition operator for BaseFolder and metadataObjects."""
        result = self
        result = _add_core_(result, other)
        return result

    def __truediv__(self, other):
        """Implement the divide operator as a grouping function for a :py:class:`BaseFolder`."""
        result = deepcopy(self)
        return _div_core_(result, other)

    def __itruediv__(self, other):
        """Implement the divide operator as an in-place a grouping function for a :py:class:`BaseFolder`."""
        result = self
        return _div_core_(result, other)

    def __eq__(self, other):
        """Test whether two objectFolders are the same."""
        if not isinstance(other, BaseFolder):
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
        """Implement the addition operator for BaseFolder and metadataObjects."""
        result = deepcopy(self)
        result = _sub_core_(result, other)
        return result

    def __isub__(self, other):
        """Implement the addition operator for BaseFolder and metadataObjects."""
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
        for val in self.groups.values():  # iterate over groups
            r = val.__repr__()
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
            "kwargs",
            "objects",
            "key",
        ]:  # pass ddirectly through for private attributes
            raise AttributeError(f"{name} is a protected attribute and may not be deleted!")
        super().__delattr__(name)

    ###########################################################################
    ###################### Private Methods ####################################

    def _recursive_getitem(self, name):
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
        raise KeyError(f"Can't index the BaseFolder with {name}")

    def __init_from_other(self, other):
        other.__clone__(other=self)

    def _marshall(self, layout=None, data=None):
        """Return the BaseFolder as a list of the members including the groups.

        Keyword Arguments:
            layout (tuple):
                number of entries and a dictionary of the groups as generated by :py:property:`BaseFolder.layout`
            data (list):
                list of entries to be marshalled if *layout* is defined.

        Returns:
            (list or self):
                If *layout* is defined then returns a copy of the BaseFolder with the entries moved around as
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
        for k in self.kwargs:  # Set from keyword arguments
            if hasattr(obj, k):
                setattr(obj, k, self.kwargs[k])
        if hasattr(self, "_object_attrs") and isinstance(self._object_attrs, dict):
            for k, val in self._object_attrs.items():
                try:
                    setattr(obj, k, val)
                except AttributeError as err:
                    raise AttributeError(f"Can't set attribute {k} to {val}") from err
        return obj

    def insert(self, index, value):  # pylint:  disable=arguments-differ
        """Implement the insert method with the option to append as well.

        Args:
            self (BaseFolder):
                DataFolder instance when not a bound method.
            index (int):
                Index to insert at
            value (metadataObject):
                Metadata object to be added.
        """
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

    def _walk_groups(
        self, walker, group=False, replace_terminal=False, only_terminal=True, walker_args=None, breadcrumb=None
    ):
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
            breadcrumb (list of strings):
                A list of the group names or key values that we've walked through

        Notes:
            The walker function should have a prototype of the form: walker(f,list_of_group_names,**walker_args)
            where f is either a objectFolder or metadataObject.
        """
        walker_args = walker_args or {}
        breadcrumb = breadcrumb or []
        if len(self.groups) > 0:
            ret = []
            removeGroups = []
            if replace_terminal:
                self.__clear__()
            for g, val in self.groups.items():
                bcumb = copy(breadcrumb)
                bcumb.append(g)
                tmp = val._walk_groups(
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
