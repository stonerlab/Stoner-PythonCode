# -*- coding: utf-8 -*-
"""Classes and support functions for the :py:attr:`Stoner.DataFolder.each`.magic attribute."""
__all__ = ["item"]
import numpy as np
from collections.abc import MutableSequence
from functools import wraps, partial
from traceback import format_exc
from .utils import get_pool
from Stoner.tools import isIterable
from Stoner.compat import string_types


def _worker(d, **kwargs):
    """Support function to run an arbitary function over a :py:class:`Stoner.Data` object."""
    byname = kwargs.get("byname", False)
    func = kwargs.get("func", lambda x: x)
    if byname:
        func = getattr(d, func, lambda x: x)
    args = kwargs.get("args", tuple())
    kargs = kwargs.get("kargs", dict)
    if hasattr(d, "setas"):
        d["setas"] = list(d.setas)
    d["args"] = args
    d["kargs"] = kargs
    d["func"] = func.__name__
    try:
        if byname:  # Ut's an instance bound moethod
            ret = func(*args, **kargs)
        else:  # It's an arbitary function
            ret = func(d, *args, **kargs)
    except Exception as e:
        ret = e, format_exc()
    return (d, ret)


class setas_wrapper(MutableSequence):

    """This class manages wrapping each member of the folder's setas attribute."""

    def __init__(self, parent):
        """Note a reference to the parent item class instance and folder."""
        self._each = parent
        self._folder = parent._folder

    def __call__(self, *args, **kargs):
        """Simple pass through the calls the setas method of each item in our folder."""
        for obj in self._folder:
            obj.setas(*args, **kargs)
        return self._folder

    def __len__(self):
        """Return the lengths of all the setas elements in the folder (or a sclar if all the same)"""
        lengths = np.array([len(data.setas) for data in self._folder])
        if len(np.unique(lengths)) == 1:
            return lengths[0]
        return lengths.tolist()

    def __getitem__(self, index):
        """Get the corresponding item from all the setas items in the folder."""
        return [data.setas[index] for data in self._folder]

    def __setitem__(self, index, value):
        """Set the value of the specified item on the setas elements in the folder.

        Args:
            index (int):
                The column index of the individual setas attributes to change
            value (iterable):
                The set of values to apply to the setas attributes.

        Notes:
            If value has a length less than the data folder, then the final value is repeated to encompass the
            remaining elements.
        """
        if len(value) < len(self._folder):
            value = value + value[-1] * (len(self._folder) - len(value))
        for v, data in zip(value, self._folder):
            data.setas[index] = v
        setas = self._folder._object_attrs.get("setas", ["."] * len(self._folder))
        if len(setas) <= index:
            setas.extend(["."] * (index + 1 - len(setas)))
        setas[index] = v
        self._folder._object_attrs["setas"] = setas

    def __delitem__(self, index):
        """Cannot delete items from the proxy setas object - so simply clear it instead."""
        for data in self._folder:
            data.setas[index] = "."
        setas = self._folder._object_attrs.get("setas", ["." * len(self._folder)])
        setas[index] = "."
        self._folder._object_attrs["setas"] = setas

    def insert(self, index, value):
        """Cannot insert items into the proxy setas object."""
        raise IndexError("Cannot insert into the objectFolder's setas - insdert into the objectFolder instead!")

    def collapse(self):
        """Collapse the setas into a single list if possible."""
        setas = []
        for v in self:
            if len(v):
                if np.unique(v).size == 1:
                    setas.append(v[0])
                else:
                    setas.append("-")
        return setas


class item:

    """Provides a proxy object for accessing methods on the inividual members of a Folder.

    Notes:
        The pupose of this class is to allow it to be explicit that we're calling methods
        on the members of the folder rather than a collective method. This allows us to work
        around nameclashes.
    """

    _folder = None

    def __init__(self, folder):

        self._folder = folder

    @property
    def setas(self):
        """Return a proxy object for manipulating all the setas objects in a folder."""
        return setas_wrapper(self)

    @setas.setter
    def setas(self, value):
        """Manipualte the setas property of all the objects in the folder."""
        setas = self.setas
        setas(value)
        self._folder._object_attrs["setas"] = setas.collapse()
        return setas

    def __call__(self, func, *args, **kargs):
        """Iterate over the baseFolder, calling func on each item.

        Args:
            func (callable, str):
                Either a callable object, or the name of a callable object (either method or global) that must take
                a metadataObject type instance as it's first argument.

        Keyword Args:
            _return (None, bool or str): Controls how the return value from *func* is added to the DataFolder

        Returns:
            A list of the results of evaluating *func* for each item in the folder.

        Notes:
            If *_return* is None and the return type of *func* is the same type as the :py:class:`baseFolder` is
            storing, then the return value replaces trhe original :py:class:`Stoner.Core.metadataobject` in the
            :py:class:`baseFolder`. If *_result* is True the return value is added to the
            :py:class:`Stoner.Core.metadataObject`'s metadata under the name of the function. If *_result* is a
            string. then return result is stored in the corresponding name.
        """
        # Just call the iter generator but assemble into a list.
        if isinstance(func, string_types) and "_byname" not in kargs:
            if func in globals() and callable(globals()[func]):
                func = globals()[func]
            else:
                func = getattr(self, func)
        return list(self.iter(func, *args, **kargs))

    def __dir__(self):
        """Return a list of the common set of attributes of the instances in the folder."""
        if self._folder and len(self._folder) != 0:
            res = set(dir(self._folder[0]))
        else:
            res = set()
        if self._folder and len(self._folder) > 0:
            for d in self._folder[1:]:
                res &= set(dir(d))
        return list(res)

    def __getattr__(self, name):
        """Handles some special case attributes that provide alternative views of the objectFolder.

        Args:
            item (string): The attribute name being requested

        Returns:
            Depends on the attribute

        Notes:
            If *name* is not present on the empty member instance, then the first member of the folder is checked as
            well. This allows the attributes of a :py:class:`Stoner.Data` object that derive from the
            Lpy:attr:`Stoner.Data.setas` attribute (such as *.x*, *.y* or *.e* etc) can be accessed.
        """
        try:
            return super(item, self).__getattr__(name)
        except AttributeError:
            pass
        try:
            instance = self._folder.instance
            if callable(getattr(instance, name, None)):  # It's a method
                ret = self.__getattr_proxy(name)
            else:  # It's a static attribute
                if name in self._folder._object_attrs:
                    ret = self._folder._object_attrs[name]
                elif len(self._folder):
                    ret = [(not hasattr(x, name), getattr(x, name, None)) for x in self._folder]
                    mask, values = zip(*ret)
                    ret = np.ma.MaskedArray(values)
                    ret.mask = mask
                else:
                    ret = getattr(instance, name, None)
                if ret is None:
                    raise AttributeError
        except AttributeError:  # Ok, pass back
            raise AttributeError("{} is not an Attribute of {} or {}".format(name, type(self), type(instance)))
        except TypeError as err:  # Can be triggered if self.instance lacks the attribute
            if len(self._folder) and hasattr(self._folder[0], name):
                ret = [(not hasattr(x, name), getattr(x, name, None)) for x in self._folder]
                mask, values = zip(*ret)
                ret = np.ma.MaskedArray(values)
                ret.mask = mask
            else:
                raise err

        return ret

    def __setattr__(self, name, value):
        """Setting the attrbute on .each sets it on all instantiated objects and in _object_attrs.

        Args:
            name(str): Attribute to set
            value (any): Value to set

        Notes:
            If *name* is not present on the empty member instance, then the first member of the folder is checked as
            well. This allows the attributes of a :py:class:`Stoner.Data` object that derive from the
            Lpy:attr:`Stoner.Data.setas` attribute (such as *.x*, *.y* or *.e* etc) can be accessed.

            If *value* is iterable and the same length as the folder, then each element in the folder is loaded and
            the corresponding element of *value* is assigned to the attribute of the member.
        """
        if hasattr(self.__class__, name) or name.startswith("_"):  # Handle setting our own attributes
            super(item, self).__setattr__(name, value)
        elif name in dir(self._folder.instance) or (
            len(self._folder) and hasattr(self._folder[0], name)
        ):  # This is an instance attribute
            if isIterable(value) and len(value) == len(self._folder):
                force_load = True
            else:
                force_load = False
                self._folder._object_attrs[name] = value  # Add to attributes to be set on load
                value = [value] * len(self._folder)
            for d, v in zip(self._folder.__names__(), value):  # And set on all instantiated objects
                if force_load or isinstance(self._folder.__getter__(d, instantiate=False), self._folder.type):
                    d = self._folder.__getter__(d)
                    setattr(d, name, v)
        else:
            raise AttributeError("Unknown attribute {}".format(name))

    def __getattr_proxy(self, item):
        """Make a prpoxy call to access a method of the metadataObject like types.

        Args:
            item (string): Name of method of metadataObject class to be called

        Returns:
            Either a modifed copy of this objectFolder or a list of return values
            from evaluating the method for each file in the Folder.
        """
        meth = getattr(self._folder.instance, item, None)

        @wraps(meth)
        def _wrapper_(*args, **kargs):
            """Wraps a call to the metadataObject type for magic method calling.

            Keyword Arguments:
                _return (index types or None): specify to store the return value in the individual object's metadata

            Note:
                This relies on being defined inside the enclosure of the objectFolder method
                so we have access to self and item
            """
            kargs["_byname"] = True
            return self(item, *args, **kargs)  # Develove to self.__call__ where we have multiprocess magic

        # Ok that's the wrapper function, now return  it for the user to mess around with.
        return _wrapper_

    def __rmatmul__(self, other):
        """Implement callable@DataFolder as a generic iterate a function over DataFolder members.

        Returns:
            An object that supports __call__ and knows about this DataFolder.
        """
        if not callable(other):
            return NotImplemented

        @wraps(other)
        def _wrapper_(*args, **kargs):
            """Wraps a call to the metadataObject type for magic method calling.

            Keyword Arguments:
                _return (index types or None): specify to store the return value in the individual object's metadata

            Note:
                This relies on being defined inside the enclosure of the objectFolder method
                so we have access to self and item
            """
            kargs["_byname"] = False  # Force the __call__ to use the callable function
            return self(other, *args, **kargs)  # Delegate to self.__call__ which has multiprocess magic.

        # Ok that's the wrapper function, now return  it for the user to mess around with.
        return _wrapper_

    def iter(self, func, *args, **kargs):
        """Iterate over the baseFolder, calling func on each item.

        Args:
            func (callable): A Callable object that must take a metadataObject type instance as it's first argument.

        Keyword Args:
            _return (None, bool or str): Controls how the return value from *func* is added to the DataFolder

        Returns:
            A list of the results of evaluating *func* for each item in the folder.

        Notes:
            If *_return* is None and the return type of *func* is the same type as the :py:class:`baseFolder` is
            storing, then the return value replaces trhe original :py:class:`Stoner.Core.metadataobject` in the
            :py:class:`baseFolder`. If *_result* is True the return value is added to the
            :py:class:`Stoner.Core.metadataObject`'s metadata under the name of the function. If *_result* is a
            string. then return result is stored in the corresponding name.
        """
        _return = kargs.pop("_return", None)
        _byname = kargs.pop("_byname", False)
        _serial = kargs.pop("_serial", False)
        self._folder.fetch()  # Prefetch thefolder in case we can do it in parallel
        p, imap = get_pool(_serial)
        for ix, (f, ret) in enumerate(
            imap(partial(_worker, func=func, args=args, kargs=kargs, byname=_byname), self._folder)
        ):
            new_d = f
            if self._folder.debug:
                print(ix, type(ret))
            if isinstance(ret, self._folder._type) and _return is None:
                try:  # Check if ret has same data type, otherwise will not overwrite well
                    if ret.data.dtype != f.data.dtype:
                        continue
                    else:
                        new_d = ret
                except AttributeError:
                    pass
            elif _return is not None:
                if isinstance(_return, bool) and _return:
                    _return = func.__name__
                new_d[_return] = ret
            name = self._folder.__names__()[ix]
            self._folder.__setter__(name, new_d)
            yield ret
        if p is not None:
            p.close()
            p.join()
