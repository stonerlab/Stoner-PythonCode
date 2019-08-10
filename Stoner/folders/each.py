# -*- coding: utf-8 -*-
"""
:py:mod:`Stoner.folders.each` provides the classes and support functions for the :py:attr:`Stoner.DataFolder.each` magic attribute.
"""
__all__ = ["item"]
from inspect import ismethod
import numpy as _np_
from functools import wraps, partial
from traceback import format_exc
from .utils import get_pool


def _worker(d, **kwargs):
    """Support function to run an arbitary function over a :py:class:`Stoner.Data` object."""
    byname = kwargs.get("byname", False)
    func = kwargs.get("func", lambda x: x)
    if byname:
        func = getattr(d, func, lambda x: x)
    args = kwargs.get("args", tuple())
    kargs = kwargs.get("kargs", dict)
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


class item(object):

    """Provides a proxy object for accessing methods on the inividual members of a Folder.
    8
    Notes:
        The prupose of this class is to allow it to be explicit that we're calling methods
        on the members of the folder rather than a collective method. This allows us to work
        around nameclashes.
    """

    _folder = None

    def __init__(self, folder):

        self._folder = folder

    def __call__(self, func, *args, **kargs):
        """Iterate over the baseFolder, calling func on each item.

        Args:
            func (callable): A Callable object that must take a metadataObject type instance as it's first argument.

        Keyword Args:
            _return (None, bool or str): Controls how the return value from *func* is added to the DataFolder

        Returns:
            A list of the results of evaluating *func* for each item in the folder.

        Notes:
            If *_return* is None and the return type of *func* is the same type as the :py:class:`baseFolder` is storing, then
            the return value replaces trhe original :py:class:`Stoner.Core.metadataobject` in the :py:class:`baseFolder`.
            If *_result* is True the return value is added to the :py:class:`Stoner.Core.metadataObject`'s metadata under the name
            of the function. If *_result* is a string. then return result is stored in the corresponding name.
        """
        # Just call the iter generator but assemble into a list.
        return list(self.iter(func, *args, **kargs))

    def __dir__(self):
        """Return a list of the common set of attributes of the instances in the folder."""
        if len(self._folder) != 0:
            res = set(dir(self._folder[0]))
        else:
            res = set()
        if len(self._folder) > 0:
            for d in self._folder[1:]:
                res &= set(dir(d))
        return list(res)

    def __getattr__(self, name):
        """Handles some special case attributes that provide alternative views of the objectFolder

        Args:
            item (string): The attribute name being requested

        Returns:
            Depends on the attribute

        """
        try:
            return super(item, self).__getattr__(name)
        except AttributeError:
            pass
        try:
            instance = self._folder.instance
            if ismethod(getattr(instance, name, None)):  # It's a method
                ret = self.__getattr_proxy(name)
            else:  # It's a static attribute
                if name in self._folder._object_attrs:
                    ret = self._folder._object_attrs[name]
                elif len(self._folder):
                    ret = [(not hasattr(x, name), getattr(x, name, None)) for x in self._folder]
                    mask, values = zip(*ret)
                    ret = _np_.ma.MaskedArray(values)
                    ret.mask = mask
                else:
                    ret = getattr(instance, name, None)
                if ret is None:
                    raise AttributeError
        except AttributeError:  # Ok, pass back
            raise AttributeError("{} is not an Attribute of {} or {}".format(name, type(self), type(instance)))
        return ret

    def __setattr__(self, name, value):
        """Setting the attrbute on .each sets it on all instantiated objects and in _object_attrs.

        Args:
            name(str): Attribute to set
            value (any): Value to set
        """
        if name in self.__dict__ or name.startswith("_"):  # Handle setting our own attributes
            super(item, self).__setattr__(name, value)
        elif name in dir(self._folder.instance):  # This is an instance attribute
            self._folder._object_attrs[name] = value  # Add to attributes to be set on load
            for d in self._folder.__names__():  # And set on all instantiated objects
                if isinstance(self._folder.__getter__(d, instantiate=False), self._folder.type):
                    d = self._folder.__gett__(d)
                    setattr(d, name, value)
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
            kargs[
                "_byname"
            ] = True  # Tell the call that we're going to find the function by name as an instance method
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
            If *_return* is None and the return type of *func* is the same type as the :py:class:`baseFolder` is storing, then
            the return value replaces trhe original :py:class:`Stoner.Core.metadataobject` in the :py:class:`baseFolder`.
            If *_result* is True the return value is added to the :py:class:`Stoner.Core.metadataObject`'s metadata under the name
            of the function. If *_result* is a string. then return result is stored in the corresponding name.
        """
        _return = kargs.pop("_return", None)
        _byname = kargs.pop("_byname", False)
        self._folder.fetch()  # Prefetch thefolder in case we can do it in parallel
        p, imap = get_pool()
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
