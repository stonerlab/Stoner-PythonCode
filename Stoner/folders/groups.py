# -*- coding: utf-8 -*-
"""Provides the classes and support functions for the :py:attr:`Stoner.DataFolder.groups` magic attribute."""

__all__ = ["GroupsDict"]

from collections.abc import Iterable
import fnmatch
from os import path

from Stoner.core.base import regexpDict


class GroupsDict(regexpDict):
    """A typeHinted dictionary to manages collections of :py:class:`Stoner.folders.core.baseFolder` objects."""

    def __init__(self, *args, **kargs):
        """Capture a *base* keyuword that sets the parent :py:class:`Stoner.DataFolder` instance."""
        self.base = kargs.pop("base", None)
        super().__init__(*args, **kargs)

    def __setitem__(self, name, value):
        """Enforce type checking on values."""
        if not isinstance(value, type(self.base)) and self.base:
            raise ValueError(f"groups attribute can only contain {type(type(self.base))} objects not {type(value)}")
        super().__setitem__(name, value)

    def compress(self, base=None, key=".", keep_terminal=False):
        """Compresses all empty groups from the root up until the first non-empty group is located.

        Returns:
            A copy of the now flattened DatFolder
        """
        if base is None:
            base = self.base
        if not len(self.base):
            for g in list(self.keys()):
                nk = path.join(key, g)
                base.groups[nk] = self[g]
                del self[g]
                base.groups[nk].compress(base=base, key=nk, keep_terminal=keep_terminal)
                if len(base.groups[nk].groups) == 0 and not keep_terminal:
                    for f in base.groups[nk].__names__():
                        obj = base.groups[nk].__getter__(f, instantiate=None)
                        nf = path.join(nk, f)
                        base.__setter__(nf, obj)
                        base.groups[nk].__deleter__(f)
                if len(base.groups[nk]) == 0 and len(base.groups[nk].groups) == 0:
                    del base.groups[nk]
        self.base.groups = self
        return self.base

    def keep(self, name):
        """Remove any groups that neither match name nor are a sub-folder of a group with a matching name.

        Args:
            name (str or iterable of str):
                Name(s) (or glob patterns) of groups to keep.

        Returns:
            A copy of the  baseFolder with the retained groups.
        """
        keys = list(self.keys())
        if isinstance(name, str):
            name = [name]
        if not isinstance(name, Iterable):
            raise TypeError(f"parameter name must be a string or iterable of strings, not a {type(name)}")
        for grp in keys:
            for nm in name:
                if fnmatch.fnmatch(grp, nm):
                    break
            else:
                g = self[grp]
                g.groups.keep(name)
                if not len(g.groups):
                    del self[grp]
        self.base.groups = self
        return self.base

    def prune(self, name=None):
        """Remove any empty groups from the objectFolder (and subgroups).

        Returns:
            A copy of thte pruned objectFolder.
        """
        keys = list(self.keys())
        for grp in keys:
            g = self[grp]
            g.groups.prune(name=name)
            if name is not None:
                if fnmatch.fnmatch(grp, name):
                    del self[grp]
            elif not len(g) and not len(g.groups):
                del self[grp]
        self.base.groups = self
        return self.base
