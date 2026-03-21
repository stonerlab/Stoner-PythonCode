# -*- coding: utf-8 -*-
"""Additional methods for BaseFolder."""
import fnmatch
import re
from collections.abc import Iterable, Mapping
from copy import copy, deepcopy
from os import path

from ..compat import _pattern_type, commonpath, int_types, string_types
from ..core.base import TypeHintedDict, metadataObject
from ..tools import operator
from ..tools.decorators import make_Class
from .utils import pathjoin

regexp_type = (_pattern_type,)


def _build_select_function(kwargs, arg):
    """Build a select function from an a list of keywords and a keyword name.

    Args:
        kwargs (dict):
            The keyword arguments passed to the select function.
        arg (str):
            Name of the keyword argument we're considering.

    Returns:
        tuple of:
            Callable function that takes two arguments and returns a boolean if the two arguments match.
            str name of key to look up
    """
    parts = arg.split("__")
    negate = kwargs.pop("negate", False)
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
        match kwargs[arg]:
            case (_, _):
                op = "between"  # Assume two length tuples are testing for range
            case Iterable() if not isinstance(kwargs[arg], str):
                op = "in"  # Assume other iterables are testing for membership
            case _:
                op = "eq"
    func = operator[op]
    if negate:

        def _func(k, v):
            return not func(k, v)

        func = _func
    return func, arg


def add_group(fldr, key):
    """Add a new group to the current BaseFolder with the given key.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.
        key(string): A hashable value to be used as the dictionary key in the groups dictionary
    Returns:
        A copy of the objectFolder

    Note:
        If key already exists in the groups dictionary then no action is taken.

    Todo:
        Propagate any extra attributes into the groups.
    """
    if key in fldr.groups:  # do nothing here
        pass
    else:
        new_group = fldr.__clone__(attrs_only=True)
        fldr.groups[key] = new_group
        fldr.groups[key].key = key
        fldr.groups[key].root = path.join(fldr.root, str(key))
    return fldr


def all(fldr):  # pylint: disable=redefined-builtin
    """Iterate over all the files in the Folder and all it's sub Folders recursely.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.

    Yields:
        (path/filename,file)
    """
    for g in fldr.groups.values():
        for p, d in g.all():
            p = path.join(fldr.key, p)
            yield p, d
    for d in fldr:
        yield d.filename, d


def clear(fldr):
    """Clear the subgroups.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.
    """
    fldr.groups.clear()
    fldr.__clear__()


def compress(fldr, base=None, key=".", keep_terminal=False):
    """Compresses all empty groups from the root up until the first non-empty group is located.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.

    Keyword Args:
        base (str,None):
            default None
        key (str):
            default "."
        keep_terminal (bool):
            default False

    Returns:
        A copy of the now flattened DatFolder
    """
    return fldr.groups.compress(base=base, key=key, keep_terminal=keep_terminal)


def count(fldr, value):  # pylint:  disable=arguments-differ
    """Provide a count method like a sequence.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.
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
            return len(fnmatch.filter(fldr.__names__(), value))
        return fldr.__names__().count(fldr.__lookup__(value))
    if isinstance(value, _pattern_type):
        match = [1 for n in fldr.__names__() if value.search(n)]
        return len(match)
    if isinstance(value, metadataObject):
        match = [1 for d in fldr if d == value]
        return len(match)
    raise TypeError(f"Failed to count as value was a {type(value)} which we couldn't use.")


def fetch(fldr):
    """Preload the contents of the BaseFolder.

    In the base  class this is a NOP because the objects are all in memory anyway.
    """
    return fldr


def file(fldr, name, value, create=True, pathsplit=None):
    """Recursely add groups in order to put the named value into a virtual tree of :py:class:`BaseFolder`.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.
        name(str):
            A name (which may be a nested path) of the object to file.
        value(metadataObject):
            The object to be filed - it should be an instance of :py:attr:`BaseFolder.type`.

    Keyword Aprameters:
        create(bool):
            Whether to create missing groups or to raise an error (default True to create groups).
        pathsplit(str or None):
            Character to use to split the name into path components. Defaults to using os.path.split()

    Returns:
        (BaseFolder):
            A reference to the group where the value was eventually filed

    """
    if pathsplit is None:
        pathsplit = r"[\\/]+"
    pathsplit = re.compile(pathsplit)
    pth = pathsplit.split(name)
    tmp = fldr
    for ix, section in enumerate(pth):
        if ix == len(pth) - 1:
            existing = tmp.__getter__(section, instantiate=None) if section in tmp.__names__() else None
            if (
                existing is None
                or (isinstance(value, fldr.type) and id(existing) != id(value))
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


def filter(  # pylint: disable=redefined-builtin
    fldr, filter=None, invert=False, copy=False, recurse=False, prune=True
):
    r"""Filter the current set of files by some criterion.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.
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
            If True, execute a :py:meth:`BaseFolder.prune` to remove empty groups after filering

    Returns:
        The current objectFolder object
    """
    names = []
    if copy:
        result = deepcopy(fldr)
    else:
        result = fldr
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


def filterout(fldr, filter, copy=False, recurse=False, prune=True):  # pylint: disable=redefined-builtin
    """Synonym for fldr.filter(filter,invert=True).

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.
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
            If True, execute a :py:meth:`BaseFolder.prune` to remove empty groups after filering

    Returns:
        The current objectFolder object with the files in the file list filtered.
    """
    return fldr.filter(filter, invert=True, copy=copy, recurse=recurse, prune=prune)


def flatten(fldr, depth=None):
    """Compresses all the groups and sub-groups iunto a single flat file list.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.

    Keyword Args:
        depth (int or None):
            Only flatten ub-=groups that are within (*depth* of the deepest level.

    Returns:
        A copy of the now flattened DatFolder
    """
    if isinstance(depth, int_types):
        if fldr.depth <= depth:
            return fldr.flatten()
        for g, val in fldr.groups.items():
            val.flatten(depth)
        return fldr

    for g, val in fldr.groups.items():
        if fldr.debug:
            print(f"{fldr.key}->{val.key}")
        val.flatten()
        for n in val.__names__():
            value = val.__getter__(n, instantiate=None)
            old_name = pathjoin(val.root, n)
            new_name = path.relpath(old_name, start=fldr.root)
            if fldr.debug:
                print(f"\t{g}::{old_name}=>{new_name}")

            if hasattr(value, "filename"):
                value.filename = new_name

            if isinstance(value, string_types):  # We haven't loaded this yet, in which case change value to new_name
                value = new_name
            fldr.__setter__(new_name, value)
        val.__clear__()
    fldr.groups = {}
    return fldr


def get(fldr, name, default=None):
    """Return either a sub-group or named object from this folder.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.
        name (str):
            Name of subgroup or object to fetch.

    Keyword Args:
        default (Any):
            What to return if no matching name. Defaults to None

    Returns:
        (Group, or metadataObject):
            Either a subgroup or object or default.
    """
    try:
        ret = fldr[name]
    except (KeyError, IndexError):
        ret = default
    return ret


def group(fldr, key):
    """Sort Files into a series of objectFolders according to the value of the key.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.
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

        def _key(x):
            return x.get(k, "None")

        key = _key
    for x in fldr:
        v = key(x)
        if v not in fldr.groups:
            fldr.add_group(v)
        fldr.groups[v].append(x)
    fldr.__clear__()
    if len(next_keys) > 0:
        for val in fldr.groups.values():
            val.group(next_keys)
    return fldr


def index(fldr, value, start=None, stop=None):
    """Provide an index method like a sequence.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.
        value(str, regexp, or :py:class:`Stoner.Core.metadataObject`):
            The thing to search for.

    Keyword Arguments:
        start (int):
            Limit the index search to a sub-range as per Python 3.5+ list.index
        stop (int):
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
        stop = len(fldr)
    search = fldr.__names__()[start:stop]
    if isinstance(value, string_types):
        if "*" in value or "?" in value:  # globbing pattern
            m = fnmatch.filter(search, value)
            if len(m) > 0:
                return search.index(m[0]) + start
            raise ValueError(f"{value} is not a name of a metadataObject in this BaseFolder.")
        return search.index(fldr.__lookup__(value)) + start
    if isinstance(value, _pattern_type):
        for i, n in enumerate(search):
            if value.search(n):
                return i + start
        raise ValueError("No match for any name of a metadataObject in this BaseFolder.")
    if isinstance(value, metadataObject):
        for i, n in enumerate(search):
            if value == n:
                return i + start
        raise ValueError("No match for any name of a metadataObject in this BaseFolder.")
    raise TypeError(f"Could not use value of type {type(value)} for index.")


def append(fldr, value):
    """Append an item to the folder object.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.
        value (metadataObject):
            Metadata object to be added.

    """
    fldr.insert(len(fldr), value)


def items(fldr):
    """Return the key,value pairs for the subbroups of this folder.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.

    Yields:
        Pass through to the group.items() iterator.
    """
    yield from fldr.groups.items()


def keys(fldr):
    """Return the keys used to access the sub-=groups of this folder.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.

    Yields:
        PAss through to the groups.keys() iterator.
    """
    yield from fldr.groups.keys()


def make_name(fldr, value=None):
    """Construct a name from the value object if possible.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.

    Keyword Args:
        value (metadataObject, None):
            Object to be named.

    Returns:
        (str):
            A new unique name.
    """
    if isinstance(value, fldr.type):
        name = getattr(value, "filename", "")
        if name == "":
            name = f"Untitled-{fldr._last_name}"
            while name in fldr:
                fldr._last_name += 1
                name = f"Untitled-{fldr._last_name}"
        return name
    if isinstance(value, string_types):
        return value
    name = f"Untitled-{fldr._last_name}"
    while name in fldr:
        fldr._last_name += 1
        name = f"Untitled-{fldr._last_name}"
    return name


def pop(fldr, name=-1, default=None):  # pylint: disable=arguments-differ,arguments-renamed
    """Return and remove either a subgroup or named object from this folder."""
    try:
        ret = fldr[name]
        del fldr[name]
    except (KeyError, IndexError):
        ret = default
    return ret


def popitem(fldr):
    """Return the most recent subgroup from this folder."""
    return fldr.groups.popitem()


def prune(fldr, name=None):
    """Remove any empty groups from the objectFolder (and subgroups).

    Returns:
        A copy of the pruned objectFolder.
    """
    return fldr.groups.prune(name=name)


def select(fldr, *args, **kwargs):
    """Select a subset of the objects in the folder based on flexible search criteria on the metadata.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.
        args (various):
            A single positional argument if present is interpreted as follows:

            *   If a callable function is given, the entire metadataObject is presented to it.
                If it evaluates True then that metadataObject is selected. This allows arbitrary select operations
            *   If a dict is given, then it and the kwargs dictionary are merged and used to select the
                metadataObjects

    Keyword Arguments:
        recurse (bool):
            Also recursively select through the sub groups
        kwargs (varuous):
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
            A new BaseFolder instance that contains just the matching metadataObjects.

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
    recurse = kwargs.pop("recurse", False)
    negate = kwargs.pop("negate", False)
    if len(args) == 1:
        if callable(args[0]):
            kwargs["__"] = args[0]
        elif isinstance(args[0], dict):
            kwargs.update(args[0])
    result = fldr.__clone__(attrs_only=True)
    if recurse:
        gkwargs = {}
        gkwargs.update(kwargs)
        gkwargs["negate"] = negate
        gkwargs["recurse"] = True
        for g, val in fldr.groups.items():
            result.groups[g] = val.select(*args, **gkwargs)
    if isinstance(fldr.pattern[0], regexp_type):
        pattern_keys = list(fldr.pattern[0].groupindex.keys())
        for karg in kwargs:
            if karg.split("__")[0] not in pattern_keys:
                must_read = True
                break
        else:
            must_read = False
    else:
        must_read = True

    for f in fldr.objects:
        if must_read and isinstance(f, string_types):
            f = fldr.__getter__(f, instantiate=True)
        placer = f
        if not must_read:
            match = fldr.pattern[0].search(f)
            f = TypeHintedDict(match.groupdict())

        for arg, val in kwargs.items():
            if callable(val) and val(f):
                break
            if isinstance(arg, string_types):
                skwargs = copy(kwargs)
                skwargs["negate"] = negate
                func, key = _build_select_function(skwargs, arg)
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


def setdefault(fldr, k, d=None):
    """Return or set a subgroup or named object."""
    fldr[k] = fldr.get(k, d)
    return fldr[k]


def slice_metadata(fldr, key, output="smart"):
    """Return an array of the metadata values for each item/file in the top level group.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.
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
    return fldr.metadata.slice(key, output=output)


def sort(fldr, key=None, reverse=False, recurse=True):
    """Sort the files by some key.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.

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
        for grp in fldr.groups.values():
            grp.sort(key=key, reverse=reverse, recurse=recurse)
    tmp = fldr.clone
    if isinstance(key, string_types):
        k = [(x.get(key), i) for i, x in enumerate(tmp)]
        k = sorted(k, reverse=reverse)
        new_order = [tmp[i] for x, i in k]
        new_names = [fldr.__names__()[i] for x, i in k]
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
        new_order = sorted(order, key=lambda x: key(fldr[x]), reverse=reverse)
        new_order = [tmp.__names__()[i] for i in new_order]
        new_names = new_order
    fldr.__clear__()
    for obj, k in zip(new_order, new_names):
        fldr.__setter__(k, obj)

    return fldr


def unflatten(fldr):
    """Take the file list an unflattens them according to the file paths.

    Returns:
        A copy of the objectFolder
    """
    if len(fldr):
        if len(fldr) == 1:
            fldr.directory = path.join(fldr.directory, path.dirname(fldr.__names__()[0]))
        else:
            fldr.directory = commonpath([path.realpath(path.join(fldr.directory, x)) for x in fldr.__names__()])
        names = fldr.__names__()
        relpaths = [path.relpath(path.join(fldr.directory, f), fldr.directory) for f in names]
        dels = []
        for i, f in enumerate(relpaths):
            ret = fldr.file(f, fldr.__getter__(names[i], instantiate=None))
            if isinstance(ret, make_Class("folders.core.BaseFolder", None)):  # filed ok
                dels.append(i)
        for i in sorted(dels, reverse=True):
            del fldr[i]
    for val in fldr.groups.values():
        val.unflatten()
    return fldr


def update(fldr, other):
    """Update this folder with a dictionary or another folder."""
    BaseFolder = make_Class("folders.core.BaseFolder", None)
    match other:
        case Mapping():
            for k in other:
                fldr[k] = other[k]
        case BaseFolder():
            for k in other.groups:
                if k in fldr.groups:
                    fldr.groups[k].update(other.groups[k])
                else:
                    fldr.groups[k] = other.groups[k].clone
            for k in other.__names__():
                if k in fldr.__names__():
                    fldr.__setter__(fldr.__lookup__(k), other.__getter__(other.__lookup__(k)).clone)
                else:
                    fldr.append(other.__getter__(other.__lookup__(k)).clone)
        case _:
            raise TypeError(f"Unable to update with a {type(other)}")


def values(fldr):
    """Return the sub-groups of this folder."""
    return fldr.groups.values()


def walk_groups(fldr, walker, **kwargs):
    """Walk through a hierarchy of groups and calls walker for each file.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.
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
        **kwargs:
            Other keyword arguments

    Notes:
        The walker function should have a prototype of the form::

            walker(f,list_of_group_names,**walker_args)

        where f is either a objectFolder or metadataObject.
    """
    group = kwargs.pop("group", False)
    replace_terminal = kwargs.pop("replace_terminal", False)
    only_terminal = kwargs.pop("only_terminal", True)
    walker_args = kwargs.pop("walker_args", {})
    walker_args = {} if walker_args is None else walker_args
    return fldr._walk_groups(
        walker,
        group=group,
        replace_terminal=replace_terminal,
        only_terminal=only_terminal,
        walker_args=walker_args,
        breadcrumb=[],
    )


def zip_groups(fldr, groups):
    """Return a list of tuples of metadataObjects drawn from the specified groups.

    Args:
        fldr (BaseFolder):
            DataFolder instance when not a bound method.
        groups(list of strings):
            A list of keys of groups in the Lpy:class:`objectFolder`

    Returns:
        A list of tuples of groups of files:
            [(grp_1_file_1,grp_2_file_1....grp_n_files_1),(grp_1_file_2,
            grp_2_file_2....grp_n_file_2)....(grp_1_file_m,grp_2_file_m...grp_n_file_m)]
    """
    if not isinstance(groups, list):
        raise SyntaxError("groups must be a list of groups")
    grps = [list(fldr.groups[x]) for x in groups]
    return zip(*grps)
