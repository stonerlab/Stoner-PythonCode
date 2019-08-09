#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base classes for the Stoner package
"""
__all__ = ["_evaluatable", "regexpDict", "string_to_type", "typeHintedDict", "metadataObject"]
from collections import OrderedDict, MutableMapping, Mapping
import re
import copy
import numpy as np
from numpy import NaN
import asteval

from ..compat import python_v3, string_types, int_types, _pattern_type
from ..tools import isiterable, isComparable
from .exceptions import assertion, StonerAssertionError

try:
    assertion(not python_v3)  # blist doesn't seem entirely reliable in 3.5 :-(
    from blist import sorteddict
except (StonerAssertionError, ImportError):  # Fail if blist not present or Python 3
    sorteddict = OrderedDict

_asteval_interp = None


def literal_eval(string):
    """Use the asteval module to interpret arbitary strings slightly safely.

    Args:
        string (str):
            String epxression to be evaluated.

    Returns:
        (object):
            Evaluation result.

    On the first call this will create a new asteval.Interpreter() instance and
    preload some key modules into the symbol table.
    """
    global _asteval_interp  # Ugly!
    if _asteval_interp is None:
        _asteval_interp = asteval.Interpreter(usersyms={"np": np, "re": re, "NaN": NaN, "nan": NaN, "None": None})
    try:
        return _asteval_interp(string, show_errors=False)
    except (SyntaxError, ValueError, NameError, IndexError, TypeError):
        raise ValueError("Cannot interpret {} as valid Python".format(string))


def string_to_type(value):
    """Given a string value try to work out if there is a better python type dor the value.

    First of all the first character is checked to see if it is a [ or { which would
    suggest this is a list of dictionary. If the value looks like a common boolean
    value (i.e. Yes, No, True, Fale, On, Off) then it is assumed to be a boolean value.
    Fianlly it interpretation as an int, float or string is tried.

    Args:
        value (string): string representation of he value
    Returns:
        A python object of the natural type for value
    """
    ret = None
    if not isinstance(value, string_types):
        raise TypeError("Value must be a string not a {}".format(type(value)))
    value = value.strip()
    if value:
        tests = ["list(" + value + ")", "dict(" + value + ")"]
        try:
            i = "[{".index(value[0])
            ret = literal_eval(tests[i])  # pylint: disable=eval-used
        except (SyntaxError, ValueError):
            if value.lower() in ["true", "yes", "on", "false", "no", "off"]:
                ret = value.lower() in ["true", "yes", "on"]  # Boolean
            else:
                for trial in [int, float, str]:
                    try:
                        ret = trial(value)
                        break
                    except ValueError:
                        continue
                else:
                    ret = None
    return ret


class _evaluatable(object):

    """Just a placeholder to indicate that special action needs to be taken to convert a string representation to a valid Python type."""

    pass


class regexpDict(sorteddict):

    """An ordered dictionary that permits looks up by regular expression."""

    allowed_keys = (object,)

    def __lookup__(self, name, multiple=False, exact=False):
        """Lookup name and find a matching key or raise KeyError.

        Parameters:
            name (str, _pattern_type): The name to be searched for

        Keyword Arguments:
            multiple (bool): Return a singl entry ()default, False) or multiple entries

        Returns:
            Canonical key matching the specified name.

        Raises:
            KeyError: if no key matches name.
        """
        ret = None
        try:  # name directly as key
            super(regexpDict, self).__getitem__(name)
            ret = name
        except (KeyError, TypeError):  # Fall back to regular expression lookup
            if not exact and not isinstance(name, string_types + int_types):
                name = repr(name)
            if exact:
                raise KeyError("{} not a key and exact match requested.".format(name))
            nm = name
            if isinstance(name, string_types):
                try:
                    nm = re.compile(name)
                except re.error:
                    pass
            elif isinstance(name, int_types):  # We can do this because we're an OrderedDict!
                try:
                    ret = sorted(self.keys())[name]
                except IndexError:
                    raise KeyError("{} is not a match to any key.".format(name))
            else:
                nm = name
            if isinstance(nm, _pattern_type):
                ret = [n for n in self.keys() if isinstance(n, string_types) and nm.match(n)]
        if ret is None or isiterable(ret) and not ret:
            raise KeyError("{} is not a match to any key.".format(name))
        else:
            if multiple:  # sort out returing multiple entries or not
                if not isinstance(ret, list):
                    ret = [ret]
            else:
                if isinstance(ret, list):
                    ret = ret[0]
            return ret

    def __getitem__(self, name):
        """Adds a lookup via regular expression when retrieving items."""
        return super(regexpDict, self).__getitem__(self.__lookup__(name))

    def __setitem__(self, name, value):
        """Overwrites any matching key, or if not found adds a new key."""
        try:
            key = self.__lookup__(name, exact=True)
        except KeyError:
            if not isinstance(name, self.allowed_keys):
                raise KeyError("{} is not a match to any key.".format(name))
            key = name
        super(regexpDict, self).__setitem__(key, value)

    def __delitem__(self, name):
        """Deletes keys that match by regular expression as well as exact matches"""
        super(regexpDict, self).__delitem__(self.__lookup__(name))

    def __contains__(self, name):
        """Returns True if name either is an exact key or matches when interpreted as a regular experssion."""
        try:
            name = self.__lookup__(name)
            return True
        except (KeyError, TypeError):
            return False

    def __eq__(self, other):
        """Define equals operation in terms of xor operation."""
        return len(self ^ other) == 0 and len(other ^ self) == 0

    def __xor__(self, other):
        """Give the difference between two arrays."""
        if not isinstance(other, Mapping):
            return NotImplemented
        mk = set(self.keys())
        ok = set(other.keys())
        if mk != ok:  # Keys differ
            return mk ^ ok
        # Do values differ?
        ret = OrderedDict()
        for (mk, mv), (ok, ov) in zip(sorted(self.items()), sorted(other.items())):
            if np.any(mv != ov) and isComparable(mv, ov):
                ret[mk] = (mv, ov)
        return ret

    def has_key(self, name):
        """Key is definitely in dictionary as literal"""
        return super(regexpDict, self).__contains__(name)


class typeHintedDict(regexpDict):

    """Extends a :py:class:`blist.sorteddict` to include type hints of what each key contains.

    The CM Physics Group at Leeds makes use of a standard file format that closely matches
    the :py:class:`DataFile` data structure. However, it is convenient for this file format
    to be ASCII text for ease of use with other programs. In order to represent metadata which
    can have arbitary types, the LabVIEW code that generates the data file from our measurements
    adds a type hint string. The Stoner Python code can then make use of this type hinting to
    choose the correct representation for the metadata. The type hinting information is retained
    so that files output from Python will retain type hints to permit them to be loaded into
    strongly typed languages (sch as LabVIEW).

    Attributes:
        _typehints (dict): The backing store for the type hint information
        __regexGetType (re): Used to extract the type hint from a string
        __regexSignedInt (re): matches type hint strings for signed intergers
        __regexUnsignedInt (re): matches the type hint string for unsigned integers
        __regexFloat (re): matches the type hint strings for floats
        __regexBoolean (re): matches the type hint string for a boolean
        __regexStrng (re): matches the type hint string for a string variable
        __regexEvaluatable (re): matches the type hint string for a compoind data type
        __types (dict): mapping of type hinted types to actual Python types
        __tests (dict): mapping of the regex patterns to actual python types

    Notes:
        Rather than subclassing a plain dict, this is a subclass of a :py:class:`blist.sorteddict` which stores the entries in a binary list structure.
        This makes accessing the keys much faster and also ensures that keys are always returned in alphabetical order.
    """

    allowed_keys = string_types
    # Force metadata keys to be strings
    _typehints = sorteddict()

    __regexGetType = re.compile(r"([^\{]*)\{([^\}]*)\}")
    # Match the contents of the inner most{}
    __regexSignedInt = re.compile(r"^I\d+")
    # Matches all signed integers
    __regexUnsignedInt = re.compile(r"^U / d+")
    # Match unsigned integers
    __regexFloat = re.compile(r"^(Extended|Double|Single)\sFloat")
    # Match floating point types
    __regexBoolean = re.compile(r"^Boolean")
    __regexString = re.compile(r"^(String|Path|Enum)")
    __regexEvaluatable = re.compile(r"^(Cluster||\d+D Array|List)")

    __types = OrderedDict(
        [  # Key order does matter here!
            ("Boolean", bool),
            ("I32", int),
            ("Double Float", float),
            ("Cluster", dict),
            ("AnonCluster", tuple),
            ("Array", np.ndarray),
            ("List", list),
            ("String", str),
        ]
    )
    # This is the inverse of the __tests below - this gives
    # the string type for standard Python classes

    __tests = [
        (__regexSignedInt, int),
        (__regexUnsignedInt, int),
        (__regexFloat, float),
        (__regexBoolean, bool),
        (__regexString, str),
        (__regexEvaluatable, _evaluatable()),
    ]

    # This is used to work out the correct python class for
    # some string types

    def __init__(self, *args, **kargs):
        """Construct the typeHintedDict.

        Args:
            *args, **kargs: Pass any parameters through to the dict() constructor.


        Calls the dict() constructor, then runs through the keys of the
        created dictionary and either uses the string type embedded in
        the keyname to generate the type hint (and remove the
        embedded string type from the keyname) or determines the likely
        type hint from the value of the dict element.
        """
        super(typeHintedDict, self).__init__(*args, **kargs)
        for key in list(self.keys()):  # Chekc through all the keys and see if they contain
            # type hints. If they do, move them to the
            # _typehint dict
            value = super(typeHintedDict, self).__getitem__(key)
            super(typeHintedDict, self).__delitem__(key)
            self[key] = value  # __Setitem__ has the logic to handle embedded type hints correctly

    @property
    def types(self):
        """Return the dictrionary of value types."""
        return self._typehints

    def findtype(self, value):
        """Determines the correct string type to return for common python classes.

        Args:
            value (any): The data value to determine the type hint for.

        Returns:
            A type hint string

        Note:
            Understands booleans, strings, integers, floats and np
            arrays(as arrays), and dictionaries (as clusters).
        """
        typ = "Invalid Type"
        if value is None:
            return "Void"
        for t in self.__types:
            if isinstance(value, self.__types[t]):
                if t == "Cluster" or t == "AnonCluster":
                    elements = []
                    if isinstance(value, dict):
                        for k in value:
                            elements.append(self.findtype(value[k]))
                    else:
                        for v in value:
                            elements.append(self.findtype(v))
                    tt = ","
                    tt = tt.join(elements)
                    typ = "Cluster (" + tt + ")"
                elif t == "Array":
                    z = np.zeros(1, dtype=value.dtype)
                    typ = "{}D Array ({})".format(len(np.shape(value)), self.findtype(z[0]))
                else:
                    typ = t
                break
        return typ

    def __mungevalue(self, t, value):
        """Based on a string type t, return value cast to an appropriate python class.

        Args:
            t (string): is a string representing the type
            value (any): is the data value to be munged into the
                        correct class
        Returns:
            Returns the munged data value

        Detail:
            The class has a series of precompiled regular
            expressions that will match type strings, a list of these has been
            constructed with instances of the matching Python classes. These
            are tested in turn and if the type string matches the constructor of
            the associated python class is called with value as its argument.
        """
        ret = None
        if t == "Invalid Type":  # Short circuit here
            return repr(value)
        for (regexp, valuetype) in self.__tests:
            m = regexp.search(t)
            if m is not None:
                if isinstance(valuetype, _evaluatable):
                    try:
                        if isinstance(value, string_types):  # we've got a string already don't need repr
                            ret = literal_eval(value)
                        else:
                            ret = literal_eval(repr(value))  # pylint: disable=eval-used
                    except ValueError:  # Oops just keep string format
                        ret = str(value)
                    except SyntaxError:
                        ret = ""
                    break
                else:
                    ret = valuetype(value)
                    break
        else:
            ret = str(value)
        return ret

    def __deepcopy__(self, memo):
        """Implements a deepcopy method for typeHintedDict to work around something that gives a hang in newer Python 2.7.x"""
        cls = self.__class__
        result = cls()
        memo[id(self)] = result
        for k in self:
            result[k] = copy.deepcopy(self[k])
            result.types[k] = self.types[k]
        return result

    def _get_name_(self, name):
        """Checks a string name for an embedded type hint and strips it out.

        Args:
            name(string): String containing the name with possible type hint embedeed
        Returns:
            (name,typehint) (tuple): A tuple containing just the name of the mateadata and (if found
                the type hint string),
        """
        search = str(name)
        m = self.__regexGetType.search(search)
        if m is not None:
            return m.group(1), m.group(2)
        elif not isinstance(name, string_types + int_types):
            return search, None
        else:
            return name, None

    def __getitem__(self, name):
        """Provides a get item method that checks whether its been given a typehint in the item name and deals with it appropriately.

        Args:
            name (string): metadata key to retrieve

        Returns:
            metadata value
        """
        key = name
        (name, typehint) = self._get_name_(name)
        name = self.__lookup__(name, True)
        value = [super(typeHintedDict, self).__getitem__(nm) for nm in name]
        if typehint is not None:
            value = [self.__mungevalue(typehint, v) for v in value]
        if len(value) == 0:  # pylint: disable=len-as-condition
            raise KeyError("{} is not a valid key even when interpreted as a sregular expression!".format(key))
        elif len(value) == 1:
            return value[0]
        else:
            return {k: v for k, v in zip(name, value)}

    def __setitem__(self, name, value):
        """Provides a method to set an item in the dict, checking the key for an embedded type hint or inspecting the value as necessary.

        Args:
            name (string): The metadata keyname
            value (any): The value to store in the metadata string

        Note:
            If you provide an embedded type string it is your responsibility
            to make sure that it correctly describes the actual data
            typehintDict does not verify that your data and type string are
            compatible.
        """
        name, typehint = self._get_name_(name)
        if typehint is not None:
            self._typehints[name] = typehint
            if value is None:  # Empty data so reset to string and set empty #RCT changed the test here
                super(typeHintedDict, self).__setitem__(name, "")
                self._typehints[name] = "String"
            else:
                try:
                    super(typeHintedDict, self).__setitem__(name, self.__mungevalue(typehint, value))
                except ValueError:
                    pass  # Silently fail
        else:
            self._typehints[name] = self.findtype(value)
            super(typeHintedDict, self).__setitem__(name, value)

    def __delitem__(self, name):
        """Deletes the specified key.

        Args:
            name (string): The keyname to be deleted
        """
        name = self._get_name_(name)[0]
        name = self.__lookup__(name)

        del self._typehints[name]
        super(typeHintedDict, self).__delitem__(name)

    def __repr__(self):
        """Create a text representation of the dictionary with type data."""
        ret = ["{}:{}:{}".format(repr(key), self.type(key), repr(self[key])) for key in sorted(self)]
        return "\n".join(ret)

    def copy(self):
        """Provides a copy method that is aware of the type hinting strings.

        This produces a flat dictionary with the type hint embedded in the key name.

        Returns:
            A copy of the current typeHintedDict
        """
        cls = self.__class__
        ret = cls()
        for k in self.keys():
            t = self._typehints[k]
            ret._typehints[k] = t
            super(typeHintedDict, ret).__setitem__(k, copy.deepcopy(self[k]))
        return ret

    def filter(self, name):
        """Filter the dictionary keys by name

        Reduce the metadata dictionary leaving only keys satisfied by name.

        Keyword Arguments:
            name(str or callable):
                either a str to match or a callable function that takes metadata key-value
                as an argument and returns True or False
        """
        rem = []
        for k in self.keys():
            if isinstance(name, string_types):
                if not name in k:
                    rem.append(k)
            elif hasattr(name, "__call__"):
                if not name(k):
                    rem.append(k)
            else:
                raise ValueError("name must be a string or a function")
        for k in rem:
            del self[k]

    def type(self, key):
        """Returns the typehint for the given k(s).

        This simply looks up the type hinting dictionary for each key it is given.

        Args:
            key (string or sequence of strings): Either a single string key or a iterable type containing
                keys
        Returns:
            The string type hint (or a list of string type hints)
        """
        if isinstance(key, string_types):
            return self._typehints[key]
        else:
            try:
                return [self._typehints[x] for x in key]
            except TypeError:
                return self._typehints[key]

    def export(self, key):
        """Exports a single metadata value to a string representation with type hint.

        In the ASCII based file format, the type hinted metadata is represented
        in the first column of a tab delimited text file as a series of lines
        with format keyname{typhint}=string_value.

        Args:
            key (string): The metadata key to export
        Returns:
            A string of the format : key{type hint} = value
        """
        if isinstance(self[key], string_types):  # avoid string within string problems and backslash overdrive
            ret = "{}{{{}}}={}".format(key, self.type(key), self[key])
        else:
            ret = "{}{{{}}}={}".format(key, self.type(key), repr(self[key]))
        return ret

    def export_all(self):
        """Return all the entries in the typeHintedDict as a list of exported lines.

        Returns:
            (list of str): A list of exported strings

        Notes:
            The keys are returned in sorted order as a result of the underlying blist.sorteddict meothd.
        """
        return [self.export(x) for x in self]

    def import_all(self, lines):
        """Reads multiple lines of strings and tries to import keys from them.

        Args:
            lines(list of str): The lines of metadata values to import.
        """
        for line in lines:
            self.import_key(line)

    def import_key(self, line):
        """Import a single key from a string like key{type hint} = value.

        This is the inverse of the :py:meth:`typeHintedDict.export` method.

        Args:
            line(str): The string line to be interpreted as a key-value pair.

        """
        parts = line.split("=")
        k = parts[0]
        v = "=".join(parts[1:])  # rejoin any = in the value string
        self[k] = v


class metadataObject(MutableMapping):

    """Provides a base class representing some sort of object that has metadata stored in a :py:class:`Stoner.Core.typeHintedDict` object.

    Attributes:
        metadata (typeHintedDict): of key-value metadata pairs. The dictionary tries to retain information about the type of data so as to aid import and
            export from CM group LabVIEW code.
   """

    def __init__(self, *args, **kargs):
        """Initialises the current metadata attribute."""
        metadata = kargs.pop("metadata", None)
        if metadata is not None:
            self.metadata.update(metadata)
        super(metadataObject, self).__init__()

    @property
    def metadata(self):
        """Read the metadata dictionary."""
        try:
            return self._metadata
        except AttributeError:  # Oops no metadata yet
            self._metadata = typeHintedDict()
            return self._metadata

    @metadata.setter
    def metadata(self, value):
        """Update the metadata object with type checking."""
        if not isinstance(value, typeHintedDict) and isiterable(value):
            self._metadata = typeHintedDict(value)
        elif isinstance(value, typeHintedDict):
            self._metadata = value
        else:
            raise TypeError(
                "metadata must be something that can be turned into a dictionary, not a {}".format(type(value))
            )

    def __getitem__(self, name):
        """Pass through to metadata dictionary."""
        return self.metadata[name]

    def __setitem__(self, name, value):
        """Pass through to metadata dictionary."""
        self.metadata[name] = value

    def __delitem__(self, name):
        """Pass through to metadata dictionary."""
        del self.metadata[name]

    def __eq__(self, other):
        """Implement am equality test for metadataObjects."""
        if not isinstance(other, metadataObject):
            return False
        if len(self) != len(other):
            return False
        ret = self.metadata ^ other.metadata
        return len(ret) == 0

    def __len__(self):
        """Pass through to metadata dictionary."""
        return len(self.metadata)

    def __iter__(self):
        """Pass through to metadata dictionary."""
        return self.metadata.__iter__()

    def keys(self):
        """Return the keys of the metadata dictionary."""
        for k in self.metadata.keys():
            yield k

    def items(self):
        """Make sure we implement an items that doesn't just iterate over self!"""
        for k, v in self.metadata.items():
            yield k, v

    def values(self):
        """Return the values of the metadata dictionary."""
        for v in self.metadata.values():
            yield v

    def save(self, filename=None, **kargs):
        """Stub method for a save function."""
        raise NotImplementedError("Save is not implemented in the base class.")

    def _load(self, filename, *args, **kargs):
        """Stub method for a load function."""
        raise NotImplementedError("Save is not implemented in the base class.")
