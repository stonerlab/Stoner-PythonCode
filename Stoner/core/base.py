#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base classes for the Stoner package."""

__all__ = ["_evaluatable", "regexpDict", "string_to_type", "TypeHintedDict", "metadataObject"]
import copy
import datetime
import re
from collections.abc import Generator, Iterable, Mapping, MutableMapping, Sequence
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import asteval
import numpy as np
from dateutil import parser
from numpy import nan

try:
    import pandas as pd
except ImportError:
    pd = None

from ..compat import _pattern_type, int_types, string_types
from ..tools import isComparable, isiterable
from .exceptions import StonerAssertionError
from .Typing import Filename, RegExp, String_Types

try:
    from blist import sorteddict as SortedDict
except (StonerAssertionError, ImportError):  # Fail if blist not present or Python 3
    from collections import OrderedDict

    SortedDict = OrderedDict

_asteval_interp = None


def _parse_date(string: str) -> datetime.datetime:
    """Run the dateutil parser with a UK sensible date order."""
    parserinfo = parser.parserinfo(dayfirst=True)
    return parser.parse(string, parserinfo)


def literal_eval(string: str) -> Any:
    """Use the asteval module to interpret arbitrary strings slightly safely.

    Args:
        string (str):
            String epxression to be evaluated.

    Returns:
        (object):
            Evaluation result.

    On the first call this will create a new asteval.Interpreter() instance and
    preload some key modules into the symbol table.
    """
    global _asteval_interp  # pylint: disable=W0603
    if _asteval_interp is None:
        _asteval_interp = asteval.Interpreter(
            usersyms={"np": np, "re": re, "NaN": nan, "nan": nan, "None": None, "datetime": datetime}
        )
    try:
        return _asteval_interp(string, show_errors=False)
    except (SyntaxError, ValueError, NameError, IndexError, TypeError) as err:
        raise ValueError(f"Cannot interpret {string} as valid Python") from err


def string_to_type(value: String_Types) -> Any:
    """Given a string value try to work out if there is a better python type dor the value.

    First of all the first character is checked to see if it is a [ or { which would
    suggest this is a list of dictionary. If the value looks like a common boolean
    value (i.e. Yes, No, True, Fale, On, Off) then it is assumed to be a boolean value.
    Finally it interpretation as an int, float or string is tried.

    Args:
        value (string):
            string representation of he value
    Returns:
        A python object of the natural type for value
    """
    ret = None
    if not isinstance(value, string_types):
        raise TypeError(f"Value must be a string not a {type(value)}")
    value = value.strip()
    if value != "None":
        tests = ["list(" + value + ")", "dict(" + value + ")"]
        try:
            i = "[{".index(value[0])
            ret = literal_eval(tests[i])  # pylint: disable=eval-used
        except (SyntaxError, ValueError):
            if value.lower() in ["true", "yes", "on", "false", "no", "off"]:
                ret = value.lower() in ["true", "yes", "on"]  # Boolean
            else:
                for trial in [int, float, _parse_date, str]:
                    try:
                        ret = trial(value)
                        break
                    except (ValueError, OverflowError, TypeError):
                        continue
                else:
                    ret = None
        except IndexError:  # raised when 0-length struing is used
            ret = value
    return ret


class _evaluatable:
    """Placeholder to indicate that special action needed to convert a string representation to valid Python type."""


class regexpDict(SortedDict):
    """An ordered dictionary that permits looks up by regular expression."""

    allowed_keys: Tuple = (object,)

    def __lookup__(
        self, name: Union[str, RegExp], multiple: bool = False, exact: bool = False
    ) -> Union[Any, List[Any]]:
        """Lookup name and find a matching key or raise KeyError.

        Parameters:
            name (str, _pattern_type):
                The name to be searched for

        Keyword Arguments:
            multiple (bool):
                Return a single entry ()default, False) or multiple entries
            exact(bool):
                Do not do a regular expression search, match the exact string only.

        Returns:
            Canonical key matching the specified name.

        Raises:
            KeyError: if no key matches name.
        """
        ret = None
        try:  # name directly as key
            super().__getitem__(name)
            ret = name
        except (KeyError, TypeError) as err:  # Fall back to regular expression lookup
            if not exact and not isinstance(name, string_types + int_types):
                name = repr(name)
            if exact:
                raise KeyError(f"{name} not a key and exact match requested.") from err
            nm = name
            if isinstance(name, string_types):
                try:
                    nm = re.compile(name)
                except re.error:
                    pass
            elif isinstance(name, int_types):  # We can do this because we're a dict!
                try:
                    ret = sorted(self.keys())[name]
                except IndexError as err:
                    raise KeyError(f"{name} is not a match to any key.") from err
            else:
                nm = name
            if isinstance(nm, _pattern_type):
                ret = [n for n in self.keys() if isinstance(n, string_types) and nm.match(n)]
                if not ret:
                    ret = [n for n in self.keys() if isinstance(n, string_types) and nm.search(n)]

        if ret is None or isiterable(ret) and not ret:
            raise KeyError(f"{name} is not a match to any key.")
        if multiple:  # sort out returning multiple entries or not
            if not isinstance(ret, list):
                ret = [ret]
        else:
            if isinstance(ret, list):
                ret = ret[0]
        return ret

    def __getitem__(self, name: Any) -> Any:
        """Add a lookup via regular expression when retrieving items."""
        return super().__getitem__(self.__lookup__(name))

    def __setitem__(self, name: Any, value: Any) -> None:
        """Overwrite any matching key, or if not found adds a new key."""
        try:
            key = self.__lookup__(name, exact=True)
        except KeyError as err:
            if not isinstance(name, self.allowed_keys):
                raise KeyError(f"{name} is not a match to any key.") from err
            key = name
        super().__setitem__(key, value)

    def __delitem__(self, name: Any) -> None:
        """Delete keys that match by regular expression as well as exact matches."""
        super().__delitem__(self.__lookup__(name))

    def __contains__(self, name: Any) -> bool:
        """Return True if name either is an exact key or matches when interpreted as a regular expression."""
        try:
            name = self.__lookup__(name)
            return True
        except (KeyError, TypeError):
            return False

    def __eq__(self, other: Any) -> bool:
        """Define equals operation in terms of xor operation."""
        if not isinstance(other, Mapping):
            return NotImplemented
        return len(self ^ other) == 0 and len(other ^ self) == 0

    def __sub__(self, other: Mapping) -> "regexpDict":
        """Give the difference between two arrays."""
        if not isinstance(other, Mapping):
            return NotImplemented
        mk = set(self.keys())
        ok = set(other.keys())
        ret = type(self)({k: self[k] for k in (mk - ok)})
        return ret

    def __xor__(self, other: Mapping) -> Union["regexpDict", Set[Any]]:
        """Give the difference between two arrays."""
        if not isinstance(other, Mapping):
            return NotImplemented
        mk = set(self.keys())
        ok = set(other.keys())
        if mk != ok:  # Keys differ
            return mk ^ ok
        # Do values differ?
        ret = type(self)()
        for (mk, mv), (ok, ov) in zip(sorted(self.items()), sorted(other.items())):
            if np.any(mv != ov) and isComparable(mv, ov):
                ret[mk] = (mv, ov)
        return ret

    def __or__(self, other):
        """Implement Python 3.9 style or operator to do a merge."""
        ret = self.copy()
        ret.update(other)
        return ret

    def __ior__(self, other):
        """Implement Python 3.9 style inplace or operator to do an update."""
        self.update(other)
        return self

    def has_key(self, name: Any) -> bool:
        """Key is definitely in dictionary as literal."""
        return super().__contains__(name)


class TypeHintedDict(regexpDict):
    """Extends a :py:class:`blist.sorteddict` to include type hints of what each key contains.

    The CM Physics Group at Leeds makes use of a standard file format that closely matches
    the :py:class:`DataFile` data structure. However, it is convenient for this file format
    to be ASCII text for ease of use with other programs. In order to represent metadata which
    can have arbitrary types, the LabVIEW code that generates the data file from our measurements
    adds a type hint string. The Stoner Python code can then make use of this type hinting to
    choose the correct representation for the metadata. The type hinting information is retained
    so that files output from Python will retain type hints to permit them to be loaded into
    strongly typed languages (sch as LabVIEW).

    Attributes:
        _typehints (dict):
            The backing store for the type hint information
        __regexGetType (re):
            Used to extract the type hint from a string
        __regexSignedInt (re):
            matches type hint strings for signed integers
        __regexUnsignedInt (re):
            matches the type hint string for unsigned integers
        __regexFloat (re):
            matches the type hint strings for floats
        __regexBoolean (re):
            matches the type hint string for a boolean
        __regexStrng (re):
            matches the type hint string for a string variable
        __regexEvaluatable (re):
            matches the type hint string for a compoind data type
        __types (dict):
            mapping of type hinted types to actual Python types
        __tests (dict):
            mapping of the regex patterns to actual python types

    Notes:
        Rather than subclassing a plain dict, this is a subclass of a :py:class:`blist.sorteddict` which stores the
        entries in a binary list structure. This makes accessing the keys much faster and also ensures that keys are
        always returned in alphabetical order.
    """

    allowed_keys: Tuple = string_types
    # Force metadata keys to be strings

    __regexGetType: RegExp = re.compile(r"([^\{]*)\{([^\}]*)\}")
    # Match the contents of the inner most{}
    __regexSignedInt: RegExp = re.compile(r"^I\d+")
    # Matches all signed integers
    __regexUnsignedInt: RegExp = re.compile(r"^U / d+")
    # Match unsigned integers
    __regexFloat: RegExp = re.compile(r"^(Extended|Double|Single)\sFloat")
    # Match floating point types
    __regexBoolean: RegExp = re.compile(r"^Boolean")
    __regexString = re.compile(r"^(String|Path|Enum)")
    __regexTimestamp: RegExp = re.compile(r"Timestamp")
    __regexEvaluatable: RegExp = re.compile(r"^(Cluster||\d+D Array|List)")

    __types: Dict[str, Type] = dict(
        [  # Key order does matter here!
            ("Boolean", bool),
            ("I32", int),
            ("Double Float", float),
            ("Cluster", dict),
            ("AnonCluster", tuple),
            ("Array", np.ndarray),
            ("List", list),
            ("Timestamp", datetime.datetime),
            ("String", str),
        ]
    )
    # This is the inverse of the __tests below - this gives
    # the string type for standard Python classes

    __tests: List[Tuple] = [
        (__regexSignedInt, int),
        (__regexUnsignedInt, int),
        (__regexFloat, float),
        (__regexBoolean, bool),
        (__regexTimestamp, datetime.datetime),
        (__regexString, str),
        (__regexEvaluatable, _evaluatable()),
    ]

    # This is used to work out the correct python class for
    # some string types

    def __init__(self, *args: Any, **kargs: Any) -> None:
        """Construct the TypeHintedDict.

        Args:
            *args, **kargs:
                Pass any parameters through to the dict() constructor.


        Calls the dict() constructor, then runs through the keys of the
        created dictionary and either uses the string type embedded in
        the keyname to generate the type hint (and remove the
        embedded string type from the keyname) or determines the likely
        type hint from the value of the dict element.
        """
        self._typehints = SortedDict()
        super().__init__(*args, **kargs)
        for key in list(self.keys()):  # Check through all the keys and see if they contain
            # type hints. If they do, move them to the
            # _typehint dict
            value = super().__getitem__(key)
            super().__delitem__(key)
            self[key] = value  # __Setitem__ has the logic to handle embedded type hints correctly

    @property
    def types(self) -> Dict:
        """Return the dictionary of value types."""
        return self._typehints

    def findtype(self, value: Any) -> str:
        """Determine the correct string type to return for common python classes.

        Args:
            value (any):
                The data value to determine the type hint for.

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
                    typ = f"{value.ndim}D Array ({self.findtype(z[0])})"
                else:
                    typ = t
                break
        return typ

    def __mungevalue(self, typ: str, value: Any) -> Any:
        """Based on a string type t, return value cast to an appropriate python class.

        Args:
            typ (string):
                is a string representing the type
            value (any):
                is the data value to be munged into the correct class
        Returns:
            Returns the munged data value

        Detail:
            The class has a series of precompiled regular
            expressions that will match type strings, a list of these has been
            constructed with instances of the matching Python classes. These
            are tested in turn and if the type string matches the constructor of
            the associated python class is called with value as its argument.
        """
        if typ == "Invalid Type":  # Short circuit here
            return repr(value)
        for regexp, valuetype in self.__tests:
            if regexp.search(typ) is None:
                continue
            if isinstance(valuetype, _evaluatable):
                try:
                    if isinstance(value, string_types):  # we've got a string already don't need repr
                        return literal_eval(value)
                    return literal_eval(repr(value))  # pylint: disable=eval-used
                except ValueError:  # Oops just keep string format
                    return str(value)
                except SyntaxError:
                    return ""
            if issubclass(valuetype, datetime.datetime):
                try:
                    return parser.parse(value)
                except ValueError:
                    try:
                        return literal_eval(value)
                    except ValueError:
                        return str(value)
            return valuetype(value)
        ret = str(value)
        try:
            return _parse_date(ret)
        except (ValueError, OverflowError):
            pass
        return ret

    def _get_name_(self, name: Union[str, RegExp]) -> Tuple[str, Optional[str]]:
        """Check a string name for an embedded type hint and strips it out.

        Args:
            name(string):
                String containing the name with possible type hint embedeed
        Returns:
            (name,typehint) (tuple):
                A tuple containing just the name of the mateadata and (if found
                the type hint string),
        """
        search = str(name)
        m = self.__regexGetType.search(search)
        if m is not None:
            return m.group(1), m.group(2)
        if not isinstance(name, string_types + int_types):
            return search, None
        return name, None

    def __getitem__(self, name: Union[str, RegExp]) -> Any:
        """Check whether its been given a typehint in the item name and deals with it appropriately.

        Args:
            name (string):
                metadata key to retrieve

        Returns:
            metadata value
        """
        key = name
        (name, typehint) = self._get_name_(name)
        name = self.__lookup__(name, True)
        value = [super(TypeHintedDict, self).__getitem__(nm) for nm in name]
        if typehint is not None:
            value = [self.__mungevalue(typehint, v) for v in value]
        if len(value) == 0:  # pylint: disable=len-as-condition
            raise KeyError(f"{key} is not a valid key even when interpreted as a sregular expression!")
        if len(value) == 1:
            return value[0]
        return {k: v for k, v in zip(name, value)}

    def __setitem__(self, name: Union[str, RegExp], value: Any) -> None:
        """Set an item in the dict, checking the key for an embedded type hint or inspecting the value as necessary.

        Arguments:
            name (string):
                The metadata keyname
            value (any):
                The value to store in the metadata string

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
                super().__setitem__(name, "")
                self._typehints[name] = "String"
            else:
                super().__setitem__(name, self.__mungevalue(typehint, value))
        else:
            if isinstance(value, string_types):
                value = string_to_type(value)
            self._typehints[name] = self.findtype(value)
            super().__setitem__(name, value)

    def __delitem__(self, name: Union[str, RegExp]) -> None:
        """Delete the specified key.

        Args:
            name (string): The keyname to be deleted
        """
        name = self._get_name_(name)[0]
        name = self.__lookup__(name)

        del self._typehints[name]
        super().__delitem__(name)

    def __repr__(self) -> str:
        """Create a text representation of the dictionary with type data."""
        ret = [f"{repr(key)}:{self.type(key)}:{repr(self[key])}" for key in sorted(self)]
        return "\n".join(ret)

    def copy(self) -> "TypeHintedDict":
        """Provide a copy method that is aware of the type hinting strings.

        This produces a flat dictionary with the type hint embedded in the key name.

        Returns:
            A copy of the current TypeHintedDict
        """
        cls = type(self)
        ret = cls()
        for k in self.keys():
            t = self._typehints[k]
            ret._typehints[k] = t
            super(TypeHintedDict, ret).__setitem__(k, copy.copy(self[k]))
        return ret

    def filter(self, name: Union[str, RegExp, Callable]) -> None:
        """Filter the dictionary keys by name.

        Reduce the metadata dictionary leaving only keys satisfied by name.

        Keyword Arguments:
            name(str or callable):
                either a str to match or a callable function that takes metadata key-value
                as an argument and returns True or False
        """
        rem = []
        for k in self.keys():
            if isinstance(name, string_types):
                if name not in k:
                    rem.append(k)
            elif hasattr(name, "__call__"):
                if not name(k):
                    rem.append(k)
            else:
                raise ValueError("name must be a string or a function")
        for k in rem:
            del self[k]

    def type(self, key: Union[str, RegExp, Sequence[Union[str, RegExp]]]) -> Union[str, List[str]]:
        """Return the typehint for the given k(s).

        This simply looks up the type hinting dictionary for each key it is given.

        Args:
            key (string or sequence of strings):
                Either a single string key or a iterable type containing keys

        Returns:
            The string type hint (or a list of string type hints)
        """
        if isinstance(key, string_types):
            return self._typehints[key]
        try:
            return [self._typehints[x] for x in key]
        except TypeError:
            return self._typehints[key]

    def export(self, key: Union[str, RegExp]) -> str:
        """Export a single metadata value to a string representation with type hint.

        In the ASCII based file format, the type hinted metadata is represented
        in the first column of a tab delimited text file as a series of lines
        with format keyname{typhint}=string_value.

        Args:
            key (string):
                The metadata key to export

        Returns:
            A string of the format : key{type hint} = value
        """
        if isinstance(self[key], string_types):  # avoid string within string problems and backslash overdrive
            ret = f"{key}{{{self.type(key)}}}={self[key]}"
        else:
            ret = f"{key}{{{self.type(key)}}}={repr(self[key])}"
        return ret

    def export_all(self) -> List[str]:
        """Return all the entries in the TypeHintedDict as a list of exported lines.

        Returns:
            (list of str): A list of exported strings

        Notes:
            The keys are returned in sorted order as a result of the underlying blist.sorteddict meothd.
        """
        return [self.export(x) for x in self]

    def import_all(self, lines: List[str]) -> None:
        """Read multiple lines of strings and tries to import keys from them.

        Args:
            lines(list of str):
                The lines of metadata values to import.
        """
        for line in lines:
            self.import_key(line)

    def import_key(self, line: str) -> None:
        """Import a single key from a string like key{type hint} = value.

        This is the inverse of the :py:meth:`TypeHintedDict.export` method.

        Args:
            line(str):
                he string line to be interpreted as a key-value pair.
        """
        parts = line.split("=")
        k = parts[0]
        v = "=".join(parts[1:])  # rejoin any = in the value string
        self[k] = v


class metadataObject(MutableMapping):
    """Represent some sort of object that has metadata stored in a :py:class:`Stoner.Core.TypeHintedDict` object.

    Attributes:
        metadata (TypeHintedDict):
            Dictionary of key-value metadata pairs. The dictionary tries to retain information about the type of data
            so as to aid import and export from CM group LabVIEW code.
    """

    def __new__(cls, *args):
        """Pre initialisation routines."""
        self = super().__new__(cls)
        self._public_attrs_real = dict()
        self._metadata = TypeHintedDict()
        return self

    def __init__(self, *args: Any, **kargs: Any) -> None:  # pylint: disable=unused-argument
        """Initialise the current metadata attribute."""
        metadata = kargs.pop("metadata", {})
        self._metadata = getattr(self, "_metadata", TypeHintedDict())
        self.metadata.update(metadata)
        super().__init__()

    @property
    def _public_attrs(self):
        """Return a dictionary of attributes setable by keyword argument with their types."""
        try:
            return self._public_attrs_real  # pylint: disable=no-member
        except AttributeError:
            self._public_attrs_real = dict()  # pylint: disable=attribute-defined-outside-init
            return self._public_attrs_real

    @_public_attrs.setter
    def _public_attrs(self, value):
        """Private property to update the list of public attributes."""
        self._public_attrs_real.update(dict(value))  # pylint: disable=no-member

    @property
    def metadata(self) -> Dict:
        """Read the metadata dictionary."""
        try:
            return self._metadata
        except AttributeError:  # Oops no metadata yet
            self._metadata = TypeHintedDict()
            return self._metadata

    @metadata.setter
    def metadata(self, value: Iterable) -> None:
        """Update the metadata object with type checking."""
        if not isinstance(value, TypeHintedDict) and isiterable(value):
            self._metadata = TypeHintedDict(value)
        elif isinstance(value, TypeHintedDict):
            self._metadata = value
        else:
            raise TypeError(f"metadata must be something that can be turned into a dictionary, not a {value}")

    def __getitem__(self, name: Union[str, RegExp]) -> Any:
        """Pass through to metadata dictionary."""
        return self.metadata[name]

    def __setitem__(self, name: Union[str, RegExp], value: Any) -> None:
        """Pass through to metadata dictionary."""
        self.metadata[name] = value

    def __delitem__(self, name: Union[str, RegExp]) -> None:
        """Pass through to metadata dictionary."""
        del self.metadata[name]

    def __eq__(self, other: Any) -> bool:
        """Implement am equality test for metadataObjects."""
        if not isinstance(other, metadataObject):
            return False
        if len(self) != len(other):
            return False
        ret = self.metadata ^ other.metadata
        return len(ret) == 0

    def __len__(self) -> int:
        """Pass through to metadata dictionary."""
        return len(self.metadata)

    def __iter__(self) -> Generator:
        """Pass through to metadata dictionary."""
        return self.metadata.__iter__()

    def keys(self) -> str:
        """Return the keys of the metadata dictionary."""
        for k in self.metadata.keys():
            yield k

    def items(self) -> Tuple[str, Any]:
        """Make sure we implement an items that doesn't just iterate over self."""
        for k, v in self.metadata.items():
            yield k, v

    def values(self) -> Any:
        """Return the values of the metadata dictionary."""
        for v in self.metadata.values():
            yield v

    def save(self, filename: Filename = None, **kargs: Any):
        """Stub method for a save function."""
        raise NotImplementedError("Save is not implemented in the base class.")

    def _load(self, filename: Filename, *args: Any, **kargs: Any) -> "metadataObject":
        """Stub method for a load function."""
        raise NotImplementedError("Save is not implemented in the base class.")


class SortedMultivalueDict(OrderedDict):
    """Implement a simple multivalued dictionary where the values are always sorted lists of elements."""

    @classmethod
    def _matching(cls, val: Tuple[int, str] | List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        match val:
            case (int(p), item):
                return [(p, item)]
            case [(int(p), item), *rest]:
                return sorted([(p, item)] + cls._matching(rest))
            case []:
                return []
            case _:
                raise TypeError("Can only add items that are a typle of int,value")

    def get_value_list(self, name):
        """Get the values stored in the dictionary under name."""
        return [item for _, item in self.get(name, [])]

    def __setitem__(self, name: Any, val: Union[List[Tuple[int, Any]], Tuple[int, Any]]) -> None:
        """Insert or replace a value and then sort the values."""
        values = self._matching(val)
        for p, value in values:  # pylint: disable=not-an-iterable
            for ix, (_, old_value) in enumerate(self.get(name, [])):
                if old_value == value:  # replacing existing value
                    self[name][ix] = (p, value)
                    break
            else:
                super().__setitem__(name, self.get(name, []) + [(p, value)])
        super().__setitem__(name, sorted(self[name], key=lambda item: (item[0], str(item[1]))))


if pd is not None and not hasattr(pd.DataFrame, "metadata"):  # Don;t double add metadata

    @pd.api.extensions.register_dataframe_accessor("metadata")
    class PandasMetadata(TypeHintedDict):
        """Add a typehintedDict to PandasDataFrames."""

        def __init__(self, pandas_obj):
            super().__init__()
            self._obj = pandas_obj
