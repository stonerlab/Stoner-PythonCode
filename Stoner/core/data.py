#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The main Data Class definition."""

import io
import copy
import pathlib
import warnings
import urllib

from collections.abc import MutableSequence, Mapping, Iterable
import inspect as _inspect_
from textwrap import TextWrapper
import csv

import numpy as np
from numpy import nan  # NOQA pylint: disable=unused-import
from numpy import ma

from ..compat import string_types, index_types
from ..tools import all_type, isiterable, get_option, make_Data
from ..tools.file import get_file_name_type, auto_load_classes, get_loader

from .exceptions import StonerLoadError, StonerSetasError
from .base import TypeHintedDict, metadataObject
from .array import DataArray
from .operators import DataFileOperatorsMixin
from .property import DataFilePropertyMixin
from .interfaces import DataFileInterfacesMixin
from .utils import copy_into, Tab_Delimited
from ..tools.file import file_dialog, FileManager, URL_SCHEMES, get_filename
from ..tools.tests import ClassTester

try:
    from tabulate import tabulate

    tabulate.PRESERVE_WHITESPACE = True
except ImportError:
    tabulate = None

try:
    import pandas as pd
except ImportError:
    pd = None


# Bring all the subclasses into memory (idnore unused imports warnings)
from .. import formats  # NOQA pylint: disable=W0611
from ..Analysis import AnalysisMixin
from ..analysis import columns
from ..analysis import features
from ..analysis import filtering
from ..analysis.fitting import functions as fitting
from . import methods
from ..plot import PlotMixin
from ..tools.decorators import class_modifier


@class_modifier([methods, fitting, columns, features, filtering], adaptor=None, no_long_names=True, overload=True)
class Data(
    DataFileInterfacesMixin,
    DataFileOperatorsMixin,
    DataFilePropertyMixin,
    metadataObject,
    MutableSequence,
    AnalysisMixin,
    PlotMixin,
):
    """Base class object that represents a matrix of data, associated metadata and column headers.

    Attributes:
        column_headers (list):
            list of strings of the column names of the data.
        data (2D numpy masked array):
            The attribute that stores the nuermical data for each Data. This is a :py:class:`DataArray` instance -
            which is itself a subclass of :py:class:`numpy.ma.MaskedArray`.
        title (string):
            The title of the measurement.
        filename (string):
            The current filename of the data if loaded from or already saved to disc. This is the default filename
            used by the :py:meth:`Stoner.Core.Data.load` and :py:meth:`Stoner.Core.Data.save`.
        header (string):
            A readonly property that returns a pretty formatted string giving the header of tabular representation.
        mask (array of booleans):
            Returns the current mask applied to the numerical data equivalent to self.data.mask.
        mime_type (list of str):
            The possible mime-types of data files represented by each matching filename pattern in
            :py:attr:`Datafile.pattern`.
        patterns (list):
            A list of filename extension glob patterns that matrches the expected filename patterns for a Data
            (*.txt and *.dat")
        priority (int):
            Used to indicathe order in which subclasses of :py:class:`Data` are tried when loading data. A higher
            number means a lower priority (!)
        setas (:py:class:`_stas`):
            Defines certain columns to contain X, Y, Z or errors in X,Y,Z data.
        shape (tuple of integers):
            Returns the shape of the data (rows,columns) - equivalent to self.data.shape.
        records (numpy record array):
            Returns the data in the form of a list of yuples where each tuple maps to the columns names.
        clone (Data):
            Creates a deep copy of the :py:class`Data` object.
        dict_records (array of dictionaries):
            View the data as an array or dictionaries where each dictionary represents one row with keys derived
            from column headers.
        dims (int):
            When data columns are set as x,y,z etc. returns the number of dimensions implied in the data set
        dtype (numpoy dtype):
            Returns the datatype stored in the :py:attr:`Data.data` attribute.
        T (:py:class:`DataArray`):
            Transposed version of the data.
        subclasses (list):
            Returns a list of all the subclasses of Data currently in memory, sorted by
            their py:attr:`Stoner.Core.Data.priority`. Each entry in the list consists of the
            string name of the subclass and the class object.
        xcol (int):
            If a column has been designated as containing *x* values, this will return the index of that column
        xerr (int):
            Similarly to :py:attr:`Data.xcol` but for the x-error value column.
        ycol (list of int):
            Similarly to :py:attr:`Data.xcol` but for the y value columns.
        yerr (list of int):
            Similarly to :py:attr:`Data.xcol` but for the y error value columns.
        zcol (list of int):
            Similarly to :py:attr:`Data.xcol` but for the z value columns.
        zerr (list of int):
            Similarly to :py:attr:`Data.xcol` but for the z error value columns.
        ucol (list of int):
            Similarly to :py:attr:`Data.xcol` but for the u (x-axis direction cosine) columns.
        vcol (list of int):
            Similarly to :py:attr:`Data.xcol` but for the v (y-axis direction cosine) columns.
        wcol (list of int):
            Similarly to :py:attr:`Data.xcol` but for the w (z-axis direction cosine) columns.
    """

    #: priority (int): is the load order for the class, smaller numbers are tried before larger numbers.
    #   .. note::
    #
    #      Subclasses with priority<=32 should make some positive identification that they have the right
    #      file type before attempting to read data.
    priority = 32

    #: pattern (list of str): A list of file extensions that might contain this type of file. Used to construct
    # the file load/save dialog boxes.
    _patterns = ["*.txt", "*.tdi"]  # Recognised filename patterns

    # mimetypes we match
    mime_type = ["text/plain"]

    _conv_string = np.vectorize(str)
    _conv_float = np.vectorize(float)

    # ====================================================================================
    ############################     Object Construction   ###############################
    # ====================================================================================

    def __new__(cls, *args, **kargs):
        """Prepare the basic Data instance before the mixins add their bits."""
        self = metadataObject.__new__(cls, *args)
        object.__setattr__(self, "debug", kargs.pop("debug", False))
        self._masks = [False]
        self._filename = None
        object.__setattr__(self, "_data", DataArray([]))
        self._baseclass = Data
        self._kargs = kargs
        return self

    def __init__(self, *args, **kargs):
        """Initialise the Data from arrays, dictionaries and filenames.

        Various forms are recognised:

        .. py:function:: Data('filename',<optional filetype>,<args>)
            :noindex:

            Creates the new Data object and then executes the :py:class:`Data`.load
            method to load data from the given *filename*.

        .. py:function:: Data(array)
            :noindex:

            Creates a new Data object and assigns the *array* to the
            :py:attr:`Data.data`  attribute.

        .. py:function:: Data(dictionary)
            :noindex:

            Creates the new Data object. If the dictionary keys are all strigns and the values are all
            numpy D arrays of equal length, then assumes the dictionary represents columns of data and the keys
            are the column titles, otherwise initialises the metadata with :parameter: dictionary.

        .. py:function:: Data(array,dictionary)
            :noindex:

            Creates the new Data object and does the combination of the
            previous two forms.


        .. py:function:: Data(Data)
            :noindex:

            Creates the new Data object and initialises all data from the
            existing :py:class:`Data` instance. This on the face of it does the same as
            the assignment operator, but is more useful when one or other of the
            Data objects is an instance of a sub - class of Data

        Args:
            args (positional arguments):
                Variable number of arguments that match one of the definitions above
            kargs (keyword Arguments):
                All keyword arguments that match public attributes are used to set those public attributes.
        """
        # init instance attributes
        super().__init__(**kargs)  # initialise self.metadata)
        self._public_attrs = {
            "data": np.ndarray,
            "filetype": str,
            "setas": (string_types, list, dict),
            "column_headers": list,
            "metadata": TypeHintedDict,
            "debug": bool,
            "filename": string_types,
            "mask": (np.ndarray, bool),
        }
        self._repr_limits = (256, 6)
        handler = [lambda *args, **kargs: None, self._init_single, self._init_double, self._init_many][
            min(len(args), 3)
        ]
        self.mask = False
        self.data._setas._get_cols()
        handler(*args, **kargs)
        try:
            kargs = self._kargs
            delattr(self, "_kargs")
        except AttributeError:
            pass
        self.metadata["Stoner.class"] = type(self).__name__
        if kargs:  # set public attributes from keywords
            to_go = []
            for k in kargs:
                if k in self._public_attrs:
                    if isinstance(kargs[k], self._public_attrs[k]):
                        self.__setattr__(k, kargs[k])
                    else:
                        self._raise_type_error(k)
                        to_go.append(k)
                else:
                    raise AttributeError(f"{k} is not an allowed attribute of {self._public_attrs}")
                    # self._public_attrs[k]=type(kargs[k])
                    # self.__setattr__(k, kargs[k])
            for k in to_go:
                del kargs[k]
        if self.debug:
            print("Done Data init")

    # ============================================================================================
    ############################   Constructor Methods ###########################################
    # ============================================================================================

    def _init_single(self, *args, **kargs):
        """Handle constructor with 1 argument - called from __init__."""
        test = ClassTester(ImageFile="Stoner.Image.core.ImageFile")
        match args[0]:
            case str() | bool() | pathlib.Path() | bytes() | io.IOBase():
                self._init_load(args[0], **kargs)
            case Data():
                self._init_datafile(args[0], **kargs)
            case pd.DataFrame():
                self._init_pandas(args[0], **kargs)
            case test.ImageFile():
                self._init_imagefile(args[0], **kargs)
            case np.ndarray():
                self._init_array(args[0], **kargs)
            case Mapping():
                self._init_dict(args[0], **kargs)
            case Iterable():
                self._init_list(args[0], **kargs)
            case _:
                raise TypeError(f"No constructor for {type(args[0])}")
        self.data._setas.cols.update(self.setas._get_cols())

    def _init_double(self, *args, **kargs):
        """Two argument constructors handled here. Called form __init__."""
        (arg0, arg1) = args
        match args:
            case (arg0, dict() as arg1):
                self._init_single(arg0, **kargs)
                self._init_single(arg1, **kargs)
            case (arg0, Iterable() as arg1) if all_type(arg1, str):
                self._init_single(arg0, **kargs)
                self._init_single(arg1, **kargs)
            case (np.ndarray() as arg0, np.ndarray() as arg1) if arg0.ndim == 1 and arg1.ndim == 1:
                self._init_many(*args, **kargs)
            case _:
                raise TypeError(f"Unable to decide how to initialise {type(args)}")

    def _init_many(self, *args, **kargs):
        """Handle more than two arguments to the constructor - called from init."""
        for a in args:
            if not (isinstance(a, np.ndarray) and a.ndim == 1):
                copy_into(self.__class__.load(a, **kargs), self)
                break
        else:
            self.data = np.column_stack(args)

    def _init_array(self, arg, **kargs):  # pylint: disable=unused-argument
        """Initialise from a single numpy array."""
        # numpy.array - set data
        if np.issubdtype(arg.dtype, np.number):
            self.data = DataArray(np.atleast_2d(arg), setas=self.data._setas)
            self.column_headers = [f"Column_{x}" for x in range(np.shape(arg)[1])]
        elif isinstance(arg[0], dict):
            for row in arg:
                self += row

    def _init_datafile(self, arg, **kargs):  # pylint: disable=unused-argument
        """Initialise from datafile."""
        for a in arg.__dict__:
            if not callable(a) and a != "_baseclass":
                super().__setattr__(a, copy.copy(arg.__getattribute__(a)))
        self.metadata = arg.metadata.copy()
        self.data = DataArray(arg.data, setas=arg.setas.clone)
        self.data.setas = arg.setas.clone

    def _init_dict(self, arg, **kargs):  # pylint: disable=unused-argument
        """Initialise from dictionary."""
        if (
            all_type(arg.keys(), string_types)
            and all_type(arg.values(), np.ndarray)
            and np.all([len(arg[k].shape) == 1 and np.all(len(arg[k]) == len(list(arg.values())[0])) for k in arg])
        ):
            self.data = np.column_stack(tuple(arg.values()))
            self.column_headers = list(arg.keys())
        else:
            self.metadata = arg.copy()

    def _init_imagefile(self, arg, **kargs):  # pylint: disable=unused-argument
        """Initialise from an ImageFile."""
        x = arg.get("x_vector", np.arange(arg.shape[1]))
        y = arg.get("y_vector", np.arange(arg.shape[0]))
        x, y = np.meshgrid(x, y)
        z = arg.image

        self.data = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        self.metadata = copy.deepcopy(arg.metadata)
        self.column_headers = ["X", "Y", "Image Intensity"]
        self.setas = "xyz"

    def _init_pandas(self, arg, **kargs):  # pylint: disable=unused-argument
        """Initialise from a pandas dataframe."""
        self.data = arg.values
        ch = []
        for ix, col in enumerate(arg):
            if isinstance(col, string_types):
                ch.append(col)
            elif isiterable(col):
                for ch_i in col:
                    if isinstance(ch_i, string_types):
                        ch.append(ch_i)
                        break
                else:
                    ch.append(f"Column {ix}")
            else:
                ch.append(f"Column {ix}:{col}")
        self.column_headers = ch
        self.metadata.update(arg.metadata)
        if isinstance(arg.columns, pd.MultiIndex) and len(arg.columns.levels) > 1:
            for label in arg.columns.get_level_values(1):
                if label not in list("xyzdefuvw."):
                    break
            else:
                self.setas = list(arg.columns.get_level_values(1))

    def _init_load(self, arg, **kargs):
        """Load data from a file-like source.

        arg(str, PurePath, IOBase, bool):
            If arg is a str, PaurePath, ioBase then open the file like object and read. If arg is bool and False,
            provide a dialog box instead.
        """
        if isinstance(arg, bool):
            if arg:
                raise ValueError("Cannot construct a Data with a single argument of True")
        elif isinstance(arg, pathlib.PurePath):
            arg = str(arg)
        copy_into(self.__class__.load(filename=arg, **kargs), self)

    def _init_list(self, arg, **kargs):
        """Initialise from a list or other ioterable."""
        if all_type(arg, string_types):
            self.column_headers = list(arg)
        elif all_type(arg, np.ndarray):
            self._init_many(*arg, **kargs)
        else:
            raise TypeError(f"Unable to construct Data from a {type(arg)}")

    # ============================================================================================
    ############################   Special Methods ###############################################
    # ============================================================================================

    def __call__(self, *args, **kargs):
        """Clone the Data, but allowing additional arguments to modify the new clone.

        Args:
            *args (tuple):
                Positional arguments to pass through to the new clone.
            **kargs (dict):
                Keyword arguments to pass through to the new clone.

        Raises:
            TypeError: If a keyword argument doesn't match an attribute.

        Returns:
            new_d (Data):
                Modified clone of the current object.
        """
        new_d = self.clone
        handler = [lambda *args, **kargs: None, new_d._init_single, new_d._init_double, new_d._init_many][
            min(len(args), 2)
        ]
        handler(*args, **kargs)
        if kargs:  # set public attributes from keywords
            myattrs = new_d._public_attrs
            for k in kargs:
                if k in myattrs:
                    if isinstance(kargs[k], myattrs[k]):
                        new_d.__setattr__(k, kargs[k])
                    else:
                        if isinstance(myattrs[k], tuple):
                            typ = "one of " + ",".join([str(type(t)) for t in myattrs[k]])
                        else:
                            typ = f"a {type(myattrs[k])}"
                        raise TypeError(f"{k} should be {typ} not a {type(kargs[k])}")

        return new_d

    def __deepcopy__(self, memo):
        """Provide support for copy.deepcopy to work."""
        cls = type(self)
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            try:
                setattr(result, k, copy.deepcopy(v, memo))
            except (TypeError, ValueError, RecursionError):
                setattr(result, k, copy.copy(v))
        return result

    def __dir__(self):
        """Returns the attributes of the current object.

        Augmenting the keys of self.__dict__ with the attributes that __getattr__ will handle."""
        attr = dir(type(self))
        col_check = {"xcol": "x", "xerr": "d", "ycol": "y", "yerr": "e", "zcol": "z", "zerr": "f"}
        if not self.setas.empty:
            for k in col_check:
                if k.startswith("x"):
                    if k in self._data._setas.cols and self._data._setas.cols[k] is not None:
                        attr.append(col_check[k])
                else:
                    if k in self._data._setas.cols and self._data._setas.cols[k]:
                        attr.append(col_check[k])
        return sorted(set(attr))

    def __getattr__(self, name):
        """Handle some special pseudo attributes that map to the setas columns.

        Args:
            name (string):
                The name of the attribute to be returned.

        Returns:
            Various:
                the Data object in various forms

        Supported attributes:
            - records:
                return the Data data as a numpy structured
                array - i.e. rows of elements whose keys are column headings
                - clone:
                    returns a deep copy of the current Data instance

        Otherwise the name parameter is tried as an argument to :py:meth:`Data.column` and the resultant column
        is returned. If Data.column raises a KeyError this is remapped as an AttributeError.
        """
        setas_cols = ("x", "y", "z", "d", "e", "f", "u", "v", "w", "r", "q", "p")
        if name != "debug" and self.debug:
            print(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            ret = self.__dict__.get(name, type(self).__dict__.get(name, None))
            if ret is not None:
                return ret
        if name in setas_cols:
            ret = self._getattr_col(name)
            if ret is not None:
                return ret
        if name in self.setas.cols:
            ret = self.setas.cols[name]
            if ret is not None and ret != []:
                return ret
        try:
            col = self._data._setas.find_col(name)
            return self.column(col)
        except (KeyError, IndexError):
            pass
        if name in setas_cols:  # Probably tried to use a setas col when it wasn't defined
            raise StonerSetasError(
                f"Tried accessing a {name} column, but setas is not defined and {name} is not a column name either"
            )
        raise AttributeError(f"{name} is not an attribute of Data nor a column name")

    #    def __reduce_ex__(self, p):
    #        """Machinery used for deepcopy."""
    #        cls=type(self)
    #        return (cls, (), self.__getstate__())

    def __repr__(self):
        """Output the :py:class:`Data` object in TDI format.

        This allows one to print any :py:class:`Data` to a stream based
        object andgenerate a reasonable textual representation of the data.shape

        Returns:
            self in a textual format.
        """
        if get_option("short_repr") or get_option("short_data_repr"):
            return self._repr_short_()
        try:
            return self._repr_table_()
        except (ImportError, ValueError, TypeError):
            return self.__repr_core__(256)

    def __setattr__(self, name, value):
        """Handle attempts to set attributes not covered with class attribute variables.

        Args:
            name (str):
                Name of attribute to set. Details of possible attributes below:
                -   mask Passes through to the mask attribute of self.data (which is a numpy masked array).
                    Also handles the case where you pass a callable object to nask where we pass each row to the
                    function and use the return reult as the mask
                -   data Ensures that the :py:attr:`data` attribute is always a :py:class:`numpy.ma.maskedarray`
        """
        if hasattr(type(self), name) and isinstance(getattr(type(self), name), property):
            super().__setattr__(name, value)
        elif len(name) == 1 and name in "xyzuvwdef" and self.setas[name]:
            self._setattr_col(name, value)
        else:
            super().__setattr__(name, value)

    def __str__(self):
        """Provide an implementation for str(Data) that does not shorten the output."""
        return self.__repr_core__(False)

    # ============================================================================================
    ############################ Private Methods #################################################
    # ============================================================================================

    def _col_args(self, *args, **kargs):
        """Create an object which has keys  based either on arguments or setas attribute."""
        return self.data._col_args(*args, **kargs)  # Now just pass through to DataArray

    def _getattr_col(self, name):
        """Get a column using the setas attribute."""
        try:
            return self._data.__getattr__(name)
        except StonerSetasError:
            return None

    def _interesting_cols(self, cols):
        """Workout which columns the user might be interested in in the basis of the setas.

        ArgsL
            cols (float):
                Maximum Number of columns to display

        Returns
            list(ints):
                The indices of interesting columns with breaks in runs indicated by -1
        """
        c = self.shape[1]
        if c > cols:
            interesting = []
            last = -1
            for ix, typ in enumerate(self.setas):
                if last != -1 and last != ix - 1:
                    interesting.append(-1)
                    last = -1
                if typ != ".":
                    interesting.append(ix)
                    last = ix
            if interesting and interesting[-1] == -1:
                interesting = interesting[:-1]
            if interesting:
                c_start = max(interesting) + 1
            else:
                c_start = 0
            interesting.extend(range(c_start, c))
            if len(interesting) < cols:
                cols = len(interesting)
            if interesting[cols - 3] != -1:
                interesting[cols - 2] = -1
            else:
                interesting[cols - 2] = c - 2
            interesting[cols - 1] = c - 1
            interesting = interesting[:cols]
            c = cols
        else:
            interesting = list(range(c))

        col_assignments = []
        for i in interesting:
            if i != -1:
                if self.setas[i] != ".":
                    col_assignments.append(f"{i} ({self.setas[i]})")
                else:
                    col_assignments.append(f"{i}")
            else:
                col_assignments.append("")
        return interesting, col_assignments, cols

    def _load(self, filename, *args, **kargs):
        """Actually load the data from disc assuming a .tdi file format.

        Args:
            filename (str):
                Path to filename to be loaded. If None or False, a dialog bax is raised to ask for the filename.

        Returns:
            Data:
                A copy of the newly loaded :py:class`Data` object.

        Exceptions:
            StonerLoadError:
                Raised if the first row does not start with 'TDI Format 1.5' or 'TDI Format=1.0'.

        Note:
            The *_load* methods shouldbe overridden in each child class to handle the process of loading data from
            disc. If they encounter unexpected data, then they should raise StonerLoadError to signal this, so that
            the loading class can try a different sub-class instead.
        """
        if filename is None or not filename:
            self.get_filename("r")
        else:
            self.filename = filename
        with FileManager(self.filename, "r", encoding="utf-8", errors="ignore") as datafile:
            line = datafile.readline()
            if line.startswith("TDI Format 1.5"):
                fmt = 1.5
            elif line.startswith("TDI Format=Text 1.0"):
                fmt = 1.0
            else:
                raise StonerLoadError("Not a TDI File")

            datafile.seek(0)
            reader = csv.reader(datafile, dialect=Tab_Delimited())
            cols = 0
            max_rows = 0
            for ix, metadata in enumerate(reader):
                if ix == 0:
                    row = metadata
                    continue
                if len(metadata) < 1:
                    continue
                if cols == 0:
                    cols = len(metadata)
                if len(metadata) > 1:
                    max_rows = ix + 1
                if "=" in metadata[0]:
                    self.metadata.import_key(metadata[0])
            col_headers_tmp = [x.strip() for x in row[1:]]
            with warnings.catch_warnings():
                datafile.seek(0)
                warnings.filterwarnings("ignore", "Some errors were detected !")
                data = np.genfromtxt(
                    datafile,
                    skip_header=1,
                    usemask=True,
                    delimiter="\t",
                    usecols=range(1, cols),
                    invalid_raise=False,
                    comments="\0",
                    missing_values=[""],
                    filling_values=[np.nan],
                    max_rows=max_rows,
                )
        if data.ndim < 2:
            data = np.ma.atleast_2d(data)
        retain = np.all(np.isnan(data), axis=1)
        self.data = DataArray(data[~retain])
        self["TDI Format"] = fmt
        if self.data.ndim == 2 and self.data.shape[1] > 0:
            self.column_headers = col_headers_tmp
        return self

    def __repr_core__(self, shorten=1000):
        """Actuall do the repr work, but allow for a shorten parameter to save printing big files out to disc."""
        outp = "TDI Format 1.5\t" + "\t".join(self.column_headers) + "\n"
        m = len(self.metadata)
        self.data = np.atleast_2d(self.data)
        r = np.shape(self.data)[0]
        md = self.metadata.export_all()
        for x in range(min(r, m)):
            if self.data.ndim != 2 or self.shape[1] == 1:
                outp += f"{md[x]}\t{self.data[x]}\n"
            else:
                outp = outp + md[x] + "\t" + "\t".join([str(y) for y in self.data[x].filled()]) + "\n"
        if m > r:  # More metadata
            for x in range(r, m):
                outp = outp + md[x] + "\n"
        elif r > m:  # More data than metadata
            if shorten is not None and shorten and r - m > shorten:
                for x in range(m, m + shorten - 100):
                    if self.data.ndim != 2 or self.shape[1] == 1:
                        outp += "\t" + f"\t{self.data[x]}\n"
                    else:
                        outp += "\t" + "\t".join([str(y) for y in self.data[x].filled()]) + "\n"
                outp += f"... {r - m - shorten + 100} lines skipped...\n"
                for x in range(-100, -1):
                    if self.data.ndim != 2 or self.shape[1] == 1:
                        outp += f"\t\t{self.data[x]}\n"
                    else:
                        outp += "\t" + "\t".join([str(y) for y in self.data[x].filled()]) + "\n"
            else:
                for x in range(m, r):
                    if self.data.ndim != 2 or self.shape[1] == 1:
                        outp += f"\t\t{self.data[x]}\n"
                    else:
                        outp = outp + "\t" + "\t".join([str(y) for y in self.data[x].filled()]) + "\n"
        return outp

    def _repr_html_private(self):
        """Version of repr_core that does and html output."""
        return self._repr_table_("html")

    def _repr_short_(self):
        ret = (
            f"{self.filename}({type(self)}) of shape {self.shape} ({''.join(self.setas)})"
            + f" and {len(self.metadata)} items of metadata"
        )
        return ret

    def _repr_table_(self, fmt="rst"):
        """Convert the Data to a 2D array and then feed to tabulate."""
        if tabulate is None:
            raise ImportError("No tabulate.")
        lb = "<br/>" if fmt == "html" else "\n"
        rows, cols = self._repr_limits
        r, c = self.shape
        interesting, col_assignments, cols = self._interesting_cols(cols)
        c = min(c, cols)
        if len(interesting) > 0:
            c_w = max([len(self.column_headers[x]) for x in interesting if x > -1])
        else:
            c_w = 0
        wrapper = TextWrapper(subsequent_indent="\t", width=max(20, max(20, (80 - c_w * c))))
        if r > rows:
            shorten = [True, False]
            r = rows + rows % 2
        else:
            shorten = [False, False]

        shorten[1] = c > cols
        r = max(len(self.metadata), r)

        outp = np.zeros((r + 1, c + 1), dtype=object)
        outp[:, :] = "..."
        ch = [self.column_headers[ix] if ix >= 0 else "...." for ix in interesting]

        for ix, (h, i) in enumerate(zip(ch, col_assignments)):
            spaces1 = " " * ((c_w - len(h)) // 2)
            spaces2 = " " * ((c_w - len(i)) // 2)
            ch[ix] = f"{spaces1}{h}{lb}{spaces2}{i}"
            if self.debug:
                print(len(spaces1), len(spaces2))
        outp[0, 1:] = ch
        outp[1, 1:] = col_assignments
        outp[0, 0] = f"TDI Format 1.5{lb}index"
        i = 1
        for md in self.metadata.export_all():
            md = md.replace("=", "= ")
            for line in wrapper.wrap(md):
                if i >= outp.shape[0]:  # pylint: disable=E1136
                    outp = np.append(outp, [[""] * outp.shape[1]], axis=0)  # pylint: disable=E1136
                outp[i, 0] = line
                i += 1
        for ic, c in enumerate(interesting):
            if c >= 0:
                if shorten[0]:
                    col_out = np.where(self.mask[: r // 2 - 1, c], "#####", self.data[: r // 2 - 1, c].astype(str))
                    outp[1 : r // 2, ic + 1] = col_out
                    col_out = np.where(self.mask[-r // 2 :, c], "#####", self.data[-r // 2 :, c].astype(str))
                    outp[r // 2 + 1 : r + 1, ic + 1] = col_out
                else:
                    col_out = np.where(self.mask[:, c], "#####", self.data[:, c].astype(str))
                    outp[1 : len(self.data) + 1, ic + 1] = col_out
        return tabulate(outp[1:], outp[0], tablefmt=fmt, numalign="decimal", stralign="left")

    def _setattr_col(self, name, value):
        """Attempt to either assign data columns if set up, or setas setting.

        Args:
            name (length 1 string):
                Column type to work with (one of x,y,z,u,v,w,d,e or f)
            value (nd array or column index):
                If an ndarray and the column type corresponding to *name* is set up, then overwrite the column(s)
                of data with this new data. If an index type, then set the corresponding setas assignment to
                these columns.
        """
        if isinstance(value, np.ndarray):
            value = np.atleast_2d(value)
            if value.shape[0] == self.data.shape[0]:
                pass
            elif value.shape[1] == self.data.shape[0]:
                value = value.T
            else:
                raise RuntimeError("Value to be assigned to data columns is the wrong shape!")
            for i, ix in enumerate(self.find_col(self.setas[name], force_list=True)):
                self.data[:, ix] = value[:, i]
        elif isinstance(value, index_types):
            self._set_setas({name: value})

    def _set_mask(self, func, invert=False, cumulative=False, col=0):
        """Apply func to each row in self.data and uses the result to set the mask for the row.

        Args:
            func (callable):
                A Callable object of the form lambda x:True where x is a row of data (numpy
            invert (bool):
                Optionally invert te reult of the func test so that it unmasks data instead
            cumulative (bool):
                if tru, then an unmask value doesn't unmask the data, it just leaves it as it is.
        """
        i = -1
        args = len(_inspect_.getargs(func.__code__)[0])
        for r in self.rows():
            i += 1
            r.mask = False
            if args == 2:
                t = func(r[col], r)
            else:
                t = func(r)
            if isinstance(t, (bool, np.bool_)):
                if t ^ invert:
                    self.data[i] = ma.masked
                elif not cumulative:
                    self.data[i] = self._data.data[i]
            else:
                for j in range(min(len(t), np.shape(self.data)[1])):
                    if t[j] ^ invert:
                        self.data[i, j] = ma.masked
                    elif not cumulative:
                        self.data[i, j] = self.data.data[i, j]

    def _push_mask(self, mask=None):
        """Copy the current data mask to a temporary store and replace it with a new mask if supplied.

        Args:
            mask (:py:class:numpy.array of bool or bool or None):
                The new data mask to apply (defaults to None = unmask the data

        Returns:
            Nothing
        """
        self._masks.append(self.mask)
        if mask is None:
            self.data.mask = False
        else:
            self.mask = mask

    def _pop_mask(self):
        """Replace the mask on the data with the last one stored by _push_mask().

        Returns:
            Nothing
        """
        self.mask = False
        self.mask = self._masks.pop()  # pylint: disable=E0203
        if not self._masks:  # pylint: disable=E0203
            self._masks = [False]

    def _raise_type_error(self, k):
        """Raise a type error when setting an attribute k."""
        if isinstance(self._public_attrs[k], tuple):
            typ = "one of " + ",".join([str(type(t)) for t in self._public_attrs[k]])
        else:
            typ = f"a {type(self._public_attrs[k])}"
        raise TypeError(f"{k} should be {typ}")

    @classmethod
    def load(cls, *args, **kargs):
        """Create a new :py:class:`Data` from a file on disc guessing a better subclass if necessary.

        Args:
            filename (string or None):
                path to file to load

        Keyword Arguments:
            auto_load (bool):
                If True (default) then the load routine tries all the subclasses of :py:class:`Data` in turn to
                load the file
            filetype (:py:class:`Data`, str):
                If not none then tries using filetype as the loader.
            loaded_class (bool):
                If True, the return object is kept as the class that managed to load it, otherwise it is copied into a
                :py:class:`Stoner.Data` object. (Default False)

        Returns:
            (Data):
                A new instance of :py:class:`Stoner.Data` or a s subclass of :py:class:`Stoner.Data` if
                *loaded_class* is True.
        Note:
            If *filetype* is a string, then it is first tried as an exact match to a subclass name, otherwise it
            is used as a partial match and the first class in priority order is that matches is used.

            Some subclasses can be found in the :py:mod:`Stoner.formats` package.

            Each subclass is scanned in turn for a class attribute priority which governs the order in which they
            are tried. Subclasses which can make an early positive determination that a file has the correct format
            can have higher priority levels. Classes should return a suitable exception if they fail to load the file.

            If no class can load a file successfully then a StonerUnrecognisedFormat exception is raised.
        """
        filename, args, kargs = get_filename(args, kargs)
        debug = kargs.pop("debug", False)
        filetype = kargs.pop("filetype", None)
        auto_load = kargs.pop("auto_load", filetype is None)
        loaded_class = kargs.pop("loaded_class", False)
        if (
            isinstance(filename, (str, pathlib.Path))
            and urllib.parse.urlparse(str(filename)).scheme not in URL_SCHEMES
        ):
            filename, filetype = get_file_name_type(filename, filetype, Data)
        if filename is None or not filename:
            filename = file_dialog("r", filename, "Data", Data)
        elif not auto_load and not filetype:
            raise StonerLoadError("Cannot read data from non-path like filenames !")
        if auto_load:  # We're going to try every subclass we canA
            ret = auto_load_classes(filename, "Data", debug=debug, args=args, kargs=kargs)
            if not isinstance(ret, Data):  # autoload returned something that wasn't a data file!
                return ret
        else:
            loader = get_loader(filetype)
            try:
                ret = loader(make_Data(), filename, *args, **kargs)
            except StonerLoadError as err:
                raise ValueError(f"Unable to load {filename}") from err

        for k, i in kargs.items():
            if not callable(getattr(ret, k, lambda x: False)):
                setattr(ret, k, i)
        ret._kargs = kargs
        filetype = ret.__class__.__name__
        if loaded_class:
            datafile = ret
        else:
            datafile = make_Data()
            datafile._public_attrs.update(ret._public_attrs)
            copy_into(ret, datafile)
            datafile.filetype = filetype
        return datafile
