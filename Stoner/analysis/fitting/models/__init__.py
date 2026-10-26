"""Sub package of various built-in models for the Stoner package."""

__all__ = [
    "generic",
    "thermal",
    "magnetism",
    "tunnelling",
    "e_transport",
    "superconductivity",
    "_get_model_",
    "make_model",
    "cfg_data_from_ini",
    "cfg_model_from_ini",
]

from collections.abc import Mapping
from configparser import ConfigParser as SafeConfigParser
from functools import wraps
from importlib import import_module
from io import IOBase

import numpy as np
from lmfit import Model
from lmfit.models import update_param_vals

from ....compat import string_types
from ....tools import make_Data
from . import e_transport, generic, magnetism, superconductivity, thermal, tunnelling


def _get_model_(model):
    """Utility meothd to manage creating an lmfit.Model.

    Args:
        model (str, callable, Model):
            The model to be setup.

    Returns:
        An llmfit.Model instance

    model can be of several different types that determine what to do:
        -   A string. In which ase it should be a fully qualified name of a function or class to be imported.
            The part after the final period will be assumed to be the name and the remainder the module to be
            imported.
        -   A callable object. In this case the callable will be passed to the constructor of Model and a fresh
            Model instance is constructed
        -   A subclass of lmfit.Model - in which case it is instantiated.
        -   A Model instance - in which case no further action is necessary.
    """
    if isinstance(model, string_types):  # model is a string, so we;ll try importing it now
        parts = model.split(".")
        model = parts[-1]
        module = ".".join(parts[:-1])
        module = import_module(module)
        model = getattr(module, model)
    if type(model).__name__ == "type" and issubclass(
        model, Model
    ):  # ok perhaps we've got a model class rather than an instance
        model = model()
    if not isinstance(model, Model) and callable(model):  # Ok, wrap the callable in a model
        model = Model(model)
    if not isinstance(model, Model):
        raise TypeError(f"model {model.__name__} is not an instance of llmfit.Model")
    return model


def make_model(model_func):
    """A decorator that turns a function into an lmfit model.

    Args:
        model_func (function):
            The function to be wrapped in an lmfit.Model object

    Notes:
        The function being wrapped into the model should have the form::

            def model_func(x_data,*parameters):
                ....

        (i.e. similar to what :py:func:`scipy.optimize.curve_fit` expects). The resulting
        class is a sub-class of :py:class:`lmfit.Model` but also adds a class method
        :py:meth:`_ModelDecorator.guesser` which can be used as a decorator to convert another function into
        a :py:meth:`lmfit.Model.guess` method. If using this decorator, the function that does the guessing should
        take the form::

            def guesser_function(y_data, x=x_data, **kwargs):
                return (param_1, param_2, ..., param_n)

        Similarly, the class provides a :py:meth:`_ModelDecorator.hinter` decorator which can be used to mark a
        function as something that can generate parameter hints for the model. In this case the function should
        take the form::

            def hinter(**kwargs):
                return {"param_1":{"max":max_val,"min":min_value,"value":start_value},"param_2":{.....}}

        Finally the new model_func class can be instantiated or just passed to :py:meth:`Data.lmfit` etc. directly.
    """

    class _ModelDecorator(Model):  # pylint: disable=abstract-method
        __doc__ = model_func.__doc__

        def __init__(self, *args, **kwargs):
            super().__init__(model_func, *args, **kwargs)
            if hasattr(self, "_limits"):
                for param, limit in self._limits().items():
                    self.set_param_hint(param, **limit)
            self.__name__ = self.func.__name__

        def guess(self, data, x=None, **kwargs):  # pylint: disable=unused-argument
            """Make a default parameter guess method.

            Just guesses 1.0 for everything like :py:func:`scipy.optimize.curve_fit` does.
            """
            return np.ones_like(self.param_names)

        @classmethod
        def hinter(cls, func):
            """Use the given function to determine the parameter hints.

            Args:
                func (callable):
                    A function that returns a dictionary of dictionaries

            Returns:
                The wrapped hinter function.

            Notes:
                This decorator will modify the instance attributes so that the instance has a method to generate
                parameter hints.

                func should only take keyword arguments as by default it will be called with no arguments during
                model initialisation.
            """

            @wraps(func)
            def _limits_proxy(self, **kwargs):
                limits = func(**kwargs)
                for param in limits:
                    if param not in self.param_names:
                        raise RuntimeError(f"Unrecognised parameter in hinter function: {param}")
                    if not isinstance(limits[param], Mapping):
                        raise RuntimeError(f"Parameter hint for {param} was not a mapping")
                return limits

            cls._limits = _limits_proxy
            return _limits_proxy

        @classmethod
        def guesser(cls, func):
            """Use the given function as the guess method.

            Args:
                func (callable):
                    A function that guesses the parameter values

            Returns:
                The wrapped guess function.

            Notes:
                This decorator will modify the instance attributes so that the instance has a working guess method.

                func should take at least one positional argument, being the y-data values used to guess parameters.
                It should return a list, tuple of guesses parameter values with one entry for each parameter in
                the model.
            """

            @wraps(func)
            def guess_proxy(self, *args, **kwargs):
                """A magic proxy call around a function to guess initial parameters."""
                guesses = func(*args, **kwargs)
                pars = dict(zip(self.param_names, guesses))
                pars = self.make_params(**pars)
                return update_param_vals(pars, self.prefix, **kwargs)

            cls.guess = guess_proxy
            return guess_proxy

    return _ModelDecorator


def _ini_csv_value(name, value, legacy_rows):
    """Convert an INI CSV option to its native loader value."""
    if name in {"header_line", "data_line"}:
        if name == "header_line" and value.lower() == "none":
            return None
        try:
            row = int(value)
        except ValueError as err:
            raise ValueError(f"[Data] {name} must be an integer row number.") from err
        minimum = 1 if legacy_rows else 0
        if row < minimum:
            convention = "one-based" if legacy_rows else "zero-based"
            raise ValueError(f"[Data] {name} must be at least {minimum} ({convention} row numbering).")
        return row - minimum
    if value.lower() == "none":
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if value == r"\t":
        value = "\t"
    if len(value) != 1:
        raise ValueError(f"[Data] {name} must be one character, \\t, or None.")
    return value


def _ini_csv_options(config, overrides):
    """Resolve CSV layout options, preserving historical names and row numbering."""
    legacy_rows = config.has_option("Data", "start_line") and not config.has_option("Data", "data_line")
    aliases = {"data_line": "start_line", "header_delim": "separator", "data_delim": "separator"}
    options = {}
    for name in ("header_line", "data_line", "header_delim", "data_delim"):
        if name in overrides:
            options[name] = overrides[name]
            continue
        source = name
        if not config.has_option("Data", source):
            source = aliases.get(name, name)
        if config.has_option("Data", source):
            value = config.get("Data", source, raw=name.endswith("_delim"))
            options[name] = _ini_csv_value(name, value, legacy_rows)
    return options


def cfg_data_from_ini(inifile, filename=None, **kwargs):
    r"""Load a measurement and assign its fitting columns from an INI file.

    Args:
        inifile (str or file):
            Path to the INI file, or an open text stream. Seekable streams are
            rewound before reading and remain open afterwards.

    Keyword Arguments:
        filename (str, path-like, Boolean or None):
            Measurement to load. None uses the [Data] filename entry; if neither
            is supplied, the existing file-selection behaviour is used.
        **kwargs (dict):
            For CSVFile, header_line, data_line, header_delim and data_delim
            override INI settings using the native loader's zero-based numbering.
            Other keywords are passed to the Data constructor.

    Returns:
        Stoner.core.data.Data:
            A new measurement with x, y and optional uncertainty roles assigned.

    Raises:
        RuntimeError:
            If the INI file lacks a [Data] section.
        ValueError:
            If an INI CSV row number or delimiter is invalid, or loading fails.

    Notes:
        The [Data] section accepts type (registered loader name), filename, x,
        y and optional e (column headers or zero-based positions). Coordinate
        defaults are x=0 and y=1, with no uncertainty column.

        CSVFile accepts header_line (None for no header), data_line and separate
        header_delim/data_delim entries. Row numbers are zero-based. Delimiters
        are literal single characters; None requests detection, ``\t`` denotes a
        tab, and quoted delimiters allow spaces. Omitted entries retain loader
        defaults: header row 0, data row 1 and comma delimiters.

        Historical separator supplies both delimiters unless each is explicitly
        configured. When start_line is present without data_line, start_line
        and any header_line use historical one-based numbering. data_line takes
        precedence over start_line and selects zero-based INI row numbering.
        Explicit loader keywords override the corresponding normalised INI
        value. CSV layout entries are ignored for other loader types.
    """
    if SafeConfigParser is None:
        raise RuntimeError("Need to have ConfigParser module installed for this to work.")
    config = SafeConfigParser()
    if isinstance(inifile, string_types):
        config.read(inifile)
    elif isinstance(inifile, IOBase):
        if inifile.seekable():
            inifile.seek(0)
        config.read_file(inifile)
    if not config.has_section("Data"):
        raise RuntimeError("Configuration file lacks a [Data] section to describe data.")

    if config.has_option("Data", "type"):
        typ = config.get("Data", "type").split(".")
        typ = typ[-1]
    else:
        typ = "Data"
    csv_names = ("header_line", "data_line", "header_delim", "data_delim")
    layout = {}
    if typ == "CSVFile":
        overrides = {key: kwargs.pop(key) for key in csv_names if key in kwargs}
        layout = _ini_csv_options(config, overrides)
    data = make_Data(**kwargs)
    if filename is None:
        if not config.has_option("Data", "filename"):
            filename = False
        else:
            filename = config.get("Data", "filename")
            if filename in ["False", "True"]:
                filename = bool(filename)
    data = data.__class__.load(filename, auto_load=False, filetype=typ, **layout)
    cols = {"x": 0, "y": 1, "e": None}  # Defaults

    for c in ["x", "y", "e"]:
        if not config.has_option("Data", c):
            pass
        else:
            try:
                cols[c] = config.get("Data", c)
                cols[c] = int(cols[c])
            except ValueError:
                pass
        if cols[c] is None:
            del cols[c]

    data.setas(**cols)  # pylint: disable=not-callable
    return data


def _setup_one_param(data, config, p, prefix, vals):
    if not config.has_section(p):
        raise RuntimeError(f"Config file does not have a section for parameter {p}")
    keys = {
        "vary": bool,
        "value": float,
        "min": float,
        "max": float,
        "expr": str,
        "step": float,
        "label": str,
        "units": str,
    }
    kwargs = {}
    for k, val in keys.items():
        if config.has_option(p, k):
            if val == bool:
                kwargs[k] = config.getboolean(p, k)
            elif val == float:
                kwargs[k] = config.getfloat(p, k)
            elif val == str:
                kwargs[k] = config.get(p, k)
    if isinstance(data, make_Data(None)):  # stuff the parameter hint data into metadata
        for k in keys:  # remove keywords not needed
            if k in kwargs:
                data[f"{prefix}{p} {k}"] = kwargs[k]
        if "lmfit.prerfix" in data:
            data["lmfit.prefix"].append(prefix)
        else:
            data["lmfit.prefix"] = [prefix]
    if "step" in kwargs:  # We use step for creating a chi^2 mapping, but not for a parameter hint
        step = kwargs.pop("step")
        if "vary" in kwargs and "min" in kwargs and "max" in kwargs and not kwargs["vary"]:  # Make chi^2?
            vals.append(np.arange(kwargs["min"], kwargs["max"] + step / 10, step))
        else:  # Nope, just make a single value step here
            vals.append(np.array(kwargs["value"]))
    else:  # Nope, just make a single value step here
        vals.append(np.array(kwargs["value"]))
    kwargs = {k: v for k, v in kwargs.items() if k in ["value", "max", "min", "vary"]}
    return kwargs


def cfg_model_from_ini(inifile, model=None, data=None):
    r"""Utility function to configure an lmfit Model from an inifile.

    Args:
        inifile (str or file):
            Path to the ini file to be read.

    Keyword Arguments:
        model (str, callable, lmfit.Model instance or sub-class or None):
            What to use as a model function.
        data (DataFile):
            If supplied, the details of the parameter hints and labels and units are included in the data's metadata.

    Returns:
        An llmfit.Model,, a 2D array of starting values for each parameter

    model can be of several different types that determine what to do:
        -   A string. In which ase it should be a fully qualified name of a function or class to be imported.
            The part after the final period will be assumed to be the name and the remainder the module to be
            imported.
        -   A callable object. In this case the callable will be passed to the constructor of Model and a fresh
            Model instance is constructed
        -   A subclass of lmfit.Model - in which case it is instantiated.
        -   A Model instance - in which case no further action is necessary.

    The returned model is configured with parameter hints for fitting with. The second return value is
    a 2D array which lists the starting values for one or more fits. If the inifile describes mapping out
    the :math:`\Chi^2` as a function of the parameters, then this array has a separate row for each iteration.
    """
    config = SafeConfigParser()
    if isinstance(inifile, string_types):
        config.read(inifile)
    elif isinstance(inifile, IOBase):
        if inifile.seekable():
            inifile.seek(0)
        config.read_file(inifile)

    if model is None:  # Check to see if config file specified a model
        try:
            model = config.get("Options", "model")
        except (IOError, ValueError, TypeError) as err:
            raise RuntimeError("Model is notspecifed either as keyword argument or in inifile") from err
    model = _get_model_(model)
    if config.has_option("option", "prefix"):
        prefix = config.get("option", "prefix")
    else:
        prefix = type(model).__name__
    prefix += ":"
    vals = []
    for p in model.param_names:
        kwargs = _setup_one_param(data, config, p, prefix, vals)
        model.set_param_hint(p, **kwargs)  # set the model parameter hint
    msh = np.meshgrid(*vals)  # make a mesh of all possible parameter values to test
    msh = [m.ravel() for m in msh]  # tidy it up and combine into one 2D array
    msh = np.column_stack(msh)
    return model, msh
