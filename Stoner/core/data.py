#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The main Data Class definition."""

from sys import float_info

# Bring all the subclasses into memory (idnore unused imports warnings)
import Stoner.formats  # NOQA pylint: disable=W0611
import Stoner.HDF5  # NOQA pylint: disable=W0611
import Stoner.Zip  # NOQA pylint: disable=W0611
from Stoner.Analysis import AnalysisMixin
from Stoner.analysis.columns import ColumnOpsMixin
from Stoner.analysis.features import FeatureOpsMixin
from Stoner.analysis.filtering import FilteringOpsMixin
from Stoner.analysis.fitting.mixins import FittingMixin
from Stoner.Core import DataFile
from Stoner.plot import PlotMixin
from Stoner.tools import format_error


class Data(AnalysisMixin, FittingMixin, ColumnOpsMixin, FilteringOpsMixin, FeatureOpsMixin, PlotMixin, DataFile):
    """The principal class for representing a data file.

    This merges:
        -   :py:class:`Stoner.Core.DataFile`,
        -   :py:class:`Stoner.Analysis.AnalysisMixin`,
        -   :py:class:`Stoner.analysis.fitting.FittingMixin`,
        -   :py:class:`Stoner.analysis.columns.ColumnOpsMixin`,
        -   :py:class:`Stoner.analysis.filtering.FilteringOpsMixin`,
        -   :py:class:`Stoner.analysis.features.FeatureOpsMixin`,
        -   :py:class:`Stoner.plot.PlotMixin`

    Also has the :py:mod:`Stoner.formats` loaded redy for use.
    """

    def format(self, key, **kargs):
        r"""Return the contents of key pretty formatted using :py:func:`format_error`.

        Args:
            fmt (str): Specify the output format, options are:

                *  "text" - plain text output
                * "latex" - latex output
                * "html" - html entities

            escape (bool):
                Specifies whether to escape the prefix and units for unprintable characters in non
                text formats )default False)
            mode (string):
                If "float" (default) the number is formatted as is, if "eng" the value and error is converted
                to the next samllest power of 1000 and the appropriate SI index appended. If mode is "sci" then a
                scientific, i.e. mantissa and exponent format is used.
            units (string):
                A suffix providing the units of the value. If si mode is used, then appropriate si
                prefixes are prepended to the units string. In LaTeX mode, the units string is embedded in \mathrm
            prefix (string):
                A prefix string that should be included before the value and error string. in LaTeX mode this is
                inside the math-mode markers, but not embedded in \mathrm.

        Returns:
            A pretty string representation.

        The if key="key", then the value is self["key"], the error is self["key err"], the default prefix is
        self["key label"]+"=" or "key=", the units are self["key units"] or "".

        """
        mode = kargs.pop("mode", "float")
        units = kargs.pop("units", self.get(key + " units", ""))
        prefix = kargs.pop("prefix", f"{self.get(key + ' label', f'{key}')} = ")
        latex = kargs.pop("latex", False)
        fmt = kargs.pop("fmt", "latex" if latex else "text")
        escape = kargs.pop("escape", False)

        try:
            value = float(self[key])
        except (ValueError, TypeError) as err:
            raise KeyError(f"{key} should be a floating point value of the metadata not a {type(self[key])}.") from err
        try:
            error = float(self[f"{key} err"])
        except (TypeError, KeyError):
            error = float_info.epsilon
        return format_error(value, error, fmt=fmt, mode=mode, units=units, prefix=prefix, scape=escape)
