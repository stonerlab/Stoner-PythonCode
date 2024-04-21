"""Provides extra classes that can load data from various instruments into _SC_.DataFile type objects.

You do not need to use these classes directly, they are made available to :py:class:`Stoner.Core.Data` which
will load each of them in turn when asked to load an unknown data file.

Each class has a priority attribute that is used to determine the order in which
they are tried by :py:class:`Stoner.Core.Data` and friends where trying to load data.
High priority is run last (so is a bit of a misnomer!).

Each class should implement a load() method and optionally a save() method. Classes should make every effort to
positively identify that the file is one that they understand and throw a
:py:exception:Stoner.Core._SC_.StonerLoadError` if not.
"""

from warnings import warn

__all__ = [
    "BNLFile",
    "BigBlueFile",
    "BirgeIVFile",
    "CSVFile",
    "EasyPlotFile",
    "ESRF_DataFile",
    "ESRF_ImageFile",
    "FmokeFile",
    "GenXFile",
    "HyperSpyFile",
    "KermitPNGFile",
    "LSTemperatureFile",
    "MDAASCIIFile",
    "MokeFile",
    "OVFFile",
    "OpenGDAFile",
    "PinkLibFile",
    "QDFile",
    "RasorFile",
    "RigakuFile",
    "SNSFile",
    "SPCFile",
    "TDMSFile",
    "VSMFile",
    "XRDFile",
]
# pylint: disable=unused-argument
from Stoner.formats.instruments import LSTemperatureFile, QDFile, RigakuFile, SPCFile, VSMFile, XRDFile
from Stoner.formats.facilities import (
    BNLFile,
    MDAASCIIFile,
    OpenGDAFile,
    RasorFile,
    SNSFile,
    ESRF_DataFile,
    ESRF_ImageFile,
)
from Stoner.formats.generic import CSVFile, KermitPNGFile, TDMSFile, HyperSpyFile
from Stoner.formats.rigs import BigBlueFile, BirgeIVFile, MokeFile, FmokeFile, EasyPlotFile, PinkLibFile
from Stoner.formats.simulations import GenXFile, OVFFile

warn(
    "*" * 80
    + "\nStoner.FileFormats is a deprecated module - use Stoner.formats and it's sub-modules  now!\n"
    + "*" * 80
)
