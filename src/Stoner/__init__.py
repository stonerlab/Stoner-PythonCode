#-----------------------------------------------------------------------------
#   $Id: __init__.py,v 1.10 2012/05/02 23:03:09 cvs Exp $
#   AUTHOR:     MATTHEW NEWMAN, CHRIS ALLEN, GAVIN BURNELL
#   DATE:       24/11/2010
#-----------------------------------------------------------------------------
#
# $Log: __init__.py,v $
# Revision 1.10  2012/05/02 23:03:09  cvs
# Update documentation, improve loading handling of external fileformats.
#
# Revision 1.9  2012/03/24 00:36:04  cvs
# Add a new DataFolder class with methods for sorting and grouping data files
#
# Revision 1.8  2011/12/03 13:58:48  cvs
# Replace the various format load routines in DataFile with subclasses of DataFile with their own overloaded load methods
# Improve the VSM load routine
# Add some new sample data sets to play with
# Updatedocumentation
#
# Revision 1.7  2011/01/10 23:11:21  cvs
# Switch to using GLC's version of the mpit module
# Made PlotFile.plot_xy take keyword arguments and return the figure
# Fixed a missing import math in AnalyseFile
# Major rewrite of CSA's PCAR fitting code to use mpfit and all the glory of the Stoner module - GB
#
# Revision 1.6  2011/01/08 20:58:35  cvs
# Add CVS log tag to get changelog in header of file
#
#

""""
@mainpage The Stoner Package
@section Introduction

This help file provides a programming and developer reference to ther Stoner Package for data analysis code.
For a general introduction, users are ferered to the User Guid pdf file that should be in the same directory as this
compiled help file.

@section Overview

The Stoner package provides two basic top-level classes that describe an individual file of experimental data and a list (such as
a directory on disc) of many experimental files.

Stoner.Core.DataFile is the base class for representing individual experimental data sets. It provides basic methods to examine and manipulate data, 
manage metadata and load and save data files. It has a large number of sub classes - most of these are in Stoner.FileFormats and are used to handle the loading of specific
file formats. Two, however, contain additional functionality for writing analysis programs.

Stoner.Analysis.AnalyseFile provides additional methods for curve-fitting, differentiating, smoothing and carrying out basic calculations on data.
Stoner.Plot.PlotFile provides additional routines for plotting data on 2D or 3D plots.

Stoner.Folders.DataFolder is a class for assisting with the work of processing lots of files in a common directory structure. It provides methods to list. filter and group data 
according to filename patterns or metadata.

@section Other Resources

Included in the package are a (small) collection of sample scripts for carrying out various operations and some sample data files for testing the 
loading and processing of data. Finally, this folder contains the LaTeX source code, dvi file and pdf version of the User Guide and this compiled help file
which has been gerneated from by Docygen from the contents of the python docstrings in the source code.
"""

__all__=['Core', 'Analysis', 'Plot', 'FileFormats', 'Folders']

# These fake the old namespace if you do an import Stoner
from .Core import *
from .Analysis import *
from Plot import *
from FileFormats import *
from Folders import *

