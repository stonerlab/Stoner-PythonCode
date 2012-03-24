#-----------------------------------------------------------------------------
#   $Id: __init__.py,v 1.9 2012/03/24 00:36:04 cvs Exp $
#   AUTHOR:     MATTHEW NEWMAN, CHRIS ALLEN, GAVIN BURNELL
#   DATE:       24/11/2010
#-----------------------------------------------------------------------------
#
# $Log: __init__.py,v $
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

__all__=['Core', 'Analysis', 'Plot', 'FileFormats', 'Folders']

# These fake the old namespace if you do an import Stoner
from .Core import *
from .Analysis import *
from Plot import *
from FileFormats import *
from Folders import *

