#----------------------------------------------------------------------------- 
#   $Id: __init__.py,v 1.6 2011/01/08 20:58:35 cvs Exp $
#   AUTHOR:     MATTHEW NEWMAN, CHRIS ALLEN, GAVIN BURNELL
#   DATE:       24/11/2010
#-----------------------------------------------------------------------------
#
# $Log: __init__.py,v $
# Revision 1.6  2011/01/08 20:58:35  cvs
# Add CVS log tag to get changelog in header of file
#
#

__all__=['Core', 'Analysis', 'Plot']

# These fake the old namespace if you do an import Stoner
from .Core import *
from .Analysis import *
from Plot import *


