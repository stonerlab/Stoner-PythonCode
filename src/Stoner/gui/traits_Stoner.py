###########################################
#
# Classes to do a traitsui for DataFiles
#
# $Id: traits_Stoner.py,v 1.2 2013/04/09 21:39:13 cvs Exp $
#
# $Log: traits_Stoner.py,v $
# Revision 1.2  2013/04/09 21:39:13  cvs
# Messing with Stoner Ploting and Traits
#
# Revision 1.1  2013/03/26 23:51:12  cvs
# Add start of extra gui package
#

import numpy
import numpy.ma as ma

import Stoner as S
from traits.api import HasTraits, Instance,  Array, Int, List, Enum,  Str,  Tuple,  Font,  Property, Dict
from traitsui.api import View, Group, HGroup,  VGroup,  Item,  CheckListEditor, Handler,  TreeEditor, TreeNode, VSplit, Tabbed,  TabularEditor, ValueEditor, Label,  ListEditor
from traitsui.tabular_adapter import TabularAdapter


class ArrayAdapter(TabularAdapter):
    """A minimalist array adapter class for managing the display of data"""

    font        = Font('Courier 10')
    alignment   = 'right'
    format      = '%.4f'
    index_text  = Property
    data=Array
    width=75.0
    
    def __init__(self, df):
        self.data=df.data


    def _get_index_text(self):
        return "Row:"+str(self.row)
        
    def delete(self, datafile, trait, row):
        datafile.del_rows(row)
    
    def insert (self, object, trait, row, value ):
        object.insert_rows(row, value)
        self.data=object.data

    def get_default_value ( self,  object, trait ):
        return numpy.zeros(object.data.shape[1])


class TraitsDataFile(S.DataFile, HasTraits):
    """A subvlass of Stoner.DataFile that also inherits from HasTraits for use in building front ends"""

    def __init__(self, *args, **kargs):
        self.adapter=ArrayAdapter(self)
        self.data=Instance(ma.masked_array([]))
        self.metadata=Instance(S.typeHintedDict())
        self.column_headers=List(Str)
        super(TraitsDataFile, self).__init__(*args, **kargs)
        
    
    # Define the default view for this ckass (a tabular list of data)
    def trait_view(self, parent=None):
        self.display_group=HGroup(
            Item('data',
                show_label = False,
                style      = 'readonly',
                editor     = TabularEditor(adapter = self.adapter, multi_select=True, operations=["insert", "delete", "edit", "move", "append"])
            ),
            Item('metadata',
                 editor=ValueEditor(),
                 show_label=False,
                 width=0.25,  height=640)
        )
        traits_view = View(self.display_group,  height=640)
        return traits_view
    
    def load(self, *args, **kargs):
        """Override the load to sort out the columns and headers of the array adapter"""
        ret=super(TraitsDataFile, self).load(*args, **kargs) # Call the parent class method
        acols=[("Column "+str(i), i) for i in range(self.data.shape[1])]
        for i in range(len(self.column_headers)):
            acols[i]=(self.column_headers[i], i)
        acols[:0]=[("index", "index")]
        self.adapter.columns=acols
        return ret
