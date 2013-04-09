import numpy
import numpy.ma as ma

from Stoner.gui.traits_Stoner  import TraitsDataFile
from traits.api import HasTraits, Instance,  Array, Int, List, Enum,  Str,  Tuple,  Font,  Property, Dict
from traitsui.api import View, Group, HGroup,  VGroup,  Item,  CheckListEditor, Handler,  TreeEditor, TreeNode, VSplit, Tabbed,  TabularEditor, ValueEditor, Label,  ListEditor, ShellEditor
from traitsui.tabular_adapter import TabularAdapter

        
class MultiplePlots(HasTraits):
    """A class that represents a set of TraitsDataFile objects"""
    datafiles=List(TraitsDataFile)
    selected = Instance(TraitsDataFile)
    shell=Dict
    
    def __init__(self, *args, **kargs):
        for arg in args:
            if isinstance(arg, str):
                self.load(arg)
        
    
    def trait_view(self, parent=None):
        self.shell=self.__dict__
        traits_view = View(VGroup(
            Item('shell',editor=ShellEditor(), id="Shell"), 
            Item('datafiles@',
                  id = 'notebook',
                  show_label = False,
                  editor = ListEditor(use_notebook = True,
                                           deletable = False,
                                           selected='selected', 
                                           export = 'DockWindowShell',
                                           page_name = '.filename')
                )), 
                width=1024,
                height=768,
                resizable=True,
                title="Stoner Plotter")
        return traits_view
                      
    def load(self, *args, **kargs):
        """Load a TraitsDataFile in"""
        d=TraitsDataFile(*args, **kargs)
        d.trait_view()
        self.datafiles.append(d)

    
