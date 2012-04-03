# Plotting DataFile wiuth Chaco, TraitsUI, TraitsUI
from traits.api import HasTraits, Instance,  Array, Int, List, Enum,  Str
from traitsui.api import View, Group, HGroup,  Item,  CheckListEditor, Handler,  TreeEditor, TreeNode, VSplit
from enable.api import ColorTrait, LineStyle
from enable.component_editor import ComponentEditor
from chaco.api import marker_trait, Plot, ArrayPlotData, ArrayDataSource
import chaco.tools.api as plottools
from enthought.traits.ui.menu import Action, CloseAction, Menu, MenuBar, OKCancelButtons, Separator

import Stoner as S

class StonerPlot(HasTraits):
    """A simple x-y plotting program to play with"""
    plot = Instance(Plot)
    color = ColorTrait("blue")
    line_style=LineStyle()
    line_color = ColorTrait("blue")

    marker = marker_trait
    marker_size = Int(4)
    outline_width=Int(1)
    xc=Str
    yc=Str
    cols=List(['X', 'Y'])
    type=Enum('scatter', 'line')
    data=Instance(S.DataFile)
    
    menubar=MenuBar(
                    Menu(
                        Action(name = 'E&xit', accelerator="Ctrl+Q",  tooltip="E&xit",  action = '_on_close'),
                        Separator(),
                        Action(name="&Open", accelerator="Ctrl+O", tooltip="&Open Data File", action="load"), # these callbacks                         
                        Action(name="&Close", accelerator="Ctrl+W", tooltip="&Close Plot", action="close_plot"), # these callbacks                         
                        name="File"))



    def trait_view(self, parent=None):
        traits_view = View(Group(
                  HGroup(Item('xc',label='X Column', editor=CheckListEditor(name='cols')),
                         Item('yc', label='Y Column', editor=CheckListEditor(name='cols')),
                         Item('type', label='Plot Type')),
                  HGroup(Item('color', label="Colour", style="custom"),
                         Item('line_color', label="Line Colour", style="custom", visible_when='type=="scatter" and outline_width>0')),
                         HGroup(Item('marker', label="Marker", visible_when='type=="scatter"'),
                  Item('line_style',  label='Line Style',  visible_when="type=='line'"),
                  Item('marker_size', label="Marker Size",  visible_when='type=="scatter"'),  Item('outline_width', label="Line Width")),
                  Item('plot', editor=ComponentEditor(), show_label=False), orientation="vertical"), menubar=self.menubar,
                  width=800, 
                  height=600, 
                  resizable=True, 
                  title="Stoner Plotter",  
                  handler=MenuController)
        return traits_view

    def __init__(self):
        super(StonerPlot, self).__init__()
        self.data=S.DataFile()

        self._paint()
        
    def _load(self):
        self.data.load(False)
        self._paint()

    def _paint(self):
        p=self.data

        if len(p)==0:
            self.cols=['X', 'Y']
            plotdata=ArrayPlotData(x=[], y=[])
        else:
            self.cols=p.column_headers
            xc=0
            yc=1
            plotdata = ArrayPlotData(x = p.column(xc), y = p.column(yc))
    
        plot = Plot(plotdata)

        if self.type=="scatter":
            self.renderer = plot.plot(("x", "y"), type=self.type, color=self.color,  outline_color=self.line_color,  line_width=self.outline_width)[0]
        else:
            self.renderer = plot.plot(("x", "y"), type=self.type, color=self.color,  outline_style=self.line_style,  line_width=self.outline_width)[0]
        self.plot = plot
        self.plot.tools.append(plottools.PanTool(self.plot))
        #self.plot.tools.append(plottools.DragZoom(self.plot, drag_button='right', speed=0.25))
        self.plot.overlays.append(plottools.ZoomTool(self.plot, tool_mode="box", drag_button='right', always_on=True))

    def _color_changed(self):
        self.renderer.color = self.color

    def _line_color_changed(self):
        self.renderer.outline_color = self.line_color

    def _marker_changed(self):
        self.renderer.marker = self.marker

    def _line_style_changed(self):
        self.renderer.line_style = self.line_style

    def _type_changed(self):
        self._paint()

    def _marker_size_changed(self):
        self.renderer.marker_size = self.marker_size


    def _outline_width_changed(self):
        self.renderer.line_width = self.outline_width


    def _xc_changed(self):
        xc=self.xc
        data=self.data.column(xc)
        self.renderer.xlabel=xc
        self.renderer.index = ArrayDataSource(data)
        self.renderer.index_mapper.range.set_bounds(min(data), max(data))

    def _yc_changed(self):
        yc=self.yc
        self.renderer.ylabel=yc
        data=self.data.column(yc)
        self.renderer.value = ArrayDataSource(data)
        self.renderer.value_mapper.range.set_bounds(min(data), max(data))

class MenuController(Handler):

    def load(self, ui_info):
        view = ui_info.ui.context['object']
        view._load()

    def close_plot(self, ui_info):
        view = ui_info.ui.context['object']
        view.__init__()


if __name__ == "__main__":
    StonerPlot().configure_traits()


