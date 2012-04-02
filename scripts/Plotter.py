# Plotting DataFile wiuth Chaco, TraitsUI, TraitsUI
from traits.api import HasTraits, Instance,  Array, Int, List, Enum,  Str
from traitsui.api import View, Group, HGroup,  Item,  CheckListEditor
from enable.api import ColorTrait, LineStyle
from enable.component_editor import ComponentEditor
from chaco.api import marker_trait, Plot, ArrayPlotData, ArrayDataSource
import chaco.tools.api as plottools

import Stoner as S

class StonerPlot(HasTraits):
    """A simple x-y plotting program to play with"""
    plot = Instance(Plot)
    color = ColorTrait("blue")
    line_style=LineStyle()
    line_color = ColorTrait("blue")

    marker = marker_trait
    marker_size = Int(4)
    xc=Str
    yc=Str
    cols=List(['X', 'Y'])
    type=Enum('scatter', 'line')
    data=S.DataFile()
    
    view = View(
        Group(HGroup(Item('color', label="Color", style="custom"),
                     Item('line_color', label="Edge Color", style="custom"),
              Item('marker', label="Marker"),
              Item('line_style',  label='Line Style'), 
              Item('marker_size', label="Size")),
              HGroup(Item('xc',label='X Column', editor=CheckListEditor(name='cols')), 
                     Item('yc', label='Y Column', editor=CheckListEditor(name='cols')), 
                     Item('type', label='Plot Type')), 
              Item('plot', editor=ComponentEditor(), show_label=False),
                   orientation = "vertical"),
              width=800, height=600, resizable=True, title="Chaco Plot")
            
    def __init__(self):
        super(StonerPlot, self).__init__()

        self.data=S.DataFile(False)
        self._paint()
        
    def _paint(self):
        p=self.data

        self.cols=p.column_headers


        xc=0
        yc=1
        
        plotdata = ArrayPlotData(x = p.column(xc), y = p.column(yc))

        plot = Plot(plotdata)
        
        if self.type=="scatter":
            self.renderer = plot.plot(("x", "y"), type=self.type, color="blue",  outline_color="blue")[0]
        else:
            self.renderer = plot.plot(("x", "y"), type=self.type, color="blue",  outline_style=self.line_style,  line_wdith=self.marker_size)[0]
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
        if self.type=="scatter":
            self.renderer.marker_size = self.marker_size
        else:
            self.renderer.line_width = self.marker_size
        
    def _xc_changed(self):
        xc=self.xc
        data=self.data.column(xc)
        self.renderer.index = ArrayDataSource(data)
        self.renderer.index_mapper.range.set_bounds(min(data), max(data))

    def _yc_changed(self):
        xc=self.yc
        data=self.data.column(xc)
        self.renderer.value = ArrayDataSource(data)
        self.renderer.value_mapper.range.set_bounds(min(data), max(data))


if __name__ == "__main__":
    StonerPlot().configure_traits()

                
