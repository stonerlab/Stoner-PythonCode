# Plotting DataFile wiuth Chaco, TraitsUI, TraitsUI
from traits.api import HasTraits, Instance,  Array, Int, List, Enum,  Str,  Tuple,  Font,  Property
from traitsui.api import View, Group, HGroup,  Item,  CheckListEditor, Handler,  TreeEditor, TreeNode, VSplit, Tabbed,  TabularEditor
from traitsui.tabular_adapter import TabularAdapter
from enable.api import ColorTrait, LineStyle,  Component, ComponentEditor

from enthought.traits.ui.menu import Action, CloseAction, Menu, MenuBar, OKCancelButtons, Separator
from enable.tools.api import DragTool

from chaco.api import add_default_axes, add_default_grids, \
        OverlayPlotContainer, PlotLabel, ScatterPlot, LinePlot, create_line_plot, create_scatter_plot,  \
        marker_trait, Plot, ArrayPlotData, ArrayDataSource
from chaco.tools.api import PanTool, ZoomTool
import chaco.tools.api as plottools

import numpy

import Stoner as S

def _create_plot_component(myplot):

    container = OverlayPlotContainer(padding = 50, fill_padding = True,
                                     bgcolor = "white", use_backbuffer=True)

    # Create the initial X-series of data
    if len(myplot.data)>0:
        x=myplot.data.column(myplot.xc)
        y=myplot.data.column(myplot.yc)
    else:
        x=numpy.zeros(10)
        y=x

    if myplot.type=="scatter and line":
        lineplot = create_line_plot((x,y), color=myplot.color, width=myplot.outline_width, dash=myplot.line_style)
        lineplot.selected_color = "none"
        scatter = ScatterPlot(index = lineplot.index,
                           value = lineplot.value,
                           index_mapper = lineplot.index_mapper,
                           value_mapper = lineplot.value_mapper,
                           color = myplot.color,
                           marker=myplot.marker, 
                           marker_size = myplot.marker_size)
        scatter.index.sort_order = "ascending"
        scatter.bgcolor = "white"
        scatter.border_visible = True
        add_default_grids(scatter)
        add_default_axes(scatter)
        container.add(lineplot)
        container.add(scatter)
    elif myplot.type=="line":
        scatter= create_line_plot((x,y), color=myplot.color, width=myplot.outline_width, dash=myplot.line_style)
        scatter.marker_size=myplot.marker_size
        add_default_grids(scatter)
        add_default_axes(scatter)
        container.add(scatter)
    elif myplot.type=="scatter":
        scatter = create_scatter_plot((x,y), color=myplot.color, marker=myplot.marker)
        add_default_grids(scatter)
        add_default_axes(scatter)
        container.add(scatter)
       
    scatter.tools.append(PanTool(scatter, drag_button="left"))

    # The ZoomTool tool is stateful and allows drawing a zoom
    # box to select a zoom region.
    zoom = ZoomTool(scatter, tool_mode="box", always_on=True, drag_button="right")
    scatter.overlays.append(zoom)

    #scatter.tools.append(PointDraggingTool(scatter))

    # Add the title at the top
    #container.overlays.append(PlotLabel("Line Editor",
    #                          component=container,
    #                          font = "swiss 16",
    #                          overlay_position="top"))

    return container

class ArrayAdapter(TabularAdapter):

    font        = Font('Courier 10')
    alignment   = 'right'
    format      = '%.4f'
    index_text  = Property
    data=Instance(S.DataFile)
        

    def _get_index_text(self):
        return str(self.row)

class StonerPlot(HasTraits):
    """A simple x-y plotting program to play with"""
    plot = Instance(Component)
    color = ColorTrait("blue")
    line_style=LineStyle()
    line_color = ColorTrait("blue")

    marker = marker_trait
    marker_size = Int(4)
    outline_width=Int(1)
    xc=Str
    yc=Str
    cols=List(['X', 'Y'])
    type=Enum('scatter', 'line', 'scatter and line')
    data=Instance(S.DataFile)
    numpy_data=Array
    adapter = ArrayAdapter()
    
    menubar=MenuBar(
                    Menu(
                        Action(name = 'E&xit', accelerator="Ctrl+Q",  tooltip="E&xit",  action = '_on_close'),
                        Separator(),
                        Action(name="&Open", accelerator="Ctrl+O", tooltip="&Open Data File", action="load"), # these callbacks                         
                        Action(name="&Close", accelerator="Ctrl+W", tooltip="&Close Plot", action="close_plot"), # these callbacks                         
                        name="File"))


    
    def _plot_default(self):
         return _create_plot_component(self)

    def trait_view(self, parent=None):
        group1=Group(
            HGroup(
                Item('xc',label='X Column', editor=CheckListEditor(name='cols')),
                Item('yc', label='Y Column', editor=CheckListEditor(name='cols')),
                Item('type', label='Plot Type')
            ),
            HGroup(
                Item('color', label="Colour", style="custom"),
                 Item('line_color', label="Line Colour", style="custom", visible_when='"scatter" in type and outline_width>0')
            ),
            HGroup(
                Item('marker', label="Marker", visible_when='"scatter" in type'),
                Item('line_style',  label='Line Style',  visible_when="'line' in type"),
                Item('marker_size', label="Marker Size",  visible_when='"scatter" in type'),
                Item('outline_width', label="Line Width")
            ),
            Item('plot', editor=ComponentEditor(), show_label=False),
            label="Plot", 
            orientation="vertical"
        )
        
        group2=Group(
            Item('numpy_data',
                show_label = False,
                style      = 'readonly',
                editor     = TabularEditor(adapter = self.adapter)
            ),
            label="Numerical Data"
        )

        traits_view = View(Tabbed(group1, group2), menubar=self.menubar,
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
        self.cols=self.data.column_headers
        self.numpy_data=self.data.data
        self.adapter.columns=self.data.column_headers
        self._paint()

    def _paint(self):
        self.plot = _create_plot_component(self)
        self.renderer=self.plot.components[0]

    def _set_renderer(self, attr, value):
        for plot in self.plot.components:
            setattr(plot, attr, value)

    def _color_changed(self):
        self._set_renderer("color", self.color)

    def _line_color_changed(self):
        self._set_renderer("outline_color", self.line_color)

    def _marker_changed(self):
        self._set_renderer("marker", self.marker)

    def _line_style_changed(self):
        self._set_renderer("line_style", self.line_style)

    def _type_changed(self):
        self._paint()

    def _marker_size_changed(self):
        self._set_renderer("marker_size", self.marker_size)


    def _outline_width_changed(self):
        self._set_renderer("line_width", self.outline_width)


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


