# Plotting DataFile wiuth Chaco, TraitsUI, TraitsUI
from traits.api import HasTraits, Instance,  Array, Int, List, Enum,  Str,  Tuple,  Font,  Property, Dict
from traitsui.api import View, Group, HGroup,  VGroup,  Item,  CheckListEditor, Handler,  TreeEditor, TreeNode, VSplit, Tabbed,  TabularEditor, ValueEditor
from traitsui.tabular_adapter import TabularAdapter
from enable.api import ColorTrait, LineStyle,  Component, ComponentEditor

from enthought.traits.ui.menu import Action, CloseAction, Menu, MenuBar, OKCancelButtons, Separator
from enable.tools.api import DragTool

from chaco.api import add_default_axes, add_default_grids, \
        OverlayPlotContainer, PlotLabel, ScatterPlot, LinePlot, create_line_plot, create_scatter_plot,  \
        marker_trait, Plot, ArrayPlotData, ArrayDataSource,  LinearMapper,  LogMapper, DataRange1D
from chaco.tools.api import PanTool, ZoomTool
from enthought.chaco.tools.cursor_tool import CursorTool, BaseCursorTool

import numpy

import Stoner as S


class ArrayAdapter(TabularAdapter):

    font        = Font('Courier 10')
    alignment   = 'right'
    format      = '%.4f'
    index_text  = Property
    data=Instance(S.DataFile)
    width=75.0
        

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
    xc=Str("X")
    yc=Str("Y")
    xm=Enum("Linear Scale", "Log Scale")
    ym=Enum("Linear Scale", "Log Scale")
    mappers={"Linear Scale":LinearMapper, "Log Scale":LogMapper}
    cols=List(['X', 'Y'])
    type=Enum('scatter', 'line', 'scatter and line')
    data=Instance(S.DataFile)
    numpy_data=Array
    metadata=Dict
    adapter = ArrayAdapter()
    
    menubar=MenuBar(
                    Menu(
                        Action(name = 'E&xit', accelerator="Ctrl+Q",  tooltip="E&xit",  action = '_on_close'),
                        Separator(),
                        Action(name="&Open", accelerator="Ctrl+O", tooltip="&Open Data File", action="load"), # these callbacks                         
                        Action(name="&Close", accelerator="Ctrl+W", tooltip="&Close Plot", action="close_plot"), # these callbacks                         
                        name="File"))


    def _create_plot(self, orientation="h",type=ScatterPlot):
        """
        Creates a ScatterPlot from a single Nx2 data array or a tuple of
        two length-N 1-D arrays.  The data must be sorted on the index if any
        reverse-mapping tools are to be used.
       
        Pre-existing "index" and "value" datasources can be passed in.
        """
        
        index = ArrayDataSource(self.data.column(self.xc), sort_order="none")
        value= ArrayDataSource(self.data.column(self.yc))
        index_range = DataRange1D()
        index_range.add(index)
        index_mapper = self.mappers[self.xm](range=index_range)
        value_range = DataRange1D()
        value_range.add(value)
        value_mapper = self.mappers[self.ym](range=value_range)
       
        plot = type(index=index, value=value,
                 index_mapper=index_mapper,
                 value_mapper=value_mapper,
                 orientation=orientation,
                 border_visible=True, 
                 bgcolor="transparent", 
                 color=self.color)
        if issubclass(type, ScatterPlot):
            plot.marker=self.marker
            plot.marker_size=self.marker_size
            plot.outline_color=self.line_color
        elif issubclass(type, LinePlot):
            plot.line_wdith=self.outline_width
            plot.line_style=self.line_style
            
        return plot

    def _create_plot_component(self):
    
        container = OverlayPlotContainer(padding = 50, fill_padding = True,
                                         bgcolor = "white", use_backbuffer=True)
        types={"line":LinePlot, "scatter":ScatterPlot}
    
        # Create the initial X-series of data
        if len(self.data)>0: # Only create a plot if we ahve datat
            if self.type=="scatter and line":
                lineplot = self._create_plot(type=LinePlot)
                lineplot.selected_color = "none"
                scatter=self._create_plot(type=ScatterPlot)
                scatter.bgcolor = "white"
                scatter.index_mapper=lineplot.index_mapper
                scatter.value_mapper=lineplot.value_mapper
                add_default_grids(scatter)
                add_default_axes(scatter)
                container.add(lineplot)
                container.add(scatter)
            else:
                plot= self._create_plot(type=types[self.type])
                add_default_grids(plot)
                add_default_axes(plot)
                container.add(plot)
                scatter=plot               
            scatter.tools.append(PanTool(scatter, drag_button="left"))
        
            # The ZoomTool tool is stateful and allows drawing a zoom
            # box to select a zoom region.
            zoom = ZoomTool(scatter, tool_mode="box", always_on=True, drag_button="right")
            scatter.overlays.append(zoom)
            csr=CursorTool(scatter, color="black", drag_button="left")
            scatter.overlays.append(csr)
        
        self.plot=container
        return container

    
    def _plot_default(self):
         return self._create_plot_component()

    def trait_view(self, parent=None):
        group1=Group(
            HGroup(
                VGroup(
                    Item('xc',label='X Column', editor=CheckListEditor(name='cols')),
                    Item('xm', label="X Scale")
                ), 
                VGroup(
                    Item('yc', label='Y Column', editor=CheckListEditor(name='cols')),
                    Item('ym', label="Y scale")
                ), 
                Item('type', label='Plot Type')
            ),
            HGroup(
                Item('color', label="Colour", style="simple", width=75),
                 Item('line_color', label="Line Colour", style="simple", visible_when='"scatter" in type and outline_width>0',  width=75), 
                Item('marker', label="Marker", visible_when='"scatter" in type'),
                Item('line_style',  label='Line Style',  visible_when="'line' in type"),
                Item('marker_size', label="Marker Size",  visible_when='"scatter" in type'),
                Item('outline_width', label="Line Width")
            ),
            Item('plot', editor=ComponentEditor(), show_label=False),
            label="Plot", 
            orientation="vertical"
        )
        
        group2=HGroup(
            Item('numpy_data',
                show_label = False,
                style      = 'readonly',
                editor     = TabularEditor(adapter = self.adapter)
            ),
            Item('metadata', 
                 editor=ValueEditor(), 
                 show_label=False, 
                 width=0.25), 
            label="Data"
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
        self.data.data=numpy.zeros((1, 2))
        self.data.column_headers=["X", "Y"]

        self._paint()
        
    def _load(self):
        self.data.metadata.clear()
        self.data.load(False)
        self.xc=self.data.column_headers[0]
        self.yc=self.data.column_headers[1]
        
        self._paint()

    def _paint(self):
        self.cols=self.data.column_headers
        self.numpy_data=self.data.data
        cols=[(self.data.column_headers[i], i) for i in range(len(self.data.column_headers))]
        cols[:0]=[("index", "index")]
        self.adapter.columns=cols
        self.metadata=self.data.metadata

        self.plot = self._create_plot_component()
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
        
    def _xm_changed(self):
        self._paint()
        
    def _ym_changed(self):
        self._paint()

class MenuController(Handler):

    def load(self, ui_info):
        view = ui_info.ui.context['object']
        view._load()

    def close_plot(self, ui_info):
        view = ui_info.ui.context['object']
        view.__init__()


if __name__ == "__main__":
    app=StonerPlot()
    app.configure_traits()


