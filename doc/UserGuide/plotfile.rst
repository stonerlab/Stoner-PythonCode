*************
Plotting Data
*************
.. currentmodule:: Stoner.Plot

Data plotting and visualisation is handled by the :py:class:`PlotFile` sub-class of :py:class:`Sonter.DataFile`. 
The purpose of the methods detailed here is to provide quick and convenient ways to plot data rather than providing 
publication ready figures.::

   import Stoner.Plot as plot
   p=plot.PlotFile(d)

The first line imports the :py:module:`Stoner.Plot` module. Strictly, this is unnecessary as the Plot module's namespace is 
imported when the Stoner package as a whole is imported. The second line creates an instance of the :py:class:`PlotFile` class. 
PlotFile inherits the constructor method of :[y:class:`Stoner.Core.DataFile` and so all the variations detailed above work with 
PlotFile. In particular, the form shown in the second line is a easy way to convert a DataFile instance to a PlotFile instance 
for plotting.

Plotting 2D data
================

*x-y* plots are produced by the :py:meth:`PlotFile.plot_xy' method::

   p.plot_xy(column_x, column_y)
   p.plot_xy(column_x, [y1,y2])
   p.plot_xy(x,y,'ro')
   p.plot_xy(x,[y1,y2],['ro','b-'])
   p.plot_xy(x,y,title='My Plot')
   p.plot_xy(x,y,figure=2)
   p.plot_xy(x,y,plotter=pyplot.semilogy)

The examples above demonstrate several use cases of the :py:meth:`PlotFile.plot_xy` method. The first parameter is always 
the x column that contains the data, the second is the y-data either as a single column or list of columns. 
The third parameter is the style of the plot (lines, points, colours etc) and can either be a list if the y-column data 
is a list or a single string. Finally additional parameters can be given to specify a title and to control which figure 
is used for the plot. All matplotlib keyword parameters can be specified as additional keyword arguments and are passed 
through to the relevant plotting function. The final example illustrates a convenient way to produce log-linear and log-log 
plots. By default, :py:meth:`PlotFile.plot_xy` uses the **pyplot.plot** function to produce linear scaler plots. There 
are a number of useful plotter functions that will work like this::

*   [pyplot.semilogx,pyplot.semilogy] These two plotting functions will produce log-linear plots, with semilogx making 
    the x-axes the log one and semilogy the y-axis.
*   [pyplot.loglog] Liek the semi-log plots, this will produce a log-log plot.
*   [pyplot.errorbar] this particularly useful plotting function will draw error bars. The values for the error bars are 
    passed as keyword arguments, *xerr* or *yerr*. In standard matplotlib, these can be numpy arrays or constants. 
    :py:meth:`PlotFile.plot_xy` extends this by intercepting these arguements and offering some short cuts::

         p.plot_xy(x,y,plotter=errorbar,yerr='dResistance',xerr=[5,'dTemp+'])

    This is equivalent to doing something like::

         p.plot_xy(x,y,plotter=errorbar,yerr=p.column('dResistance'),xerr=[p.column(5),p.column('dTemp+')])

    If you actually want to pass a constant to the *x/yerr* keywords you should use a float rather than an integer.

The X and Y axis label will be set from the column headers.

Plotting 3D Data
================

A number of the measurement rigs will produce data in the form of rows of $x,y,z$ values. Often it is desirable to plot 
these on a surface plot or 3D plot. The :py:meth:`PlotFile.plot_xyz` method can be used for this.::

    p.plot_xyz(col_x,col_y,col_z)
    p.plot_xyz(col_x,col_y,col_z,cmap=matplotlib.cm.jet)
    p.plot)xyz(col-x,col-y,col-z,plotter=pyplot.pcolor)
    p.plot_xyz(col_x,col_y,col_z,xlim=(-10,10,100),ylim=(-10,10,100))

By default the :py:meth:`PlotFile.plot_xyz` will produce a 3D surface plot with the z-axis coded with a rainbow colourmap 
(specifically, the matplotlib provided *matplotlib.cm.jet* colourmap. This can be overriden with the *cmap* keyword 
parameter. If a simple 2D surface plot is required, then the *plotter* parameter should be set to a suitable function 
such as **pyplot.pcolor**.

 Like :py:meth:`PlotFile.plot_xy`, a *figure* parameter can be used to control the figure being used and any additional 
keywords are passed through to the plotting function. The axes labels are set from the corresponding column labels.
 
 Another option is a contour plot based on ``(x,y,z)`` data points. This can be done with the :py:meth:`PlotFile.contour_xyz``
 method.::

 	p.contour_xyz(xcol,ycol,zcol,shape=(50,50))
 	p.contour_xyz(xcol,ycol,zcol,xlim=(10,10,100),ylim=(-10,10,100))

Both :py:meth:`PlotFile.plot_xyz` and :py:meth:`PlotFile.contour\_xyz` make use of a call to :py:meth:`PlotFile.griddata`
 which is a utility method of the :py:class:`PlotFile` -- essentially this is just a pass through method to the underlying 
*scipy.interpolate.griddata** function. The shape of the grid is determined through a combination of the *xlim*, *ylim*
 and *shape* arguments.::

    X,Y,Z=p.griddata(xcol,ycol,zcol,shape=(100,100))
    X,Y,Z=p.griddata(xcol,ycol,zcol,xlim=(-10,10,100),ylim=(-10,10,100))

If a *xlim* or *ylim* arguments are provided and are two tuples, then they set the maximum and minimum values of the relevant axis.
If they are three tuples, then the third argument is the number of points along that axis and overrides any setting in the *shape*
 parameter. If the *xlim* or *ylim* parameters are not presents, then the maximum and minimum values of the relevant axis are used. 
If no *shape* information is provided, the default is to make the shape a square of sidelength given by the square root of the 
number of points.

 Alternatively, if your data is already in the form of a matrix, you can use the :py:meth:`PlotFile.plot_matrix` method::

    p.plot_matrix()
    p.plot_matrix(xvals,yvals,rectang,title="Title",xlabel="X Axis",ylabel="Y Axis",zlabel="Z Axis",cmap=matplotlib.cm.jet)
    p.plot_matrix(plotter=pyplot.pcolor,figure=False)

The first example just uses all the default values, in which case the matrix is assumed to run from the 2nd column in the 
file to the last and over all of the rows. The x values for each row are found from the contents of the first column, and 
the y values for each column are found from the column headers interpreted as a floating pint number. The colourmap defaults 
to the built in `jet' theme. The x axis label is set to be the column header for the first column, the y axis label is set 
either from the meta data item 'ylabel or to 'Y Data'. Likewise the z axis label is set from the corresponding metadata 
item or defaults to 'Z Data;. In the second form these parameters are all set explicitly. The *xvals* parameter can be 
either a column index (integer or sring) or a list, tuple or numpy array. The *yvals* parameter can be either a row number 
(integer) or list,tuple or numpy array. Other parameters (including *plotter*, *figure* etc) work as for the 
:py:meth:`PlotFile.plot_xyz` method. The *rectang* parameter is used to select only part of the data array to use as the matrix. 
It may be 2-tuple in which case it specifies just the origin as (row,column) or a 4-tuple in which case the third and forth 
elements are the number of rows and columns to include. If *xvals* or *yvals* specify particular column or rows then the 
origin of the matrix is moved to be one column further over and one row further down (ie the matrix is to the right and 
below the columns and rows used to generate the x and y data values). The final example illustrates how to generate a 
new 2D surface plot in a new window using default matrix setup.

Getting More Control on the Figure
==================================

It is useful to be able to get access to the matplotlib figure that is used for each :py:class:`PlotFle` instance. The 
:py:attr:`PlotFile.fig` attribute can do this, thus allowing plots from multiple :py:class:`PlotFile` instances to be 
combined in a single figure.::

    p1.plot_xy(0,1,'r-')
    p2.plot_xy(0,1,'bo',figure=p1.fig)

 Likewise the :py:attr:`PlotFile.axes` attribute returns the current axes object of the current figure in use by the :py:class:`PlotFile`
instance.

There's a couple of extra methods that just pass through to the pyplot equivalents::

    p.draw()
    p.show()

Setting Axes Labels, Plot Titles and Legends
--------------------------------------------

.. todo::
   Document the other attributes of :py:xlass:`PlotFile`

Making Multi-plot Figures
-------------------------

.. todo::
   Write about :py:attr:`PlotFile.subplots`