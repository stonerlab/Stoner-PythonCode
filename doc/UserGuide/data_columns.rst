Choosing columns and assigning roles
************************************

.. currentmodule:: Stoner

Column names tell you what was measured. Column roles tell Stoner which columns
to use as coordinates, signals and uncertainties when plotting or fitting.
You can select a column by its name or by its position; positions start at zero.

For a three-column measurement containing temperature, resistance and resistance
uncertainty, start with::

    d.column_headers = ["Temperature", "Resistance", "Uncertainty"]
    d.setas = "xye"
    temperature = d.column("Temperature")
    resistance = d.y

Here ``x`` is the horizontal coordinate, ``y`` the measured signal, and ``e`` its
uncertainty. Use the example in :doc:`datafile` if you do not yet have a Data
object. Later examples illustrate alternative layouts, including tables with
five or more columns; adapt them to the columns in your own measurement.

.. contents:: On this page
   :local:
   :depth: 1

.. _setas:

Assign column roles with setas
==============================

Assign one character to each column, in order. Unmarked columns still contain
data; Stoner simply does not give them a coordinate or uncertainty role.

======================  ================================================
Role                    Meaning
======================  ================================================
``x``, ``y``, ``z``      Coordinates or measured signals
``d``, ``e``, ``f``      Uncertainties in x, y and z, respectively
``u``, ``v``, ``w``      Vector components at a coordinate (x, y, z)
``.``                   Unmarked column
``-``                   Leave the existing assignment unchanged
======================  ================================================

Many plotting and analysis methods use these assignments as their defaults.
Vector-component uncertainties do not have assigned role letters.

A saved ``setas`` reference follows the current Data object through column
renaming, reordering and insertion. Use ``clone`` for independent role and header
descriptions::

    roles = d.setas
    detached = roles.clone
    detached[1] = "z"

Editing ``detached`` leaves ``d`` unchanged. ``roles.cols`` returns newly derived
coordinate/error groups; changing that dictionary does not assign roles. Change
``roles`` itself to update the Data object. Invalid role assignments leave the
existing assignments unchanged, including on detached objects.

Setting Column Types
--------------------

To set which columns contain 'x','y' etc data use the :py:attr:`~Stoner.core.data.Data.setas` attribute. This attribute can take
a list of single character strings from the set 'x','y','z','d','e', 'f', 'u', 'v', 'w' or '.' where each element of the list refers to
the columns of data in order. To specify that a column has unmarked data use the '.' string. The string '-' can also be used - this
indicates that the current assignment is to be left as is.

Alternately, you can pass :py:attr:`~Stoner.core.data.Data.setas` a string. In the simplest case, the string is just read in the same way that
the list would have been  - each character corresponds to one column. However, if the string contains an integer, then the next
non-numeric character will be interpreted that many times, so::

    d.setas="3.xy"
    d.setas="...xy"
    d.setas=['.','.','.','x','y']

There are still more ways of setting column types with the :py:attr:`~Stoner.core.data.Data.setas` attribute::

    d.setas[3]="x"
    d.setas["x"]=3
    d.setas("3.xy")
    d.setas(['.','.','.','x','y'])
    d.setas(x=3,y=4)

All achieve the same effect of setting the same columns as containing x and y data.

Once you have identified columns for the various types, you also have access to utility attributes to access those columns::

    d.setas="3.xye"
    d.x == d.column(3)
    d.y == d.column(4)
    d.e == d.column(5)

Note that if :py:attr:`~Stoner.core.data.Data.setas` is not set then attempting to use the quick access column attributes will
use the count-based defaults below where available, or raise an exception. Once the
:py:attr:`~Stoner.core.data.Data.setas` attribute is set, a further set of *virtual* or *derived* column attributes
become available::

    d.setas="xyz"
    d.r # The magnitude of the point x,y,z
    d.q # The angle in the x-y plane relative to the x axis
    d.p # The angle relative to the z axis
    d.setas="xyz..uvw"
    d.r # the magnitude of (u,v,w)
    d.q # The angle in the x-y plane of (u,v) relative to +x
    d.p # The angle of (u,v,w) relative to +z

Where fewer than 3 or 6 dimensions are specified, these virtual columns fall back to working with the appropriate
reduced number of dimensions.

There are some more convenience ways to set which columns to use as x,y,z etc::

    d.setas={"x":0,"y":"Y Column title"}
    d.x=0
    d.y="Y Column title"
    d.setas["Temperature"]="y"

Single-letter role keys such as ``'x'`` and ``'y'`` have their role-assignment
meaning in ``setas``. If a header has the same spelling, resolve it with
``d.find_col('y')`` and use the resulting positions as keys to assign its role.

Reading Column Types
--------------------

The normal representation of :py:attr:`~Stoner.core.data.Data.setas` is as a list, but it also has a string conversion available. You can also find which column
has been assigned as 'x', 'y' etc. by treating the :py:attr:`~Stoner.core.data.Data.setas` as a dictionary::

    d.column_headers=["One","Two","three","Four"]
    d.setas="xy.z"

    print(list(d.setas)) #['x','y','.','z']
    print(d.setas) #'xy.z'
    print(d.setas['x']) # "One"
    print(d.setas["#x"]) # 0

Note that the :py:attr:`~Stoner.core.data.Data.setas` attribute supports reading keys that are either the single letter to get the name of the column or the letter
preceded by a # character to get the number of the column.

Alternatively, and equivalently, you can access the column indexes via attributes of :py:attr:`~Stoner.core.data.Data.setas`:

    d.setas.has_xcol # True
    d.setas.has_ucol # False
    d.setas.ycol # [1]
    d.setas.xcol # 0

Implied Column Assignments
--------------------------

If you do not specify the column types via the setas attributes, then :py:class:`~Stoner.core.data.Data` will try to guess sensible columns assignments based on the number
of columns in your data file. These default assignments are only done at the point at which the :py:attr:`~Stoner.core.data.Data.setas` attribute is consulted. The default
assignments are:

=================  ================
Number of Columns  Assignments
=================  ================
2                  x, y
3                  x, y, e
4                  x, d, y, e
5                  x, y, u, v, w
6                  x, y, z, u, v, w
=================  ================

Swapping and Rotating Column Assignments
----------------------------------------

Finally, if the :py:attr:`~Stoner.core.data.Data.setas` attribute has been set with *x*, *y* (and *z*) columns then these assignments can be
swapped around by the **invert** operator **~**. This either swaps *x* and *y* with their associated error bars for 2-D datasets, or rotates
*x* to *y*, *y* to *z* and *z* to *x* (again with their associated error bars)::

    d.setas="xye"
    print(d.setas)
    >>> ['x','y','e']
    e=~d
    print(e.setas)
    >>> ['y','x','d']

Read columns by name or position
================================

This is all very well, but often you want to examine a particular column of data
or a particular row::

  d.column(0)
  d.column('Temperature')
  d.column(['Temperature',0])

In the first example, the first column of numeric data will be returned. In the
second example, the column headers will first be checked for one labelled exactly
*Temperature* and then if no column is found, the column headers will be
searched for the literal substring *Temperature*. This selects both
*Temperature (K)* and *Sample Temperature* if present. The third
example results in a 2 dimensional numpy array containing two columns in the
order that they appear in the list (ie not the order that they are in the data
file). For completeness, the :py:meth:`~Stoner.core.data.Data.column` method also allows one to
pass slices to select columns and should do the expected thing.

There are a couple of convenient short cuts. Firstly the *floor-division* operator //
is an alias for the :py:meth:`~Stoner.core.data.Data.column` method and secondly for working
with cases where the column headers are not the same as the names of any of the attributes
of the :py:class:`~Stoner.core.data.Data` object::

  d//"Temperature"
  d.Temperature
  d.column('Temperature')

all return the same data.

Whenever the Stoner package needs to refer to a column of data, you can specify it in a number of ways:-

 1) As an integer where the first column on the left is index 0
 2) As a string: select all exact header matches first. If none exist, try case-sensitive
      whole-header glob matching when the string contains ``*``, ``?`` or a complete
      bracket expression; otherwise try a literal substring. If that stage finds no
      matches, try regular-expression search. Stop at the first successful stage.
 3) As a regular expression directly - this is similar to the case above with a string, but allows you to pass a pre-compiled regular
      expression in directly with any extra options (like case insensitivity flags) set.
 4) As a slice object (e.g. ``0:10:2``) this is expanded to a list of integers.
 5) As a list of any of the above, in which case the column finding routine is called recursively in turn for each element of the list and
      the final result would be to use a list of column indices.

For example::

    import re
    col=re.compile('^temp',re.IGNORECASE)
    d.column(col)

Advanced: patterns, column indices and iteration
================================================

Many methods in Stoner index columns by searching the column
headings. If one wishes to find the numeric index of a column then the
:py:meth:`~Stoner.core.data.Data.find_col` method can be used::

   index=d.find_col(1)
   index=d.find_col('Temperature')
   index=d.find_col('Temp.*')
   index=d.find_col('1')
   index=d.find_col(slice(1, 10, 2))
   index=d.find_col(['Temperature',2,'Resistance'])
   index=d.find_col(re.compile(r"^[A-Z]"))

Integer selectors use positions, including ordinary negative indices. String
selectors return all matches from the first successful stage: exact equality,
glob or literal substring, then regex fallback. Glob matching covers the entire
header and is case-sensitive on every platform. Compiled regular expressions
bypass those stages and preserve flags such as ``re.IGNORECASE``.

One string match returns an integer; several return a list. Compiled patterns,
slices and collections always return lists. Collections preserve selector order
and repeated positions. ``force_list=True`` always returns a list. To deliberately
choose just the first match, use::

   index = d.find_col('Temperature', first=True)
   values = d.column(index)
   indices = d.find_col('Temperature', first=True, force_list=True)

``first=True`` validates the whole selector before taking its first result. It
raises ``KeyError`` for an empty selection. Unmatched names/patterns also raise
``KeyError``; invalid positional indices raise ``IndexError``; invalid regex
fallback raises ``re.error``. A numeric string such as ``'1'`` selects a header
or pattern; use integer ``1`` to select a position. Boolean selectors are invalid.

Exact names containing units or metacharacters still win: ``'Field [T]'``
selects that literal header before interpreting brackets as glob syntax. A
complete glob bracket expression triggers glob dispatch only when no exact
match exists; an unmatched bracket alone remains a literal substring until
regex fallback is needed. A failed glob goes directly to regex search.

Exact duplicate names select every occurrence, without including longer headers.
For example, ``'Signal'`` selects two headers named exactly ``Signal`` before it
considers ``Signal error``. ``'Sig'`` selects all three by substring matching.
To force regex interpretation rather than glob/substring precedence, pass
``re.compile(pattern)``. An invalid regex-looking string can still succeed as an
exact name or literal substring without being compiled.

Methods requiring one coordinate, predicate column or observation reject an
ambiguous explicit selector with ``ValueError``. Use an integer position or
explicitly call ``find_col(..., first=True)`` to make the choice. Methods supporting
multiple channels use every selected column; general curve fitting also supports
several predictor columns. Native ``d.data`` retains ordinary pandas indexing.

Numerical assignment uses NumPy broadcasting: a scalar fills all selected cells;
an ``(nrows, 1)`` array broadcasts one observation vector across several selected
columns. Role assignment broadcasts one role, or accepts a role sequence whose
length equals the expanded selection. Repeated/overlapping assignments are
processed in order, with later writes winning; invalid bound role edits roll back.

This is the function that is used internally by :py:meth:`~Stoner.core.data.Data.column`,
:py:meth:`~Stoner.core.data.Data.search` etc and for this reason the trivial integer and slice
forms are implemented to allow these other functions to work with multiple
columns.

Runnable example: duplicate voltage channels
============================================

This example constructs two measurement channels with the same voltage header.
It checks matching precedence and scalar ambiguity, follows saved role/header
views through renaming, and repeats a masked channel while preserving column
identity rules. The two x groups plot their own assigned uncertainties.
The assertions run as part of the documentation-example tests.

.. literalinclude:: ../samples/column_schema.py
   :language: python

Download :download:`column_schema.py <../samples/column_schema.py>` and run it
with the package installed. It creates a Matplotlib figure; call
``matplotlib.pyplot.show()`` in an interactive session to display it.

Iterate over rows and columns
=============================

Sometimes you may want to iterate over all of the rows or columns in a data set.
This can be done quite easily::

  for row in d.rows():
      print(row)

  for column in d.columns():
      print(column)

If there is no mask set, then the first example could also have been written more compactly as::

  for row in d:
      print(row)

.. note::

    :py:meth:`~Stoner.core.data.Data.rows` and :py:meth:`~Stoner.core.data.Data.columns` both take an optional parameter *not_masked*. If this is True then these iterator
    methods will skip over any rows/columns with masked out data values. Direct iteration includes masked rows; use ``d.rows(not_masked=True)``
    when you want to skip rows containing masked cells.

Next steps
==========

Continue with :doc:`data_indexing` to read individual cells and slices.
To draw your measurement now, continue with :doc:`plotfile`.
Return to :doc:`datafile` for the full learning path.
