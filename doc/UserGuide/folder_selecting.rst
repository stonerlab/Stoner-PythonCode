Inspect, sort and select folder members
***************************************

.. currentmodule:: Stoner.folders

Use ``f`` for a DataFolder created in :doc:`datafolder` or
:doc:`folder_loading`. Positions start at zero; string keys retrieve
members by name. Filtering can change the folder in place.

Getting a List of Files
=======================

To get a list of the names of the files in a :py:class:`~Stoner.folders.mixins.DataFolder`, you can use the :py:attr:`~Stoner.folders.core.BaseFolder.ls` attribute.
Sub-:py:class:`~Stoner.folders.mixins.DataFolder` s also have a name (essentially a string key to the dictionary that holds them), this can be accessed
via the :py:attr:`~Stoner.folders.core.BaseFolder.lsgrp` generator function::

    list(f.ls)
    list(f.lsgrp)

.. note::

    Both the :py:attr:`~Stoner.folders.core.BaseFolder.ls` and the :py:attr:`~Stoner.folders.core.BaseFolder.lsgrp` are generators, so they only return entries as they
    are iterated over. This is (roughly) in line with the Python 3 way of doing things - if you actually want the whole list
    then you should wrap them in a *list()*.

If you just need the actual filename part and not the directory portion of the filename, the generator :py:attr:`Stoner.folders.mixins.DiskBasedFolderMixin.basenames`
will do this.

As well as the list of filenames, you can get at the underlying stored objects through the :py:attr:`~Stoner.folders.core.BaseFolder.files` attribute.
This will return a list of either instances of the stored :py:class:`Stoner.core.data.Data` type if they have already been loaded
or the filename if they haven't been loaded into memory yet::

   f.files

The various subfolder are stored in a dictionary in the :py:attr:`~Stoner.folders.core.BaseFolder.groups` attribute.

   f.groups

Both the files and groups in a :py:class:`~Stoner.folders.mixins.DataFolder` can be accessed either by integer index or by name. If a string name is used
and doesn't exactly match, then it is interpreted as a regular expression and that is matched instead. This only applies for retrieving
items - for setting items an exact name or integer index is required.

Index individual files
======================

:py:class:`~Stoner.folders.mixins.DataFolder` is also indexable and has a length::

   len(f)
   f[0]
   f['filename']

For the second case of indexing, the code will search the list of file names for a matching file
and return the corresponding Data object. See :ref:`groups`
for creating a sub DataFolder with a named index.

Sorting
========
The order of the files in a :py:class:`~Stoner.folders.mixins.DataFolder` is arbitrary. If it is important to process
them in a given order then the :py:meth:`~Stoner.folders.mixins.DataFolder.sort` method can be used::

   f.sort()
   f.sort('Temperature')
   f.sort('Temperature',reverse=True)
   f.sort(lambda x:len(x))

The first variant simply sorts the files by filename. The second and third variants both
look at the ''temperature'' metadata in each file and use that as the sort key. In the
third variant, the *reverse* keyword is used to reverse the order of the sort. In the final
variant, each file is loaded in turn and the supplied function is called and evaluated to find a
sort key.

Filtering
=========

The :py:meth:`~Stoner.folders.mixins.DataFolder.filter` method can be used to prune the list of files used by the
:py:class:`~Stoner.folders.mixins.DataFolder`::

   f.filter('[ab]*.dat')
   import re
   f.filter(re.compile(r'i10-\d*\.dat'))
   f.filter(lambda x: x['Temperature']>150)
   f.filter(lambda x: x['Temperature']>150,invert=True,copy=True)
   f.filterout(lambda x: x['Temperature']>150,copy=True)

The first form performs the filter on the filenames (using the standard python fnmatch module).
One can also use a regular expression as illustrated in the second example -- although unlike using
the *pattern* keyword in :py:meth:`~Stoner.folders.mixins.DataFolder.getlist`, there is no option to capture metadata
(although one could then subsequently set the pattern to achieve this). The third variant calls the
supplied function, passing the current file as a :py:class:`~Stoner.core.data.Data` object in each time.
If the function evaluates to be **True** then the file is kept. The *invert* keyword is used to invert
the sense of the filter (a particularly silly example here, since the greater than sign could simply
be replaced with a less than or equals sign !). The *copy* keyword argument causes the :py:class:`~Stoner.folders.mixins.DataFolder` to
be duplicated before the duplicate is filtered - without this, the filtering will modify the current
:py:class:`~Stoner.folders.mixins.DataFolder` in place. Finally, the :py:meth:`~Stoner.folders.mixins.DataFolder.filterout` method is an alias for the :py:meth:`~Stoner.folders.mixins.DataFolder.filter`
method with the *invert* keyword set.

Selecting Data
==============

Selecting data from the :py:class:`~Stoner.folders.mixins.DataFolder` is somewhat similar to filtering, but allows an easy way to build complex selection rules
based on metadata values::

    f.select(temperature_T1=4.2)
    f.select(temperature_T1__gt=77.0)
    f.select(temperature__not__between=(4.5,8.2))
    f.select(user__contains="phygbu", project__contains="transport")
    f.select(user__contains="phygbu").select(project__icontains="superconduct")
    f.select({"temp:T1":4.2})

The basic pattern of the :py:meth:`~Stoner.folders.mixins.DataFolder.select` method is that each keyword argument determines both the name of the metadata to use
as the basis of the selection and also the operation to be performed. The value of the keyword argument is the value use to check. The operation is
separated from the column name by a double underscore.

In the first example, only those files with a metadata value "temperature_T1" which is 4.2 will be selected, here there is no operator specified,
so for a single scalar value it is assumed to be ''__eq'' for equals. For a tuple it would be ''__between'' and for a longer list ''__in''.
In the second example, the ''__gt'' (greater than) operator is used and in the third it is ''__between'', but in addition, this is inverted with
'__not''. The fourth option illustrates a test with metadata whose values are strings. In addition, the use of the two keyword arguments is the
logical OR of testing for either. The equivalent process for a logical AND is shown in the fifth example with successive selects (the ''__icontains''
operator is a case-insensitive match). The final example uses a dictionary passed as a non-keyword argument to show how to select metadata keys
that are not valid Python identifiers.

To organise selected members into subfolders, see :doc:`folder_grouping`.
