Combine measurements and process groups
***************************************

.. currentmodule:: Stoner.folders

Use ``f`` for a folder whose files have suitable column roles and metadata.
See :doc:`folder_grouping` to create groups before reducing them.

An important driver for the development of the :py:class:`~Stoner.folders.mixins.DataFolder` class has been to aid
data reduction tasks. The simplest form of data reduction would be to gather one or more
columns from each of a folder of files and return it as a single large table or matrix. This task is
easily accomplished by the :py:meth:`~Stoner.folders.mixins.DataFolder.gather` method::

    f.gather("X Data","Y Data")
    f.gather("X Data",["Ydata 1","Y Data 2"])
    f.gather()

In the first two forms you specify the x column and one or more y columns. In the third form, the
x and y columns are determined by the values from the :py:attr:`~Stoner.core.data.Data.setas` attribute.
(you can set the value of this attribute for all files in the :py:class:`~Stoner.folders.mixins.DataFolder` by setting the
``folder.each.setas`` attribute.)

A similar operation to :py:meth:`~Stoner.folders.mixins.DataFolder.gather` is to build a new set of data where each row corresponds
to a set of metadata values from each file in the :py:class:`~Stoner.folders.mixins.DataFolder`. This can be achieved with the
:py:meth:`~Stoner.folders.mixins.DataFolder.extract` method::

    f.extract(["Temperature","Angle","Other_metadata"])

The argument to the :py:meth:`~Stoner.folders.mixins.DataFolder.extract` method is a list of metadata values to be extracted from each file. The
metadata should be convertible to an array type so that it can be included in the final result matrix. Any metadata that doesn't
appear to be so convertible in the first file in the :py:class:`~Stoner.folders.mixins.DataFolder` is ignored. The column headings of the final results
table are the names of the metadata that were used in the extraction.

One task you might want to do would be to work through all the groups in a :py:class:`~Stoner.folders.mixins.DataFolder`
and run some function either with each file in the group or on the whole group. This is further
complicated if you want to iterate over all the sub-groups within a group. The
:py:meth:`~Stoner.folders.mixins.DataFolder.walk_groups` method is useful here::

     f.walk_groups(func,group=True,replace_terminal=True,walker_args={"arg1":"value1"})

This will iterate over the complete hierarchy of groups and sub groups in the folder and
execute the function *func* once for each group. If the *group* parameter is **False**
then it will execute *func* once for each file. The function *fun* should be defined something like::

    def func(group, group_keys, arg1="value1"):
        # Process this group; group_keys gives its path within the folder.
        return group

The first parameter should expect an instance of :py:class:`~Stoner.core.data.Data` if
*group* is **False** or an instance of :py:class:`~Stoner.folders.mixins.DataFolder` if *group* is **True**.
The second parameter receives a list of group keys from
the topmost group to the lowest (terminal) group.

By default, ``only_terminal=True`` visits terminal groups only. Set
``only_terminal=False`` to visit non-terminal groups too, after their subgroups
(or each group's direct members when ``group=False``). The option is honoured
at every depth. The starting folder receives an empty list of group keys.

The *replace_terminal* parameter applies when *group* is **True** and the function returns a
:py:class:`Stoner.core.data.Data` object. This indicates that the group on which the function was
called should be removed from the list of groups and the returned :py:class:`~Stoner.core.data.Data`
object should be added to the list of files in the folder. This operation is useful when one is
processing a group of files to combined them into a single dataset. Combining a multi-level grouping
operation and successive calls to :py:meth:`~Stoner.folders.mixins.DataFolder.walk_groups` can rapidly reduce a large set of
data files representing a multi-dimensional data set into a single file with minimal coding.


In some cases you will want to work with sets of files coming from different groups in order.
For example, if above we had a sequence of 10 data files for each field and temperature and we wanted
to process the positive and negative field curves together for a given temperature in turn.
In this case the :py:meth:`~Stoner.folders.mixins.DataFolder.zip_groups` method can be useful::

   f.groups[4.2].zip_groups(['positive','negative'])

This would return a list of tuples of :py:class:`~Stoner.core.data.Data` objects where the tuples
would be the first positive and first negative field files, then the second of each, then third of
each and so. This presupposes that the files started of sorted by some suitable parameter
(eg a gate voltage).

For another way to build a summary table, see :doc:`folder_metadata`.
