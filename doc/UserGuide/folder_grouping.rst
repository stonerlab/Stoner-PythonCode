Group folders and manage their hierarchy
****************************************

.. currentmodule:: Stoner.folders

Use ``f`` for a DataFolder. Groups are folders themselves, so they
can contain files and further groups. These operations organise the
collection in memory; they do not rearrange files on disc.

.. _groups:

Grouping
========

One of the more common tasks is to group a long list of data files into separate groups
according to some logical test --  for example gathering files with magnetic field sweeps in a positive
direction together and those with magnetic field in a negative direction together. The
:py:meth:`~Stoner.folders.mixins.DataFolder.group` method provides a powerful way to do this. Suppose we have a series of
data curves taken at a variety of temperatures and with three different magnetic fields::

   f.group('temperature')
   f.group(lambda x:"positive" if x['B-Field']>0 else "negative")
   f.group(['temperature',lambda x:"positive" if x['B-Field']>0 else "negative"])
   f2=f/'temperature'
   f/='temperature'
   f.groups

The :py:meth:`~Stoner.folders.mixins.DataFolder.group` method splits the files in the :py:class:`~Stoner.folders.mixins.DataFolder` into several
groups each of which share a common value of the argument supplied to the :py:meth:`~Stoner.folders.mixins.DataFolder.group`
method. A group is itself another instance of the :py:class:`~Stoner.folders.mixins.DataFolder` class. As explained above, each
:py:class:`~Stoner.folders.mixins.DataFolder` object maintains a dictionary called :py:attr:`~Stoner.folders.core.BaseFolder.groups` whose keys
are the distinct values of the argument of the :py:meth:`~Stoner.folders.mixins.DataFolder.group` methods and whose values are
:py:class:`~Stoner.folders.mixins.DataFolder` objects. So, if our :py:class:`~Stoner.folders.mixins.DataFolder` *f* contained files measured at
4.2, 77 and 300K and at fields of 1T and -1T then the first variant would create 3 groups: 4.2, 77 and
300 each one of which would be a :py:class:`~Stoner.folders.mixins.DataFolder` object containing the files measured at those
temperatures. The second variant would produce 2 groups -- ''positive'' containing the files measured with
magnetic field of 1T and ''negative'' containing the files measured at -1T. The third variant then goes
one stage further and would produce 3 groups, each of which in turn had 2 groups. The groups are accessed
via the :py:attr:`~Stoner.folders.core.BaseFolder.groups` attribute::

   f.groups[4.2].groups["positive"].files

would return a list of the files measured at 4.2K and 1T.

If you try indexing a :py:class:`~Stoner.folders.mixins.DataFolder` with a string it first checks to see if there is a matching group
with a key of the same string then :py:class:`~Stoner.folders.mixins.DataFolder` will return the
corresponding group. This allows a more compact navigation through an extended group structure::

    f.group(['project','sample','device']) # group will take a list
    f['ASF']['ASF038']['A'] # Successive indexing
    f['ASF','ASF038','A'] # index with a tuple

The last variant will index through multiple levels of groups and then index for a file with a matching name and
then finally index metadata in that file.

If you just want to create a new empty group in your :py:class:`~Stoner.folders.mixins.DataFolder`, you can use
the :py:meth:`~Stoner.folders.mixins.DataFolder.add_group` method::

    f.add_group("key_value")

which will create the new group with a key of ''key_value''.

Manipulating the File List in a Folder
======================================

The :py:meth:`~Stoner.folders.mixins.DataFolder.flatten` method will do the same as passing the *flat* keyword argument when creating the :py:class:`~Stoner.folders.mixins.DataFolder` - although
the search for folders on disk is recursive, the resulting :py:class:`~Stoner.folders.mixins.DataFolder` contains a flat list of files.

You can also use the :py:meth:`Stoner.folders.groups.GroupsDict.prune` - which is aliased as :py:meth:`~Stoner.folders.mixins.DataFolder.prune` method to remove
groups (including nested  groups) that have no data files in them. If you supply a *name* keyword to the
:py:meth:`Stoner.folders.groups.GroupsDict.prune` method it will instead remove any sub-folder with a matching name (and all sub-folders within it):

::

    Root---> (0 files)
         |
         |
         |-> A--> (0 files)
         |    |
         |    |--> B--> (5 files)
         |    |     |
         |    |     |--> C--> (0 files)
         |    |     |     |
         |    |     |     |--> D (0files)
         |    |     |
         |    |     |--> E--> (0 files)
         |    |
         |    |--> F--> (0 files)
         |
         |-->G--> (2 files)

**root.groups.prune()** will have the effect of removing sub-folders *C*, *D*, *E*, and *F*

::

    Root---> (0 files)
         |
         |
         |-> A--> (0 files)
         |    |
         |    |--> B--> (5 files)
         |
         |-->G--> (2 files)

**root.groups.prune(name="B")** will have the effect of removing sub-folders *B*, *C*, *D* and *E*

::

    Root---> (0 files)
         |
         |
         |-> A--> (0 files)
         |    |
         |    |--> F--> (0 files)
         |
         |-->G--> (2 files)

In contrast, the :py:meth:`Stoner.folders.groups.GroupsDict.keep` method will retain the tree branches that contain the groups that match the *name*
parameter. For example,

**root.groups.keep("B")** will have the effect of deleting everything except the folders *A*, *B*, *C*, *D* and *E*.

::

    Root---> (0 files)
         |
         |
         |-> A--> (0 files)
              |
              |--> B--> (5 files)
                    |
                    |--> C--> (0 files)
                    |     |
                    |     |--> D (0files)
                    |
                    |--> E--> (0 files)


The :py:meth:`Stoner.folders.groups.GroupsDict.compress` is useful when a :py:class:`~Stoner.folders.mixins.DataFolder` contains a chain of subfolders that have only one sub-folder in them - as can
result when reading one specific directory from a deep directory tree. The :py:meth:`~Stoner.folders.mixins.DataFolder.compress` method adjusts the virtual tree so that the
root group is at the first level that contains more than just a single sub-folder.

::

    Root---> (0 files)
         |
         |
         |-> A--> (0 files)
              |
              |--> B--> (0 files)
                    |
                    |--> C--> (5 files)

**root.groups.compress()** will reformat the :py:class:`~Stoner.folders.mixins.DataFolder` to:

::

    Root/A/B/C---> (5 files)

:py:meth:`Stoner.folders.groups.GroupsDict.compress` takes a keyword argument *keep_terminal* which will keep the final group if set to **True**. In the example above,
**root.compress(keep_terminal=True)** gives:

::

    Root/A/B--> (0 files)
            |
            |-->C--> (5 files)

You can also use the sorted filenames in a :py:class:`~Stoner.folders.mixins.DataFolder` to reconstruct the directory structure as
groups by using the :py:meth:`~Stoner.folders.mixins.DataFolder.unflatten` method. Alternatively the *invert* operator ~ will
flatten and unflatten a :py:class:`~Stoner.folders.mixins.DataFolder`::

    g=~f # Flatten (if f has groups) or unflatten (if f has no groups)
    f.unflatten()

.. note::

    The unary invert operator ~ will always create a new :py:class:`~Stoner.folders.mixins.DataFolder` before doing the :py:meth:`~Stoner.folders.mixins.DataFolder.flatten` or
    :py:meth:`~Stoner.folders.mixins.DataFolder.unflatten` - so that the original :py:class:`~Stoner.folders.mixins.DataFolder` is left unchanged. In contrast the
    :py:meth:`~Stoner.folders.mixins.DataFolder.flatten` and :py:meth:`~Stoner.folders.mixins.DataFolder.unflatten` methods will change the :py:class:`~Stoner.folders.mixins.DataFolder` as well as return the
    changed :py:class:`~Stoner.folders.mixins.DataFolder`.

If you need to combine multiple :py:class:`~Stoner.folders.mixins.DataFolder` objects or add :py:class:`Stoner.core.data.Data`
objects to an existing :py:class:`~Stoner.folders.mixins.DataFolder` then the arithmetic addition operator can be used::

    from Stoner import Data, DataFolder

    f2=DataFolder('/data/test1')
    f3=DataFolder('/data/test2')
    f=f2+f3

    f+=Data('/data/test3/special.txt')

This will firstly combine all the files and then recursively merge the groups. If each :py:class:`~Stoner.folders.mixins.DataFolder` instance has the same
groups, then they are merged with the addition operator.

.. note::

    Strictly, the last example is adding an instance of the :py:attr:`~Stoner.folders.core.BaseFolder.type` to the :py:class:`~Stoner.folders.mixins.DataFolder` - type checking
    is carried out to ensure that this is so.

Continue with :doc:`folder_reducing` to combine measurements or visit
groups with a custom analysis function.
