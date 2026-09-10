**************************
Working with Lots of Files
**************************
.. currentmodule:: Stoner.folders


A common case is that you have measured lots of data and now have a large stack of data
files sitting in a tree of directories on disc and need to process all of them with some code.
The :py:mod:`Stoner.folders` contains classes to make this job much easier.

For the end-user, the top-level classes are :py:class:`Stoner.folders.mixins.DataFolder` for :py:class:`Stoner.core.data.Data` and
:py:class:`Stoner.Image.folders.ImageFolder` for collections of :py:class:`Stoner.Image.core.ImageFile` objects. These complement the corresponding data classes.
Both :py:class:`~Stoner.core.data.Data` and :py:class:`~Stoner.folders.mixins.DataFolder` are exported directly from the ``Stoner`` package, whilst
:py:class:`~Stoner.Image.folders.ImageFolder` is exported from the :py:mod:`Stoner.Image` subpackage.

:py:class:`~Stoner.folders.mixins.DataFolder` and its friends are essentially containers for :py:class:`Stoner.core.data.Data` (or similar classes from the
:py:mod:`Stoner.Image` package) and for other instances of :py:class:`~Stoner.folders.mixins.DataFolder` to allow a nested hierarchy to be built up.
The :py:class:`~Stoner.folders.mixins.DataFolder` supports both sequence-like and mapping-like interfaces to both the :py:class:`Stoner.core.data.Data` objects and the
'sub'-:py:class:`~Stoner.folders.mixins.DataFolder` objects (meaning that they work like both a list or a dictionary).
:py:class:`~Stoner.folders.mixins.DataFolder` is also lazy about loading files from disc - if an operation doesn't need to load a file it generally won't bother to keep memory usage
down and speed up.

There are further variants that can work with compressed ZIP archives, :py:class:`Stoner.folders.zip.ZipFolder`, or store multiple files in a single HDF5 file,
:py:class:`Stoner.folders.hdf5.HDF5Folder`.

Finally, for the case of image files, there is a specialised :py:class:`Stoner.Image.ImageStack` class that is optimised for image files of the same dimension
and stores the images in a single 3D numpy array to allow much faster operations (at the expense of taking more RAM).

In the documentation below, except where noted explicitly, you can use a :py:class:`Stoner.Image.ImageFolder` in place of the :py:class:`~Stoner.folders.mixins.DataFolder`, but working
with :py:class:`Stoner.Image.ImageFile` instead of :py:class:`~Stoner.core.data.Data`.

Basic Operations
================

Building a (virtual) Folder of Data
-----------------------------------

The first thing you probably want to do is to get a list of data files in a directory
(possibly including its subdirectories) and probably matching some sort of filename pattern.::

   from Stoner import DataFolder
   f=DataFolder(pattern='*.dat')

In this very simple example, the :py:class:`~Stoner.folders.mixins.DataFolder` class is imported in the first line and
then a new instance *f* is created. The optional *pattern* keyword is used to only collect
the files with a .dat extension. In this example, it is assumed that the files are readable by
the :py:class:`Stoner.core.data.Data` general class. Format loaders are selected automatically from the file contents and suffix::

   f=DataFolder(pattern='*.dql')

Strictly, the class supplied with the *type* keyword should be a subclass of :py:class:`Stoner.core.base.metadataObject`
and should have a constructor that understands the initial string parameter to be a filename to load the object from. The class
is then available via the :py:attr:`~Stoner.folders.core.BaseFolder.type` attribute and a default instance of the class is available via the
:py:attr:`~Stoner.folders.core.BaseFolder.instance` attribute.

Additional parameters needed for the class's constructor can be passed via a dictionary to the *extra_args* keyword of the
:py:class:`~Stoner.folders.mixins.DataFolder` constructor.

To specify a particular directory to look in, simply give the directory as the first
argument - otherwise the current directory will be used.::

   f=DataFolder('/home/phygbu/Data',pattern='*.tdi')

If you pass False into the constructor as the first argument then the :py:class:`~Stoner.folders.mixins.DataFolder` will
display a dialog box to let you choose a directory. If you add the *multifile* keyword argument and set it to True
then you can use the dialog box to select multiple individual files.

More Options on Reading the Files on Disk
-----------------------------------------

The *pattern* argument for :py:class:`~Stoner.folders.mixins.DataFolder` can also take a list of multiple patterns if there are different filename types in the directory tree.::

   f=DataFolder(pattern=['*.tdi',*/txt'])

Sometimes a more complex filename matching mechanism than simple ''globbing'' is useful.
The *pattern* keyword can also be a compiled regular expression::

   import re
   p=re.compile('i10-\d*.dat')
   f=DataFolder(pattern=p)
   p2=re.compile('i10-(?P<run>\d*)')
   f=DataFolder(pattern=p)
   f[0]['run']

The second case illustrates a useful feature of regular expressions - they can be used to capture
parts of the matched pattern -- and in the python version, one can name the capturing groups.
In both cases above the :py:class:`~Stoner.folders.mixins.DataFolder` has the same file members (basically these
would be runs produced by the i10 beamline at Diamond), but in the second case the run
number (which comes after ''i10-'' would be captured and presented as the *run* parameter in
the metadata when the file was read.

.. warning::
   Note that the files are not modified - the extra metadata is only added as the file is read by the \
   :py:class:`~Stoner.folders.mixins.DataFolder`.

The loading process will also add the metadata key ''Loaded From'' to the file which will give you a
note of the filename used to read the data. If the attribute ``read_means`` is set to **True**
then additional metadata is set for each file that contains the mean value and standard deviation of each column of data.
If you don't want the file listing to be recursive, this can be suppressed by using the *recursive*
keyword argument and the file listing can be suppressed altogether with the *nolist* keyword.::

   f=DataFolder(pattern='*.dat',recursive=False)
   f2=DataFolder(readlist=False)
   f3=DataFolder(flat=True)

If you don't want to create groups for each sub-directory, then set the keyword parameter
*flat* **True** as shown in the last example above.

Dealing With Revision Numbers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Leeds CM Physics LabVIEW maeasurement software (aka 'The One Code') has a feature that adds a *revision number* into the filename when it is asked to
overwrite a saved data file. This revision number is incremented until a non-colliding filename is created - thus ensuring that data isn't accidentally
overwritten. The downside of this is that sometimes only the latest revision number actually contains the most useful data - in this case the option
*discard_earlier* in the :py:meth:`~Stoner.folders.mixins.DataFolder.__init__` constructor can be useful, or equivalently the :py:meth:`~Stoner.folders.mixins.DataFolder.keep_latest` method::

    f=DataFolder(".",discard_earlier=true)
    # is equivalent to....
    f=DataFolder(".")
    f.keep_latest()

More DataFolder features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since a :py:class:`~Stoner.core.data.Data` represents data in named columns, the :py:class:`~Stoner.folders.mixins.DataFolder` offers a couple of additional options for actions to take
when reading the files in from disk. It is possible to have the mean and standard deviation of each column of data to be calculated and added as
metadata as each file is loaded. The *read_means* boolean parameter can enable this.

Other Options
-------------

Setting the *debug* parameter will cause additional debugging information to be sent as the code runs.

Any other keyword arguments that are not attributes of :py:class:`~Stoner.folders.mixins.DataFolder` are instead kept and used to set
attributes on the individual :py:class:`~Stoner.core.data.Data` instances as they are loaded from disc. This,
for example, can allow one to set the default :py:attr:`~Stoner.core.data.Data.setas` attribute for each file.

.. note::

    A particularly useful parameter to set in the DataFolder constructor is the *setas* parameter - this will ensure that the :py:attr:`Stoner.core.data.Data.setas`
    attribute is set to identify columns of data as x, y etc. as the data files are loaded into the folder - thus allowing subsequent calls to
    :py:class:`~Stoner.core.data.Data` methods to run without needing to explicitly set the columns each time.

All of these keywords to the constructor will set corresponding attributes on the created :py:class:`~Stoner.folders.mixins.DataFolder`, so it is possible to redo the
process of reading the list of files from disk by directly manipulating these attributes.

The current root directory and pattern are set in the *directory* and *pattern* keywords and stored in the similarly named attributes.
The :py:meth:`~Stoner.folders.mixins.DataFolder.getlist` method can be used to force a new listing of files.::

   f.directory='/home/phygbu/Data'
   f.pattern='*.txt'
   f.getlist()


Manipulating the File List in a Folder
--------------------------------------

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

**root.groups.prune(name="B")** will have the effect of removing sub-folders *C*, *D*, and *F*

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


The :py:meth:`Stoner.folders.groups.GroupsDict.compress` is useful when a :py:class:`~Stoner.folders.mixins.DataFolder` contains a chain of sub-folers that have only one sub-folder in them - as can
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

**root.groups.compress** will reformat the :py:class:`~Stoner.folders.mixins.DataFolder` to:

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
    f.unlatten()

.. note::

    The unary invert operator ~ will always create a new :py:class:`~Stoner.folders.mixins.DataFolder` before doing the :py:meth:`~Stoner.folders.mixins.DataFolder.flatten` or
    :py:meth:`~Stoner.folders.mixins.DataFolder.unflatten` - so that the original :py:class:`~Stoner.folders.mixins.DataFolder` is left unchanged. In contrast the
    :py:meth:`~Stoner.folders.mixins.DataFolder.flatten` and :py:meth:`~Stoner.folders.mixins.DataFolder.unflatten` methods will change the :py:class:`~Stoner.folders.mixins.DataFolder` as well as return a
    copy of the changed :py:class:`~Stoner.folders.mixins.DataFolder`.

If you need to combine multiple :py:class:`~Stoner.folders.mixins.DataFolder` objects or add :py:class:`Stoner.core.data.Data`
objects to an existing :py:class:`~Stoner.folders.mixins.DataFolder` then the arithmetic addition operator can be used::

    f2=DataFolder('/data/test1')
    f3=DataFolder('/data/test2')
    f=f2+f3

    f+=Data('/data/test3/special.txt')

This will firstly combine all the files and then recursively merge the groups. If each :py:class:`~Stoner.folders.mixins.DataFolder` instance has the same
groups, then they are merged with the addition operator.

.. note::

    Strictly, the last example is adding an instance of the :py:attr:`~Stoner.folders.core.BaseFolder.type` to the :py:class:`~Stoner.folders.mixins.DataFolder` - type checking
    is carried out to ensure that this is so.


Getting a List of Files
-----------------------

To get a list of the names of the files in a :py:class:`~Stoner.folders.mixins.DataFolder`, you can use the :py:attr:`~Stoner.folders.core.BaseFolder.ls` attribute.
Sub-:py:class:`~Stoner.folders.mixins.DataFolder` s also have a name (essentially a string key to the dictionary that holds them), this can be accessed
via the :py:attr:`~Stoner.folders.core.BaseFolder.lsgrp` generator fumnction.::

    list(f.ls)
    list(f.lsgrp)

.. note::

    Both the :py:attr:`~Stoner.folders.core.BaseFolder.ls` and the :py:attr:`~Stoner.folders.core.BaseFolder.lsgrp` are generators, so they only return enties as they
    are iterated over. This is (roughly) in line with the Python 3 way of doing things - if you actually want the whole list
    then you should wrap them in a *list()*.

If you just need the actual filename part and not the directory portion of the filename, the generator :py:attr:`Stoner.folders.mixins.DiskBasedFolderMixin.basenames`
will do this.

As well as the list of filenames, you can get at the underlying stored objects through the :py:attr:`~Stoner.folders.core.BaseFolder.files` attribute.
This will return a list of either instances of the stored :py:class:`Stoner.core.data.Data` type if they have already been loaded
or the filename if they haven't been loaded into memory yet.::

   f.files

The various subfolder are stored in a dictionary in the :py:attr:`~Stoner.folders.core.BaseFolder.groups` attribute.

   f.groups

Both the files and groups in a :py:class:`~Stoner.folders.mixins.DataFolder` can be accessed either by integer index or by name. If a string name is used
and doesn't exactly match, then it is interpreted as a regular expression and that is matched instead. This only applies for retrieving
items - for setting items an exact name or integer index is required.


Doing Something With Each File
==============================

A :py:class:`~Stoner.folders.mixins.DataFolder` is an object that you can iterate over, loading the :py:class:`Stoner.core.data.Data`
type object for each of the files in turn. This provides an easy way to run through a set of files,
performing the same operation on each::

    folder=DataFolder(pattern='*.tdi',type=Stoner.Data)
    for f in folder:
      	f.normalise('mac116','mac119')
     	f.save()

or even more compacts::

  [f.normalise('mac116','macc119').save() for f in DataFolder(pattern='*.tdi',type=Stoner.Data)]

of even (!)::

    DataFolder(pattern='*.tdi',type=Stoner.Data).each.normalise('mac116','mac119').save()

This last example illustrates a special ability of a :py:class:`~Stoner.folders.mixins.DataFolder` to use the methods of the
type of :py:class:`~Stoner.core.data.Data` inside the DataFolder. The special :py:attr:`~Stoner.folders.core.BaseFolder.each` attribute (which is actually a
:py:class:`Stoner.folders.each.Item`) provides special hooks to let you call methods of the underlying :py:attr:`~Stoner.folders.core.BaseFolder.type` class on each
file in the :py:class:`~Stoner.folders.mixins.DataFolder` in turn. When you access a method on :py:attr:`~Stoner.folders.core.BaseFolder.each` that
is actually a method of the DataFile, they call a method that wraps a call to each :py:class:`~Stoner.core.data.Data` in turn. If the method
on :py:class:`~Stoner.core.data.Data` returns the :py:class:`~Stoner.core.data.Data` back, then this is stored in the :py:class:`~Stoner.folders.mixins.DataFolder`. In this case the result back`
to the user is the revised :py:class:`~Stoner.folders.mixins.DataFolder`. If, on the otherhand, the method when executed on the :py:class:`~Stoner.core.data.Data` returns some other
return value, then the user is returned a list of all of those return values. For example::

    newT=np.linspace(1.4,10,100)
    folder=DataFolder(pattern="*.txt",type=Stoner.Data)
    ret=folder.each.interpolate(newT,xcol="Temp",replace=True)
    # ret will be a copy of folder as Data,interpolate returns a copy of itself.

    ret=folder.each.span("Resistance")
    # ret is a list of tuples as the return value of Data.span() is a tuple

What happens if the analysis routine you want to run through all the items in :py:class:`~Stoner.folders.mixins.DataFolder` is not a method of the :py:class:`Stoner.core.data.Data`
class, but a function written by you? In this case, so long as you write your custom analysis function so that the first positional argument
is the :py:class:`~Stoner.core.data.Data` to be analysed, then the following syntax can be used::

    def my_analysis(data,arg1,arg2,karg=True)
        """Some sort of analysis function with some arguments and keyword argument that works
        on some data *data*."""
        return data.modified()

    f.each(my_analysis,arg1,arg2,karg=False)

(or alternatively using the matrix multiplication operator @)::

    (my_analysis@f)(arg1,arg2,karg=False)

*(my_analysis@f)* creates the callable object that iterates *my_analysis* over f, the second set of parenthesis above just calls this iterating object.

If the return value of the function is another instance of :py:class:`~Stoner.core.data.Data` (or whatever is being stored as the items in the
:py:class:`~Stoner.folders.mixins.DataFolder`) then it will replace the items inside the :py:class:`~Stoner.folders.mixins.DataFolder`. The call to :py:attr:`~Stoner.folders.core.BaseFolder.each` will also return a
simple list of the return values. If the function returns something else, then you can have it added to the metadata of each item in the
:py:class:`~Stoner.folders.mixins.DataFolder` by adding a *_return* keyword that can either be True to use the function name as the metadata name or a string to specify
the name of the metadata to store the return value explicitly.

Thus, if your analysis function calculates some parameter that you want to call *beta* you might use the following::

    f=DataFolder(",",pattern="*.txt")
    f.each(my_analysis,arg1,arg2,karg=False,_return="beta")

:py:class:`~Stoner.folders.mixins.DataFolder` is also indexable and has a length::

   f=DataFolder()
   len(f)
   f[0]
   f['filename']

For the second case of indexing, the code will search the list of file names for a matching file
and return that (roughly equivalent to doing *f.files.index("filename")]*) But see :ref:`groups`
for creating a sub DataFolder with a named index.

Working on the Metadata
=======================

Since each object inside a :py:class:`~Stoner.folders.mixins.DataFolder` will be some form of :py:class:`Stoner.core.base.metadataObject`, the :py:class:`~Stoner.folders.mixins.DataFolder`
provides a mechanism to access the combined metadata of all of the objects it is storing via a
:py:attr:`~Stoner.folders.core.BaseFolder.metadata` attribute. Like :py:attr:`~Stoner.folders.core.BaseFolder.each` this is actually a special class (in this case
:py:class:`Stoner.folders.metadata.MetadataProxy`) that manages the process of iterating over the contents of the :py:class:`~Stoner.folders.mixins.DataFolder` to get and set
metadata on the individual :py:class:`~Stoner.core.data.Data` objects.

Indexing the :py:attr:`~Stoner.folders.core.BaseFolder.metadata` will return an array of the requested metadata key, with one element from each data file in the
folder. If the metadata key is not present in all files, then the array is a masked array and the mask is set for the files where it
is missing.::

    f.metadata["Info.Sample_Material"]
    >>> masked_array(data=['Er', --, 'None', 'FeNi'],
             mask=[False,  True, False, False],
       fill_value='N/A',
            dtype='<U4')

Writing to the contents of :py:attr:`~Stoner.folders.core.BaseFolder.metadata` simply sets the corresponding metadata value on all the files in the folder::

    f.metadata["test"]=12.56
    f.metadata["test"]
    >>> array([12.56, 12.56, 12.56, 12.56])

The :py:meth:`Stoner.folders.metadata.MetadataProxy.slice` method provides more control over how the metadata stored in the data folder can be returned::

    f.metadata.slice("Startupaxis-X")
    >>> [{'Startupaxis-X': 2},
         {'Startupaxis-X': 2},
         {'Startupaxis-X': 2},
         {'Startupaxis-X': 2}]

    f.metadata.slice(["Startupaxis-X","Datatype,Comment"])
    >>> [{'Datatype,Comment': 1, 'Startupaxis-X': 2},
         {'Startupaxis-X': 2, 'Datatype,Comment': 1},
         {'Datatype,Comment': 1, 'Startupaxis-X': 2},
         {'Datatype,Comment': 1, 'Startupaxis-X': 2}]
    f.metadata.slice("Startupaxis-X",values_only=True)
    >>> [2, 2, 2, 2]

    f.metadata.slice("Startupaxis-X",output="Data")
    >>>
    ==========================  ===============
    TDI Format 1.5                Startupaxis-X
    index                                     0
    ==========================  ===============
    Stoner.class{String}= Data                2
                                          2
                                          2
                                          2
    ==========================  ===============

As can be seen from these examples, :py:meth:`~Stoner.folders.metadata.MetadataProxy.slice` defaults to returning either a list of dictionaries
or, if *values_only* is True, a list of values. The *output* parameter can change this. The options for *output* are:

    -   "dict" or dict (the default if *values_only* is False)

        return a list of dictionary subsets of the metadata
    -   "list" or list (the default if *values_only* is True)

        return a list of values for each metadata item. If only one item of metadata is requested, it returns a one-dimensional list.
    -   "array" or np.array

        return a single array - like list above, but returns as a numpy array. This can create a 2D array from multiple keys
    -   "Data" or Stoner.Data

        returns the metadata in a Stoner.Data object where the column headers are the metadata keys.
    -   "smart"

        switch between dict and list depending whether there is one or more keys.

:py:meth:`~Stoner.folders.metadata.MetadataProxy.slice` searches for matching metadata names by string, including using *glob* patterns.

**root.metadata.slice("Model:*")** will return all metadata items in all files in the DataFolder that start with 'Model:'. Since one of the
common uses of DataFolder is to fit a series of data files with a model, :py:meth:`~Stoner.folders.metadata.MetadataProxy.slice` also accepts an
``lmfit.Model`` and uses it to pull out the fitting parameters after using ``fldr.each.curve_fit`` or a similar method::

    from Stoner.analysis.fitting.models.generic import Gaussian
    fldr.each.lmfit(Gaussian,result=True)
    summary=fldr.metadata.slice(Gaussian,output="data")

Since :py:class:`Stoner.folders.metadata.MetadataProxy` implements :py:class:`collections.abc.MutableMapping`, it supplies the standard dictionary-like
methods such as :py:meth:`~Stoner.folders.metadata.MetadataProxy.keys`, :py:meth:`~Stoner.folders.metadata.MetadataProxy.values` and :py:meth:`~Stoner.folders.metadata.MetadataProxy.items`
- each of which work with the set of keys common to **all** the data files in the :py:class:`~Stoner.folders.mixins.DataFolder`. If you instead want to work with all the
keys defined in **any** of the data files, then there are versions :py:meth:`~Stoner.folders.metadata.MetadataProxy.all_keys`,
:py:meth:`~Stoner.folders.metadata.MetadataProxy.all_values` and :py:meth:`~Stoner.folders.metadata.MetadataProxy.all_items`. The :py:attr:`Stoner.folders.metadata.MetadataProxy.all`
provides a list of all the metadata dictionaries for all the data files in the :py:class:`~Stoner.folders.mixins.DataFolder`.

Using *output="Data"* is particularly powerful as it can gather the results of curve fitting across many data files into a
single :py:class:`Stoner.core.data.Data` object ready for plotting or further analysis::

    fldr=DataFolder(".",pattern="*.txt",setas="xy")
    fldr.each.curve_fit(PowerLaw)
    result=fldr.metadata.slice(["Temperature:T1","PowerLaw:A","PowerLaw:A error"],output="Data")
    result.setas="xye"
    result.plot(fmt="k.")

In this example all the text files in the current directory tree are read in, a power-law is fitted to the first two columns and the result of the fit is
plotted versus a temperature parameter.

.. _groups:

Sorting, Filtering and Grouping Data Files
==========================================

Sorting
-------
The order of the files in a :py:class:`~Stoner.folders.mixins.DataFolder` is arbitrary. If it is important to process
them in a given order then the :py:meth:`~Stoner.folders.mixins.DataFolder.sort` method can be used::

   f.sort()
   f.sort('tmperature')
   f.sort('Temperature',reverse=True)
   f.sort(lambda x:len(x))

The first variant simply sorts the files by filename. The second and third variants both
look at the ''temperature'' metadata in each file and use that as the sort key. In the
third variant, the *revers* keyword is used to reverse the order of the sort. In the final
variant, each file is loaded in turn and the supplied function is called and evaluated to find a
sort key.

Filtering
---------

The :py:meth:`~Stoner.folders.mixins.DataFolder.filter` method can be used to prune the list of files used by the
:py:class:`~Stoner.folders.mixins.DataFolder`::

   f.filter('[ab]*.dat')
   import re
   f.filter(re.compile('i10-\d*\.dat'))
   f.filter(lambda x: x['Temperature']>150)
   f.filter(lambda x: x['Temperature']>150,invert=True,copy=True)
   f.filterout(lambda x: x['Temperature']>150,copy=True)

The first form performs the filter on the filenames (using the standard python fnmatch module).
One can also use a regular expression as illustrated int he second example -- although unlike using
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
--------------

Selecting data from the :py:class:`~Stoner.folders.mixins.DataFolder` is somewhat similar to filtering, but allows an east way to build complex selection rules
based on metadata values.::

    f.setlect(temperature_T1=4.2)
    f.select(temperature_T1__gt=77.0)
    f.select(tmemperature__not__between(4.5,8.2))
    f.select(user__contains="phygbu",user__contains="phyma")
    f.select(user__contains="phygbu").select(project__icontains="superconduct")
    f.select({"temp:T1":4.2})

The basic pattern of the :py:meth:`~Stoner.folders.mixins.DataFolder.select` method is that each keyword argument determines both the name of the metadata to use
as the asis of the selection and also the operation to be performed. The value of the keyword argument is the value use to check. The operation is
separated from the column name by a double underscore.

In the first example, only those files with a metadata value "temperature_T1" which is 4.2 will be selected, here there is no operator specified,
so for a single scalar value it is assumed to be ''__eq'' for equals. For a tuple it would be ''__between'' and for a longer list ''__in''.
In the second example, the ''__gt'' (greater than) operator is used and in the third it is ''__between'', but in addition, this is inverted with
'__not''. The fourth option illustrates a test with memtadata whose values are strings. In addition, the use of the two keyword arguments is the
logical OR of testing for either. The equiavblant process for a logical AND is shown in the sixth example with successive selects (the ''__icontains''
operator is a case insenesitive match). The final example uses a dictionary passed as a non-keyword argument to show how to select memtadata keys
that are not valid Python identifiers.

Grouping
--------

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
300 each one of which would be a :py:class:`~Stoner.folders.mixins.DataFolder` object congaing the files measured at those
temperatures. The second variant would produce 2 groups -- ''positive'' containing the files measured with
magnetic field of 1T and ''negative'' containing the files measured at -1T. The third variant then goes
one stage further and would produce 3 groups, each of which in turn had 2 groups. The groups are accessed
via the :py:meth:`~Stoner.folders.mixins.DataFolder.group` attribute::

   f.groups[4.2].groups["positive"].files

would return a list of the files measured at 4.2K and 1T.

If you try indexing a :py:class:`~Stoner.folders.mixins.DataFolder` with a string it first checks to see if there is a matching group
with a key of the same string then :py:class:`~Stoner.folders.mixins.DataFolder` will return the
corresponding group. This allows a more compact navigation through an extended group structure.::

    f.group(['project','sample','device']) # group will take a list
    f['ASF']['ASF038']['A'] # Succsive indexing
    f['ASF','ASF038','A'] # index with a tuple

The last variant will index through multiple levels of groups and then index for a file with a matching name and
then finally index metadata in that file.

If you just want to create a new empty group in your :py:class:`~Stoner.folders.mixins.DataFolder`, you can use
the :py:meth:`~Stoner.folders.mixins.DataFolder.add_group` method.::

    f.add_group("key_value")

which will create the new group with a key of ''key_value''.

Reducing Data
=============

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
:py:attr:`~Stoner.folders.core.BaseFolder.setas` attribute.)

A similar operation to :py:meth:`~Stoner.folders.mixins.DataFolder.gather` is to build a new set of data where each row corresponds
to a set of metadata values from each file in the :py:class:`~Stoner.folders.mixins.DataFolder`. This can be achieved with the
:py:meth:`~Stoner.folders.mixins.DataFolder.extract` method.::

    f.extract(["Temperature","Angle","Other_metadata"])

The argument to the :py:meth:`~Stoner.folders.mixins.DataFolder.extract` method is a list of metadata values to be extracted from each file. The
metadata should be convertible to an array type so that it can be included in the final result matrix. Any metadata that doesn't
appear to be so convertible in the first file in the ;py:class:`DataFolder` is ignored. The column headings of the final results
table are the names of the metadata that were used in the extraction.

One task you might want to do would be to work through all the groups in a :py:class:`~Stoner.folders.mixins.DataFolder`
and run some function either with each file in the group or on the whole group. This is further
complicated if you want to iterate over all the sub-groups within a group. The
:py:meth:`~Stoner.folders.mixins.DataFolder.walk_groups` method is useful here.::

     f.walk_groups(func,group=True,replace_terminal=True,walker_args={"arg1":"value1"})

This will iterate over the complete hierarchy of groups and sub groups in the folder and
execute the function *func* once for each group. If the *group* parameter is **False**
then it will execute *func* once for each file. The function *fun* should be defined something like::

    def func(group,list_of-group_keys,arg1,arg2...)

The first parameter should expect and instance of :py:class:`~Stoner.core.data.Data` if
*group* is **False** or an instance of :py:class:`~Stoner.folders.mixins.DataFolder` if *group* is **True**.
The second parameter will be given a list of of strings representing the group key values from
the topmost group to the lowest (terminal) group.

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
In this case the :py:meth:`~Stoner.folders.mixins.DataFolder.zip_groups` method can be useful.::

   f.groups[4.2].zip_groups(['positive','negative'])

This would return a list of tuples of :py:class:`~Stoner.core.data.Data` objects where the tuples
would be the first positive and first negative field files, then the second of each, then third of
each and so. This presupposes that the files started of sorted by some suitable parameter
(eg a gate voltage).
