**************************
Working with Lots of Files
**************************
.. currentmodule:: Stoner.Folders


A common case is that you have measured lots of data curves and now have a large stack of data 
files sitting in a tree of folders on disc and now need to process all of them with some code. 
The :py:class:`DataFolder` class is designed to make it easier to process lots of files.

Building a (virtual) Folder of Data Files
=========================================

The first thing you probably want to do is to get a list of data files in a directory 
(possibly including its subdirectories) and probably matching some sort of filename pattern.::

   from Stoner.Folders import DataFolder
   f=DataFolder(pattern='*.dat')

In this very simple example, the :py:class:`DataFolder class is imported in the first line and 
then a new instance *f* is created. The optional *pattern* keyword is used to only collect 
the files with a .dat extension. In this example, it is assumed that the files are readable by 
:py:class:`{DataFile`, if they are in some other format then the *type* keyword can be used::

   from Stoner.FileFormats import XRDFile
   f=DataFolder(type=XRDFile,pattern='*.dql')

To specify a particular directory to look in, simply give the directory as the first 
argument - otherwise the current duirectory will be used.::

   f=DataFolder('/home/phygbu/Data',pattern='*.tdi')

If you pass False into the constructor as the first argument then the :py:class:`DataFolder` will
display a dialog box to let you choose a directory. If you add a 'multifile=True' keyword argument
then you can use the dialog box to select multiple individual files.

Any other keyword arguments that are not attributes of :py:class:`DataFolder` are instead passed to the
constructor for the individual :py:class:`Stoner.Core.DataFile` instances as they are loaded from disc. This,
for example, can allow one to set the default :py:attr:`Stoner.Core.DataFile.setas` attribute for each file.

By default the :py:class:`DataFolder` constructor will perform a recursive drectory listing of 
the working folder. Each sub-directory is given a separate *group* within the structure. 
This allows the :py:class:`DataFolder` to logically represent the on-disc layout of the files.

Manipulating the File List in a Folder
======================================

If you don't want the file listing to be recursive, this can be suppressed by using the *recursive*
 keyword argument and the file listing can be suppressed altogether with the *nolist* keyword::

   f=DataFolder(pattern='*.dat',recursive=False)
   f2=DataFolder(nolist=True)

If you don't want to create groups for each sub-directory, then set the keyword parameter 
*flatten* **True**, or call the :py:meth:`DataFolder.flatten` method. You can also use the 
:py:meth:`DataFolder.prune` method to remove groups (including nested groups) that have 
no data files in them.::

	f.prune()
	f.flatten()

If you need to combine multiple :py:class:`DataFolder` objects or add :py:class:`Stoner.Core.DataFile`
objects to an existing :py:ckass:`DataFolder` then the arithmetic addition operator can be used::

   f2=DataFolder('/data/test1')
   f3=DataFolder('/data/test2')
   f=f2+f3

    f+=DataFile('/data/test3/special.txt')


Getting a List of Files
========================

The resulting list of files can be accessed via the :py:attr:`DataFolder.files` attribute 
and sub groups with the :py:attr:`DataFolder.group` attribute (see :ref:`groups`::

   f.files
   f.groups

.. warning::
   In some circumstances entries in the :py:attr:`DataFolder.files` attribute can be 
   :py:class:`Stoner.Core.DataFile` objects rather than strings. If you want to ensure 
   that you get a list of strings representing the filenames, use :py:attr:`DataFolder.ls` instead.


Controlling the Gathering of the List of Files
==============================================

The current root directory and pattern are stored in the *directory* and *pattern* keywords and the
:py:meth:`DataFolder.getlist` method can be used to force a new listing of files.::

   f.dirctory='/home/phygbu/Data'
   f.pattern='*.txt'
   f.getlist()

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
In both cases above the :py:class:`DataFolder` has the same file members (basically these 
would be runs produced by the i10 beamline at Diamond), but in the second case the run 
number (which comes after ''i10-'' would be captured and presented as the ''run'' parameter in 
the metadata when the file was read. 

.. warning::
   Note that the files are not modified - the extra metadata is only added as the file is read by the \
   :py:class:`DataFlder`. 

The loading process will also add the metadata key ''Loaded From'' to the file which will give you a 
note of the filename used to read the data. If the attribute :py:attr:`DataFoilder.read_means` is set to **True**
then additional metadata is set for each file that contains the mean value of each column of data.

Doing Something With Each File
===============================

A :py:class:`DataFolder` is an object that you can iterate over, lading the :py:class:`Stoner.Core.DataFile`
 type object for each of the files in turn. This probides an easy way to run through a set of files, 
performing the same operation on each::

    folder=DataFolder(pattern='*.tdi')
    for f in folder:
        	f=AnalyseFile(f)
      	f.normalise('mac116','mac119')
     	f.save()

or even more compacts::

  [f.normalise('mac116','macc119').save() for f in DataFolder(pattern='*.tdi',type=AnalyseFile)]

:py:class:`DataFolder` is also indexable and has a length::

   f=DataFolder()
   len(f)
   f[0]
   f['filename']

For the second case of indexing, the cose will search the list of file names for a matching file 
and return that (roughly equivalent to doing ``f.files.index("filename")]``)

If you want to know the filename of all the files in the :py:class:`DataFolder` then there is a 
handy attributes::

    f.ls
    f.basenames

The difference between these two is that :py:attr:`DataFolder.basenames` will return only the file part 
of the filename whilst :py:attr:`DataFolder.ls` returns the complete path from the root directory.

.. _groups: Sorting, Filtering and Grouping Data Files
======================================================

The order of the files in a :py:class:`DataFolder` is arbitrary. If it is important to process 
them in a given order then the :py:meth:`DataFolder.sort` method can be used::

   f.sort()
   f.sort('tmperature')
   f.sort('Temperature',reverse=True)
   f.sort(lambda x:len(x))

The first variant simply sorts the files by filename. The second and third variants both 
look at the ''temperature'' metadata in each file and use that as the sort key. In the 
third variant, the *revers* keyword is used to reverse the order of the sort. In the final 
variant, each file is loaded in turn and the supplied function is called and evaluated to find a 
sort key.

The :py:meth:`DataFolder.filter` method can be used to prune the list of files to be used by the 
:py:class:`DataFoler`::

   f.filter('[ab]*.dat')
   import re
   f.filter(re.compile('i10-\d*\.dat'))
   f.filter(lambda x: x['Temperature']>150)
   f.filter(lambda x: x['Temperature']>150,invert=True)

The first form performs the filter on the filenames (using the standard python fnmatch module). 
One can also use a regular expression as illustrated int he second example -- although unlike using 
the *pattern* keyword in :py:meth:`DataFolder.getlist`, there is no option to capture metadata 
(although one could then subsequently set the pattern to achieve this). The third variant calls the 
supplied function, passing the current file as a :py:class:`Stoner.Core.DataFile` object in each time. 
If the function evaluates to be **True** then the file is kept. The *invert* keyword is used to invert 
the sense of the filter (a particularly silly example here, since the greater than sign could simply 
be replaced with a less than or equals sign !).

One of the more common tasks is to group a long list of data files into separate groups 
according to some logical test --  for example gathering files with magnetic field sweeps in a positive 
direction together and those with magnetic field in a negative direction together. The 
:py:meth:`DataFolder.group` method provides a powerful way to do this. Suppose we have a series of 
data curves taken at a variety of temperatures and with three different magnetic fields::

   f.group{'temperature'}
   f.group(lambda x:"positive" if x['B-Field']>0 else "negative")
   f.group(['temperature',lambda x:"positive" if x['B-Field']>0 else "negative"])
   f.groups

The :py:meth:`DataFolder.group` method splits the files in the :py:class:`DataFolder` into several 
groups each of which share a common value of the arguement supplied to the :py:meth:`DataFolder.group` 
method. A group is itself another instance of the :py:class:`DataFolder` class. Each 
:py:class:`DataFolder` object maintains a dictionary called :py:attr:`DataFolder.groups` whose keys 
are the distinct values of the argument of the :py:meth:`DataFolder.group` methods and whose values are 
:py:class:`DataFolder` objects. So, if our :py:class:`DataFolder` *f* contained files measured at 
4.2, 77 and 300K and at fields of 1T and -1T then the first variant would create 3 groups: 4.2, 77 and 
300 each one of which would be a :py:class:`DataFolder` object congaing the files measured at those 
temperatures. The second variant would produce 2 groups -- ''positive'' containing the files measured with 
magnetic field of 1T and ''negative'' containing the files measured at -1T. The third variant then goes 
one stage further and would produce 3 groups, each of which in turn had 2 groups. The groups are accessed 
via the :py:attr:`DataFolder.group` attribute::

   f.groups[4.2].groups["positive"].files

would return a list of the files measured at 4.2K and 1T.

If you try indexing a :py:class:`DataFolder` with a string and there is no file with as its filename 
and there is a group with a key of the same string then :py:clkass:`DataFolder` will return the 
corresponding group. This allows a more compact navigation through an extended group structure.::

    f.group(['project','sample','device'])
    f['ASF']['ASF038']['A']

If you just ant to create a new empty group in your :py:class:`DataFoler`, you can use 
the :py:meth:`DataFolder.add_group` method.::

`f.add_group("key_value")

which will create the new group with a key of ''key_value''.

Reducing Data
-------------

An important driver for the development of the :py:class:`DataFolder` class has been to aid
data reduction tasks. The simplest form of data reduction would be to gather one or more
columns from each of a folder of files and return it as a single large table or matrix. This task is
easily accomplished by the :py:meth:`DataFolder.gather` method::

    f.gather("X Data","Y Data")
    f.gather("X Data",["Ydata 1","Y Data 2"])
    f.gather()

In the first two forms you specify the x column and one or more y columns. In the third form, the
x and y columns are determined by the values from the :py:attr:`Stoner.Core.DataFile.setas` attribute.
(you can set the value of this attribute for all files in the :py:class:`DataFolder` by setting the 
:py:attr:`DataFolder.setas` attribute.)


One task you might want to do would be to work through all the groups in a :py:class:`DataFolder` 
and run some function either with each file in the group or on the whole group. This is further 
complicated if you want to iterate over all the sub-groups within a group. The 
:py:meth:`DataFolder.walk_groups` method is useful here.::

     f.walk_groups(func,group=True,replace_terminal=True,walker_args={"arg1":"value1"})

This will iterate over the complete hierarchy of groups and sub groups in the folder and 
execute the function *func* once for each group. If the *group* parameter is **False** 
then it will execute *func* once for each file. The function *fun* should be defined something like::

    def func(group,list_of-group_keys,arg1,arg2...)

The first parameter should expect and instance of :py:class:`Stoner.Core.DataFile` if 
*group* is **False** or an instance of :py:class:`DataFolder` if *group* is **True**. 
The second parameter will be given a list of of strings representing the group key values from 
the topmost group to the lowest (terminal) group.

The *replace_terminal* parameter applies when *group* is **True** and the function returns a 
:py:class:`Stoner.Core,DataFile` object. This indicates that the group on which the function was 
called should be removed from the list fo groups and the returned :py:class:`Stoner.Core.DataFile`
object should be added to the list of files in the folder. This operation is useful when one is 
processing a group of files to combined them into a single dataset. Combining a multi-level grouping 
operation and successive calls to :py:meth:`DataFolder.walk_groups` can rapidly reduce a large set of 
data files representing a multi-dimensional data set into a single file with minimal coding.


In some cases you will want to work with sets of files coming from different groups in order. 
For example, if above we had a sequence of 10 data files for each field and temperature and we wanted 
to process the positive and negative field curves together for a given temperature in turn. 
In this case the :py:meth:`DataFolder.zip_groups` method can be useful.::

   f.groups[4.2].zip_groups(['positive','negative'])

This would return a list of tuples of :py:class:`Stoner.Core.DataFile` objects where the tuples 
would be the first positive and first negative field files, then the second of each, then third of 
each and so. This presupposes that the files started of sorted by some suitable parameter 
(eg a gate voltage).
