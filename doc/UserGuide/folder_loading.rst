Build folders and load files
****************************

.. currentmodule:: Stoner.folders

A folder lists matching files first and loads their data when needed.
Replace the illustrative directories and filename patterns below with
those of your measurements.

Building a (virtual) Folder of Data
===================================

The first thing you probably want to do is to get a list of data files in a directory
(possibly including its subdirectories) and probably matching some sort of filename pattern::

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
argument - otherwise the current directory will be used::

   f=DataFolder('/home/phygbu/Data',pattern='*.tdi')

If you pass False into the constructor as the first argument then the :py:class:`~Stoner.folders.mixins.DataFolder` will
display a dialog box to let you choose a directory. If you add the *multifile* keyword argument and set it to True
then you can use the dialog box to select multiple individual files.

More Options on Reading the Files on Disk
=========================================

The *pattern* argument for :py:class:`~Stoner.folders.mixins.DataFolder` can also take a list of multiple patterns if there are different filename types in the directory tree::

   f=DataFolder(pattern=['*.tdi','*.txt'])

Sometimes a more complex filename matching mechanism than simple ''globbing'' is useful.
The *pattern* keyword can also be a compiled regular expression::

   import re
   p=re.compile(r'i10-\d+\.dat')
   f=DataFolder(pattern=p)
   p2=re.compile(r'i10-(?P<run>\d+)')
   f=DataFolder(pattern=p2)
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
keyword argument and the file listing can be suppressed altogether with the *readlist=False* option::

   f=DataFolder(pattern='*.dat',recursive=False)
   f2=DataFolder(readlist=False)
   f3=DataFolder(flat=True)

If you don't want to create groups for each sub-directory, then set the keyword parameter
*flat* **True** as shown in the last example above.

Dealing With Revision Numbers
-----------------------------

The Leeds CM Physics LabVIEW measurement software (aka 'The One Code') has a feature that adds a *revision number* into the filename when it is asked to
overwrite a saved data file. This revision number is incremented until a non-colliding filename is created - thus ensuring that data isn't accidentally
overwritten. The downside of this is that sometimes only the latest revision number actually contains the most useful data - in this case the option
*discard_earlier* in the :py:meth:`~Stoner.folders.mixins.DataFolder.__init__` constructor can be useful, or equivalently the :py:meth:`~Stoner.folders.mixins.DataFolder.keep_latest` method::

    f=DataFolder(".",discard_earlier=True)
    # is equivalent to....
    f=DataFolder(".")
    f.keep_latest()

More DataFolder features
------------------------

Since a :py:class:`~Stoner.core.data.Data` represents data in named columns, the :py:class:`~Stoner.folders.mixins.DataFolder` offers a couple of additional options for actions to take
when reading the files in from disk. It is possible to have the mean and standard deviation of each column of data to be calculated and added as
metadata as each file is loaded. The *read_means* boolean parameter can enable this.

Other Options
=============

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
The :py:meth:`~Stoner.folders.mixins.DataFolder.getlist` method can be used to force a new listing of files::

   f.directory='/home/phygbu/Data'
   f.pattern='*.txt'
   f.getlist()

Continue with :doc:`folder_selecting` to inspect and choose members.
