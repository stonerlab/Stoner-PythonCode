Read and collect folder metadata
********************************

.. currentmodule:: Stoner.folders

The examples use ``f`` or ``fldr`` for a folder. Its metadata proxy
reads and writes metadata across the stored measurements.
See :doc:`folder_each` for applying the analysis that produces results.

Since each object inside a :py:class:`~Stoner.folders.mixins.DataFolder` will be some form of :py:class:`Stoner.core.base.metadataObject`, the :py:class:`~Stoner.folders.mixins.DataFolder`
provides a mechanism to access the combined metadata of all of the objects it is storing via a
:py:attr:`~Stoner.folders.core.BaseFolder.metadata` attribute. Like :py:attr:`~Stoner.folders.core.BaseFolder.each` this is actually a special class (in this case
:py:class:`Stoner.folders.metadata.MetadataProxy`) that manages the process of iterating over the contents of the :py:class:`~Stoner.folders.mixins.DataFolder` to get and set
metadata on the individual :py:class:`~Stoner.core.data.Data` objects.

Indexing the :py:attr:`~Stoner.folders.core.BaseFolder.metadata` will return an array of the requested metadata key, with one element from each data file in the
folder. If the metadata key is not present in all files, then the array is a masked array and the mask is set for the files where it
is missing::

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

Slicing includes only direct members by default. To include members at every
level of a grouped folder, pass ``recurse=True``::

    f.metadata.slice("Startupaxis-X", recurse=True, output="Data")
    f.slice_metadata("Startupaxis-X", recurse=True)

Recursive slicing uses ``walk_groups(..., only_terminal=False)``: subgroups
are visited depth-first in stored order, then the parent's direct members.
Member order is preserved. Output formats remain the same and no group-path
columns are added. Common-key selection and ``mask_missing`` apply across all
visited members; the folder hierarchy is unchanged.

As can be seen from these examples, :py:meth:`~Stoner.folders.metadata.MetadataProxy.slice` defaults to returning either a list of dictionaries
or, if *values_only* is True, a list of values. The *output* parameter can change this. The options for *output* are:

    -   "dict" or dict (the default if *values_only* is False)

        return a list of dictionary subsets of the metadata
    -   "list" or list (the default if *values_only* is True)

        return a list of values for each metadata item. If only one item of metadata is requested, it returns a one-dimensional list.
    -   "array" or np.array

        return a single array - like list above, but returns as a numpy array. This can create a 2D array from multiple keys
    -   "Data" or Stoner.Data

        returns numerical metadata in a Stoner.Data object, with the keys as column headers.
    -   "frame"

        returns a pandas DataFrame, suitable for text and mixed metadata as well as numbers.
    -   "smart"

        switch between dict and list depending whether there is one or more keys.

:py:meth:`~Stoner.folders.metadata.MetadataProxy.slice` searches for matching metadata names by string, including using *glob* patterns.

**root.metadata.slice("Model:*")** will return all metadata items in all files in the DataFolder that start with 'Model:'. Since one of the
common uses of DataFolder is to fit a series of data files with a model, :py:meth:`~Stoner.folders.metadata.MetadataProxy.slice` also accepts an
``lmfit.Model`` and uses it to pull out fitting parameters after a bulk fitting call::

    from lmfit.models import GaussianModel as Gaussian
    fldr.each.fit.lmfit(Gaussian, result=True)
    summary=fldr.metadata.slice(Gaussian,output="data")

Since :py:class:`Stoner.folders.metadata.MetadataProxy` implements :py:class:`collections.abc.MutableMapping`, it supplies the standard dictionary-like
methods such as :py:meth:`~Stoner.folders.metadata.MetadataProxy.keys`, :py:meth:`~Stoner.folders.metadata.MetadataProxy.values` and :py:meth:`~Stoner.folders.metadata.MetadataProxy.items`
- each of which work with the set of keys common to **all** the data files in the :py:class:`~Stoner.folders.mixins.DataFolder`. If you instead want to work with all the
keys defined in **any** of the data files, then there are versions :py:meth:`~Stoner.folders.metadata.MetadataProxy.all_keys`,
:py:meth:`~Stoner.folders.metadata.MetadataProxy.all_values` and :py:meth:`~Stoner.folders.metadata.MetadataProxy.all_items`. The :py:attr:`Stoner.folders.metadata.MetadataProxy.all`
provides a list of all the metadata dictionaries for all the data files in the :py:class:`~Stoner.folders.mixins.DataFolder`.

Using *output="Data"* is particularly powerful as it can gather the results of curve fitting across many data files into a
single :py:class:`Stoner.core.data.Data` object ready for plotting or further analysis::

    from Stoner import DataFolder
    from Stoner.analysis.fitting.models.generic import PowerLaw

    fldr=DataFolder(".",pattern="*.txt",setas="xy")
    fldr.each.fit.curve_fit(PowerLaw)
    result=fldr.metadata.slice(["Temperature:T1","PowerLaw:A","PowerLaw:A error"],output="Data")
    result.setas="xye"
    result.plot(fmt="k.")

In this example all the text files in the current directory tree are read in, a power-law is fitted to the first two columns and the result of the fit is
plotted versus a temperature parameter.
