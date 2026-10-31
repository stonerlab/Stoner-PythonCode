Apply operations to every file
******************************

.. currentmodule:: Stoner.folders

Use ``f`` or ``folder`` for a DataFolder. The ``each`` helper applies
a method or function to its members and collects their results.

A :py:class:`~Stoner.folders.mixins.DataFolder` is an object that you can iterate over, loading the :py:class:`Stoner.core.data.Data`
type object for each of the files in turn. This provides an easy way to run through a set of files,
performing the same operation on each::

    from Stoner import DataFolder

    folder = DataFolder(pattern="*.tdi")
    for f in folder:
        f.normalise('mac116','mac119')
        f.save()

To apply a method to every file, use ``each``::

    folder = DataFolder(pattern="*.tdi", setas="xy")
    folder.each.normalise("mac116", "mac119")
    folder.each.save()

Each call returns a list of results, one per file; it does not return the folder
for chaining. Methods such as ``normalise`` change the stored objects. Methods
returning a new Data object replace the corresponding folder member. Other
results, such as a span or a plot's Axes, are collected in the returned list::

    import numpy as np

    new_temperatures = np.linspace(1.4, 10, 100)
    results = folder.each.interpolate(new_temperatures, xcol="Temp", replace=True)
    spans = folder.each.span("Resistance")  # A list of (minimum, maximum) pairs.

If an individual call fails, its list entry contains the exception and traceback.
Check the returned results when processing a batch of files.

What happens if the analysis routine you want to run through all the items in :py:class:`~Stoner.folders.mixins.DataFolder` is not a method of the :py:class:`Stoner.core.data.Data`
class, but a function written by you? In this case, so long as you write your custom analysis function so that the first positional argument
is the :py:class:`~Stoner.core.data.Data` to be analysed, then the following syntax can be used::

    def my_analysis(data,arg1,arg2,karg=True):
        """Some sort of analysis function with some arguments and keyword argument that works
        on some data *data*."""
        data["Analysis complete"] = True
        return data

    f.each(my_analysis,arg1,arg2,karg=False)

(or alternatively using the matrix multiplication operator @)::

    (my_analysis@f.each)(arg1,arg2,karg=False)

*(my_analysis@f.each)* creates the callable object that iterates *my_analysis* over f, the second set of parenthesis above just calls this iterating object.

If the return value of the function is another instance of :py:class:`~Stoner.core.data.Data` (or whatever is being stored as the items in the
:py:class:`~Stoner.folders.mixins.DataFolder`) then it will replace the items inside the :py:class:`~Stoner.folders.mixins.DataFolder`. The call to :py:attr:`~Stoner.folders.core.BaseFolder.each` will also return a
simple list of the return values. If the function returns something else, then you can have it added to the metadata of each item in the
:py:class:`~Stoner.folders.mixins.DataFolder` by adding a *_return* keyword that can either be True to use the function name as the metadata name or a string to specify
the name of the metadata to store the return value explicitly.

Thus, if your analysis function calculates some parameter that you want to call *beta* you might use the following::

    f=DataFolder(".",pattern="*.txt")
    f.each(my_analysis,arg1,arg2,karg=False,_return="beta")

See :doc:`folder_metadata` to collect analysis results, :doc:`plot_folders`
for bulk plotting, and :doc:`fitting_folders` for fitting collections.
