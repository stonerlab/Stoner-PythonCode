*****************
Developer's Guide
*****************

.. currentmodule:: Stoner.Core

This section provides some notes and guidance on extending the Stoner Package.

Understanding the Class Structure
=================================

The Stoner Package makes use of several classes to handle representing experimental data that are hidden away from the casual user of the package.
From the end-user perspective, the :py:class:`DataFile` class contains several attributes that behave like data types that the user is familiar with:

-   data
    Although this is essentially a numpy array, it is in fact a subclass of a :py:class:`numpy.ma.MaskedArray`, :py:class:`DataArray`, with an extra attribute 
    :py:attr:`DataArray.setas` that is used to label the columns of data and to track assignments to types of columns (x,y,z etc). As 
    :py:class:`DataArray` can thus track what columns mean in cartersian coordinate systems, it can provide an on-the-fly conversion to polar
    coordinate systems. :py:class:`DataArray` also keeps track of the row numbers of the array and can label its rows with the :py:attr:`DataArray.i`
    attribute.

-   metadata
    In many respects this looks like a standard dictionary, albeit one that sorts its keys in alphabetical order. Actually it is a subclass of a
    blist.sortedDict that maintains a second dictionary with matching keys that stores information about the data type of the values in the metadata
    in order to support export to non-duck typed environments.

-   column_headers
    This appears to be a standard list of strings, except that it is kept in sync with the size of the numerical data array, adding extra entries or
    losing them as the data array gains or loses columns. Actually this is proxied through to the :py:attr:`DataArray.setas` attribute    

-   setas
    This is a simple proxy through to the :py:attr:`DataFile.data` attribute's :py:attr:`DataArray.setas` attribute. As described above, this provides
    the column labelling and assignment functionality, it also handles the code to index columns by number, name, regular expression pattern. Internally
    this attribute stores a simple list of the column assignments, but supports being called with a variety of arguments and keyword arguments to manipulate
    that list. Attempting to set this attribute results in a translation to the appropriate function call, whilst getting it is converted to expressing it
    as a list. The attribute also supports both dictionary like and list like get and set item calls to access elements by index, or indices by assignment.

.. figure:: figures/class_heirarchy.svg

    The relationships between the various classes that make up the :py:class:`DataFile` structure. The dashed arrows represent proxy attributes, solid
    arrows represent direct attributes. The arrow labels indicate the attribute names, whilst the labels in the boxes are the class names.

Adding New Data File Types
==========================

The first question to ask is whether the data file format that you are working with is one that others in the group will be interested in using. 
If so, then the best thing would be to include it in the :py:mod:`Stoner.FileFormats` module in the package, otherwise you should just write 
the class in your own script files. In either case, develop the class in your own script files first.

The best way to implement handling a new data format is to write a new subclass of :py:class:`DataFile`::

    class NewInstrumentFile(DataFile):
        """Extends DataFile to load files from somewhere else
    
        Written by Gavin Burnell 11/3/2012"""
     
A document string should be provided that will help the user identify the function of the new class (and avoid using names that might be commonly replicated !). 
Only one method needs to be implemented: a new :py:meth:`DataFile._load` method. Ththis should have the following structure::
    
    def load(self,filename=None,*args):
        """Just call the parent class but with the right parameters set"""
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename

then follows the code to actually read the file. It **must** at the very least provide a column header for 
every column of data and read in as much numeric data as possible and it **should** read as 
much of the meta data as possible. The function terminates by returning a copy of the current object::

    return self

One useful function for reading metadata from files is :py:meth:`typeHintedDict.string\_to\_type` which will try to convert a 
string representation of data into a sensible Python type.

There are two class attributes, :py:attr:`DataFile.priority` that can be used to tweak the automatic file 
importing code.::

      self.priority=32

When the subclasses are tried to see if they can load an undetermined file, they are tried in order of 
priority. If your load code can make a positive determination that it has the correct file 
(eg by looking for some magic combination of characters at the start of the file) and can throw an 
exception if it tries loading an incorrect file, then you can give it a lower priority number to 
force it to run earlier. Conversely if your only way of identifying your own files is seeing they make 
sense when you try to load them and that you might partially succeed with a file from another system (as 
can happen if you have a tab or comma separated text file), then you should raise the priority number. 
Currently Lpy:class:`DataFile` defaults to 32, :py:class:`Stoner.FileFormats.CSVFile` and 
:py:class:`Stoner.FileFormats.BigBlueFile` have values of 128 and 64 respectively.

THe other attribute one might add is :py:attr:`DataFile.patterns` which is a list of filename extension glob patterns that
will match the expected filenames for this format::

    self.patterns=["*.txt","*/dat","*/qda"]

Any custome data class **should** try to identify if the file is the correct format and if so it **must** raise a
:py:class:`StonerLoadError` exception in order that the autoloading code will known to try a different subclass of :py:class:`DataFile`.

If you need to write any additional methods please make sure that they have Google code-style document 
strings so that the API documentation is picked up correctly.
