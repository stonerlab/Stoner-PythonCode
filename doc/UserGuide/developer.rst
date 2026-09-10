*****************
Developer's Guide
*****************

This section gives an overview of the internal data classes and explains how to
add file-format support to the Stoner package.

Understanding the class structure
=================================

The public :py:class:`Stoner.core.data.Data` class presents a compact interface,
but delegates some responsibilities to specialised objects:

``data``
    A :py:class:`Stoner.core.array.DataArray`, derived from
    :py:class:`numpy.ma.MaskedArray`. It carries a
    :py:class:`Stoner.core.setas.Setas` object that records column roles such as
    x, y and error columns, and it retains row indices through its ``i``
    attribute.

``metadata``
    A :py:class:`Stoner.core.base.TypeHintedDict`. It behaves like a dictionary
    while retaining type information needed when metadata is written to formats
    that do not preserve Python types directly.

``column_headers``
    A list-like view of the labels maintained alongside the numerical columns.

``setas``
    A proxy to the data array's :py:class:`~Stoner.core.setas.Setas` instance.
    It accepts column numbers, names and regular-expression matches, and keeps
    assignments synchronised when columns are added or removed.

The public :py:class:`~Stoner.core.data.Data` name is a convenience re-export of
:py:class:`Stoner.core.data.Data`. Internal documentation should normally link
to the defining module so that Sphinx can resolve the target unambiguously.

Adding a data-file format
=========================

Current versions of Stoner register loader functions rather than requiring a
new ``Data`` subclass for every format. Package loaders live under
:py:mod:`Stoner.formats.data`; project-specific loaders may be defined in the
application that needs them.

A loader receives a newly created :py:class:`~Stoner.core.data.Data` object. It
must either populate and return that object or raise
:py:exc:`Stoner.core.exceptions.StonerLoadError` when the input is not its
format. A minimal outline is::

    import numpy as np

    from Stoner.core.data import Data
    from Stoner.core.exceptions import StonerLoadError
    from Stoner.formats.decorators import register_loader
    from Stoner.tools.file import get_filename


    @register_loader(patterns=[(".example", 32)], name="ExampleFile", what="Data")
    def load_example(new_data: Data, *args, **kwargs) -> Data:
        """Load an Example Instrument data file."""
        filename, args, kwargs = get_filename(args, kwargs)
        if filename is None:
            raise StonerLoadError("No filename supplied")

        new_data.filename = filename
        with open(filename, encoding="utf-8") as source:
            signature = source.readline().strip()
            if signature != "EXAMPLE DATA":
                raise StonerLoadError("Not an Example Instrument file")
            new_data.data = np.genfromtxt(source)

        new_data.column_headers = ["Field", "Signal"]
        new_data.setas = "xy"
        return new_data

The :py:func:`Stoner.formats.decorators.register_loader` ``patterns`` and
``mime_types`` arguments narrow the candidates considered by automatic loading.
The priority determines their order: lower numbers are tried first. Give an
early priority only to a loader that can identify its format reliably. Broad
text loaders should run later because they may partially parse unrelated files.

Read as much useful metadata as the format provides and give every numerical
column a header. Metadata strings can be converted to suitable Python values
with :py:func:`Stoner.core.base.string_to_type`.

Saving formats use the corresponding
:py:func:`Stoner.formats.decorators.register_saver` decorator. Loader and saver
functions should have Google-style docstrings so that their arguments, return
values and exceptions appear correctly in the API documentation.

Testing a format
================

Add a small representative file to ``sample-data`` or the focused test-data
directory and test both sides of format detection:

* the intended file loads with the expected data, headers and metadata;
* an unrelated file causes the loader to raise ``StonerLoadError`` promptly;
* save-and-reload round trips preserve the fields supported by the format.

Where a loader depends on an optional package, its tests should also describe
the behaviour when that dependency is unavailable.
