*****************
Developer's Guide
*****************

This section gives an overview of the internal data classes and explains how to
add file-format support to the Stoner package.

Contributor workflow
====================

Storage migration primitives
----------------------------

On the migration branch, ``Stoner.core.storage`` supplies ``Column``,
``DataStorage`` and ``resolve_columns``. Ordinary ``Data(...)``, ``Data.load``
and the explicit import factories now construct frame-backed objects. Consult
``STORAGE_MIGRATION_PLAN.md`` and ``STORAGE_INTERCHANGE.md`` in the repository
for the current integration gate.

For example, this boundary preserves an excluded integer through a detached copy::

    import numpy as np
    from Stoner.core.storage import DataStorage, resolve_columns

    source = np.ma.array([[0, 100]], mask=[[False, True]], dtype=np.int16)
    package = DataStorage.from_numpy(source, headers=["Field", "Moment"], roles="xy")
    restored = package.copy()
    restored.excluded[0, 1] = False
    assert restored.to_numpy()[0, 1] == 100
    assert package.excluded[0, 1]
    assert resolve_columns(package.schema, "Moment") == 1

Packages contain values, exclusions, schema, typed metadata, fill value and dtype.
Direct record edits require validation before use; factory methods and copies
detach their input state. ``to_pandas()`` is a lossy convenience export and warns
when exclusions or user metadata are omitted. Numeric strings select headers or
patterns, not integer positions; use integers explicitly. The resolver rejects
Boolean positions and negative positions outside the ordinary bounds. Compiled
patterns retain every matching duplicate column as a distinct position.

The internal ``Stoner.core.storage_owner.DataOwner`` now supplies stable header
and role interfaces, masks, detached reads and transactional numerical edits over
these packages. ``Data.from_storage(package)`` and ``Data.from_pandas(frame)``
construct real ``Data`` instances backed by this owner::

    from Stoner import Data

    data = Data.from_storage(package)
    with data.edit_numpy() as draft:
        draft[0, 1] = 25
    assert data[0, 1] == 25
    assert data.column_ids == tuple(column.id for column in package.schema)

Data supports direct assignment, fixed-width header/role edits, independent copies,
row/column insertion, deletion, selection, sorting and reordering. These operations
move exclusions and roles with their columns; repeated selections receive fresh
IDs after the first occurrence. Row appends match duplicate headers by occurrence.
Column concatenation remaps colliding IDs. Metadata merges prefer the left value.

``data``, columns and indexed array results are read-only detached snapshots.
Use ``data[rows, columns] = values``, ``data.mask[...] = exclusions`` or an editing
context to write. ``edit_pandas()`` edits raw values while retaining exclusions;
both editing contexts require fixed shape and dtype. Whole-property assignment
``data.data = array`` explicitly replaces values and masks, retaining overlapping
positional column IDs. Use structural methods to move complete columns. Reordering
with ``headers_too=False`` or ``setas_too=False`` is rejected.

String, compressed-string, mapping, indexed and operator Setas assignments are
supported. Masked scalar reads return ``numpy.ma.masked``. Empty construction has
shape ``(0, 0)``. Native pandas inputs require string headers and a RangeIndex;
use ``Data.from_pandas(..., index="discard")`` to discard an external index.

Construction and loading use the frame owner from the outset. Detached NumPy masked arrays may carry explicitly prepared row and role
annotations. These do not propagate through further NumPy operations or commit
snapshot writes. The old DataArray, ImageArray, KerrArray and ImageStackMixin
implementations have been removed; do not import them or recreate their storage. Numerical consumers use selected columns or an explicit detached
array where masked-array semantics or positional array indexing are required.
Do not export a whole array merely to inspect its shape or dtype, or repeat an
export inside a row loop. A raw DataFrame alone does not carry the separate
exclusion mask. There is no persistent parallel array store or automatic write-back.

Folder metadata can contain text and other non-numeric values. Request
``output="frame"`` or ``output="array"`` for these values; ``output="data"``
requires numeric metadata. Automatic folder metadata selection returns a DataFrame
for non-numeric fields. Constructor header lists retain historical padding and
truncation; subsequent schema assignments must match the existing column count.

Image storage transition
------------------------

The Stage 6 foundations are available in ``Stoner.Image.storage`` and
``Stoner.Image.storage_owner``. ``ImageStorage`` carries an eager xarray Dataset,
typed metadata and fill values; stack packages also carry stable frame records
and valid extents. Populated standalone ``ImageFile`` instances now use
``ImageOwner`` as their sole store. ``ImageFile()`` remains an unpopulated loader
placeholder until pixels are assigned. Explicit zero-sized images are rejected.

Construct an image package with ``ImageStorage.from_numpy(array)`` and a
ragged stack with ``ImageStorage.from_images(images, names=names)``.
Package copies preserve typed metadata and calibration; ``to_xarray()`` exports
detached native state and warns when typed metadata is omitted. Calibrated packing
maps separable axes to padded physical coordinates and scalars to frame coordinates.
``ImageStack``, Kerr, MaskStack, Attocube and Maximus now use this package as
their sole numerical store. Kerr items retain their specialised type; their
``image`` snapshots are read-only, so write pixels through the item itself or
an editing context. Scan headers live in stack metadata. Frame metadata reads
include these shared defaults, while frame assignments create local overrides.

Attocube scan groups and Maximus scans also provide
``scan.to_xarray(format="channels")`` for native analysis. Attocube channels
become named variables on a common grid, each with an exclusion variable;
measured PosX/PosY channels also become two-dimensional physical coordinates.
This retains irregular measured positions without inventing separable axes.
Maximus exposes its explicitly named detector, recorded spatial axes and
stack-axis values, preserving the loader's existing image orientation and
recorded units. Its current loader supports a single detector for this view;
multi-detector and multi-region input mapping remains a separate limitation.

These channel views are detached exports, not another storage owner. The
default ``format="stack"`` retains canonical frame/y/x export, and
``export_storage()`` retains typed metadata for lossless interchange. Channel
exports require a common grid; Maximus rejects a grid whose size no longer
matches its header calibration. Specialised HDF5 saving retains raw pixels,
exclusions, fill values and effective scan metadata, including frame overrides;
this does not add arbitrary native-coordinate serialisation to those formats.

The internal ``Stoner.Image.stack_owner.StackOwner`` now owns validated stack
packages. ``owner.frame(position_or_id)`` returns a stable frame handle;
``owner.image(position_or_id)`` binds an ``ImageFile`` to that handle. Saved
handles survive insertion and reordering and raise ``ReferenceError`` after
deletion. Frame pixels, masks, drawing and shared crops write through to the
stack, while metadata replacement updates the frame record. All frames share
the parent's transaction lock. Clones and numerical exports detach.

``insert``, ``reorder`` and ``delete`` validate structural changes before
publishing them. Whole-stack replacement retains surviving frame IDs but
invalidates earlier crop bounds; deleted IDs cannot be reused within that owner.
Extraction maps per-frame physical coordinates onto the valid image rectangle.
Insertion preserves compatible Dataset, variable and coordinate attributes.
Incompatible units, attributes, scalar coordinate sets or scalar dtypes require
explicit conversion or reconciliation before packing. Two-dimensional coordinate
maps pack as ``(frame, y, x)`` with NaN padding and extract without losing their
names or units. Cropped map images retain their pixel labels through separate
``index_y``/``index_x`` stack coordinates when necessary. Other spatial auxiliary
coordinates must first be expressed as explicit y/x maps.
Public ``stack[index]`` items now use these frame handles. ``stack.frame_ids``
reports their identities; ``stack.reorder(ids)`` and ``stack.sort(...)`` retain
saved handles. ``stack.imarray`` returns a detached read-only masked snapshot.
Use ``stack[frame, y, x] = value``, item assignment or ``stack.edit_numpy()`` /
``stack.edit_xarray()`` for writes. ``from_storage()``, ``export_storage()``,
``from_xarray()``, ``to_xarray()`` and ``to_numpy()`` expose stack interchange.

Bulk ``stack.each`` methods operate on detached images before committing results,
so crops can change frame extents. Direct dtype/shape changes through a saved frame
require a clone and parent assignment, or a whole-stack conversion. Fixed-shape
bulk edits retain saved crops; structural replacement invalidates crop bounds.
Means and deviations exclude user masks and padding. Standard errors count valid
samples at each pixel. Calibrated reductions preserve matching coordinates and
reject conflicting calibration until explicitly reconciled. Non-broadcastable
mask assignments raise instead of silently repeating an incompatible mask.

``image.image`` and ``image.data`` return detached read-only compatibility arrays.
Write pixels through ``image[y, x]`` and exclusions through ``image.mask[y, x]``;
edit metadata through ``image.metadata``. For a whole-array calculation, assign
``image.image = result`` or use a transaction::

    with image.edit_numpy() as pixels:
        pixels[0, 0] = 12
        pixels.mask[1, 2] = True

``edit_xarray()`` provides the same atomic boundary over the Dataset's intensity
and excluded variables. Both contexts preserve shape, dtype and coordinates and
reject competing public writes. ``to_numpy()`` returns an independently writable
masked array; ``from_storage()``, ``export_storage()``, ``from_xarray()`` and
``to_xarray()`` provide explicit interchange.

``image.crop(xmin, xmax, ymin, ymax, copy=False, _=None)`` returns a shared region
without changing the parent's extent. Pixel and mask writes, including drawing,
reach the parent. Use ``copy=True`` or ``clone`` for independence. Ordinary
fixed-shape method commits keep regions live; whole-parent replacement invalidates
them. Crops and rectangular slices preserve calibration. Transpose, ``T``,
``swapaxes``, ``flip_h``, ``flip_v``, ``CW`` and ``CCW`` now permute coordinates
and their units together with pixels and exclusions. ``T`` and the flip/quarter-turn
properties return independent ImageFiles. Explicit in-place permutations invalidate
saved crop bounds; clone a shared frame or region before changing its geometry.
``image.T`` delegates to ``image.transpose(_=None)``: both use the same coordinate
transform, with ``T`` explicitly selecting an independent result. The method keeps
the existing ``_`` argument for controlling clone/in-place behaviour.
Auxiliary spatial coordinates follow their source dimension and scalar coordinates
are retained. Calibrated rotation, rescale, resize, warp, translation, shift, zoom,
affine transforms and gridimage now resample physical coordinates as explicit
``physical_y(y, x)`` and ``physical_x(y, x)`` maps. Existing spatial auxiliary
coordinates follow the same mapping; scalar coordinates and units are retained.
The new dimension axes are positional pixels. Coordinate maps use linear
interpolation, with NaN outside the measured source; these pixels are excluded
even when an intensity boundary mode supplies reflected or wrapped values.
Exclusions are expanded conservatively for higher-order interpolation and resize
anti-aliasing. Numerical interpolation is therefore not a lossless round trip.

For calibrated images, ``resize(output_shape)`` uses scikit-image interpolation;
the inherited NumPy masked-array resize method cannot safely resize owned data.
Registration methods retain their existing numerical shift convention and place
the shifted values on the reference image's grid (or the source grid for a raw
array reference). Unlike arithmetic, registration explicitly reconciles grids.
Folder alignment works on independent images, supplies detached reference pixels
and coordinates, and reports an individual registration exception before trying
to aggregate translation metadata. Shared frame/region geometry requires a
clone, and in-place geometry changes invalidate earlier crop handles. External
output buffers are rejected. Custom warp mappings must be deterministic: the
adapter evaluates the mapping for pixels, exclusions and coordinate fields.

Arithmetic and numerical methods receiving two calibrated ImageFiles require
identical coordinate values and attributes, including units, before conversion to
positional numerical arrays. Mismatches raise without mutation; no automatic
xarray alignment or unit conversion occurs. Raw arrays, scalars and uncalibrated
images retain positional operand semantics.

Calibration does not change argument units: indexing and crop bounds remain pixel
positions, and ``rotate`` still takes radians. Physical-unit arguments need an
explicit API decision; loaders may already supply physical axes and units through
the storage package.

General contribution checks
---------------------------

Start with the repository's `AGENTS.md
<https://github.com/stonerlab/Stoner-PythonCode/blob/stable/AGENTS.md>`_,
`maintenance plan
<https://github.com/stonerlab/Stoner-PythonCode/blob/stable/MAINTENANCE_PLAN.md>`_
and `docstring standard
<https://github.com/stonerlab/Stoner-PythonCode/blob/stable/DOCSTRING_STYLE.md>`_.
They describe the checkout conventions, current work and expected Google-style
docstrings. Preserve the scientific fixtures and the committed ``doc/plot_cache``.

From the repository root, create the test environment and run a focused test::

    conda env create -f tests/test-env.yml
    conda run -n test-environment python -m pip install --no-deps -e .
    conda run -n test-environment python -m pytest tests/Stoner/test_Core.py

Run the full suite before completing changes that affect shared behaviour.
``mamba`` can be used instead of ``conda`` to create the environment. Use a
supported Python version; the separate Python 3.6 LabVIEW environment is not a
development environment for this package.

The documentation environment is defined in ``doc/docs-env.yml``. For a cached
HTML build in PowerShell, use::

    conda env create -f doc/docs-env.yml
    $env:READTHEDOCS = 'True'
    conda run -n rtd-build python -m sphinx -b html doc doc/_build/html
    Remove-Item Env:READTHEDOCS

The Conda specification includes the Graphviz executable used for inheritance
diagrams. If using ``doc/requirements.txt`` with pip instead, install Graphviz
separately and ensure its ``dot`` command is on PATH.

``READTHEDOCS=True`` consumes the retained plot cache. Leaving it unset executes
the plotting examples and deliberately refreshes that cache. Documentation
examples are also exercised by ``tests/Stoner/test_doc_samples.py``; use that
test module to check changes to example behaviour.

Edit the repository-root ``README.rst`` as the definitive project overview.
``make -C doc readme`` copies it to ``doc/readme.rst``; both ``make commit``
and ``make -C doc html`` run that copy step. When building directly with Sphinx,
refresh the copy yourself if the README changed. The installation guide includes
the marked installation section from the root README to keep its commands aligned.

To inspect the built pages in a browser, serve the HTML output over localhost::

    conda run --no-capture-output -n rtd-build python -m http.server 8765 --bind 127.0.0.1 --directory doc/_build/html

Open ``http://127.0.0.1:8765/index.html`` and check the page layout and relevant
navigation links. Substitute the actual HTML output directory if using a
maintenance build, and another high-numbered port if 8765 is occupied. Serve
only the generated HTML directory. Binding to ``127.0.0.1`` keeps the preview
local to your computer. This approach works with browser tools that reject
direct ``file://`` navigation, without changing their URL policy.

Reload the browser after rebuilding. Keep the server running while reviewing
the pages and stop it with Ctrl+C when finished. Browser inspection complements
the build's warning report and API inventory checks.

Package versions and releases
=============================

Set the package version only in ``Stoner/__init__.py``, in the literal
``__version__`` assignment. Setuptools reads this attribute for wheel and sdist
metadata; the Conda recipe reads the same assignment. Runtime
``Stoner.__version_info__`` and the Sphinx full and short versions are derived
from it. The Conda build number is a separate recipe revision, not another
package version.

For a release, commit the version change and create the matching ``v<version>``
Git tag on that commit, then publish the GitHub release. The release jobs reject
a tag that disagrees with the source before building/uploading packages or
building documentation. They build the tagged checkout, including the docs.
The check also accepts an unprefixed version tag. Manual workflow dispatches
use the selected ref and do not require a release tag.

Package builds belong in GitHub Actions. ``Package validation`` builds wheel
and sdist archives on pushes and pull requests, checks their contents, and
installs each into a separate fresh environment for resource and sample-data
probes. The release workflow builds and uploads the PyPI and Conda packages;
the Conda recipe checks its installed import and version metadata. Record the
actual CI run results for the release commit in the maintenance plan.

Understanding the class structure
=================================

The public :py:class:`Stoner.core.data.Data` class presents a compact interface,
but delegates some responsibilities to specialised objects:

``data``
    A detached, read-only numerical snapshot. The owner holds a pandas DataFrame,
    a Boolean exclusion mask and stable column schema. Snapshot results carry
    column roles and positional row indices for the retained public array API.

``metadata``
    An owner-aware mapping over a :py:class:`Stoner.core.base.TypeHintedDict`.
    It retains explicit type information and guards writes during numerical edits.

``column_headers``
    A list-like view of the labels maintained alongside the numerical columns.

``setas``
    An owner-bound interface using :py:class:`~Stoner.core.setas.Setas` syntax.
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
