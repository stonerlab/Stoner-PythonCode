Stoner
======

Scientific data analysis for experimental condensed matter physics.

.. image:: https://github.com/stonerlab/Stoner-PythonCode/actions/workflows/run-tests-action.yaml/badge.svg?branch=stable
   :target: https://github.com/stonerlab/Stoner-PythonCode/actions/workflows/run-tests-action.yaml
   :alt: Tests

.. image:: https://coveralls.io/repos/github/stonerlab/Stoner-PythonCode/badge.svg?branch=stable
   :target: https://coveralls.io/github/stonerlab/Stoner-PythonCode?branch=stable
   :alt: Test coverage

.. image:: https://app.codacy.com/project/badge/Grade/a9069a1567114a22b25d63fd4c50b228
   :target: https://app.codacy.com/gh/stonerlab/Stoner-PythonCode/dashboard
   :alt: Code quality

.. image:: https://badge.fury.io/py/Stoner.svg
   :target: https://pypi.org/project/Stoner/
   :alt: PyPI version

.. image:: https://anaconda.org/phygbu/stoner/badges/version.svg
   :target: https://anaconda.org/phygbu/stoner
   :alt: Conda version

.. image:: https://readthedocs.org/projects/stoner-pythoncode/badge/?version=stable
   :target: https://stoner-pythoncode.readthedocs.io/en/stable/
   :alt: Documentation status

.. image:: https://zenodo.org/badge/10057055.svg
   :target: https://zenodo.org/badge/latestdoi/10057055
   :alt: Citation DOI

Stoner helps you load experimental measurements, preserve their metadata,
fit physical models, process images and apply the same analysis to collections
of files. It originated in the Condensed Matter Physics group at the University
of Leeds and builds on NumPy, SciPy, Matplotlib and lmfit.

Start with the `user guide`_, browse the `API reference`_, or explore the
`example scripts`_ and `sample data`_ in the repository.

Working with measurements
-------------------------

The five main classes are available directly from ``Stoner``:

* ``Data``: a numerical table with masks, column roles and experimental metadata;
  includes loading, transformations, plotting and curve fitting.
* ``DataFolder``: collections of measurements that can be filtered, grouped and
  processed together, with access to their combined metadata.
* ``ImageFile``: an image with metadata, masks and image-processing methods.
* ``ImageFolder``: collections of images with grouping and bulk operations.
* ``ImageStack``: images held together in a three-dimensional array for stack
  operations.

Many analysis methods modify the object in place and return that same object
for chaining. Other methods return results, arrays or plotting objects; check
individual method documentation. Use ``data.clone`` when you need an independent
copy before processing.

Registered loaders recognise supported instrument and facility formats.
HDF5 and ZIP support lives in ``Stoner.formats`` and ``Stoner.folders``; support
for specific layouts does not imply that every arbitrary HDF5 file can be read.

Installation
------------

.. installation-start

The current source requires **Python 3.11 or newer**. The test workflow covers
Python 3.11-3.14 on Linux and Python 3.14 on macOS. Published package versions
may lag behind the source branch; consult the selected release's requirements.

We recommend a Conda-based Python distribution, such as **Anaconda or Miniforge**.
Create a dedicated environment using the ``phygbu`` and ``conda-forge`` channels::

    conda create -n stoner -c phygbu -c conda-forge python=3.14 stoner
    conda activate stoner

Alternatively, install the published package into an activated Python
virtual environment with pip::

    python -m pip install Stoner

Optional features
~~~~~~~~~~~~~~~~~

Core installation does not require OCR, specialist format readers or a Qt GUI.
For pip installations, extras select additional functionality, for example::

    python -m pip install "Stoner[TDMS,image_alignment,ocr]"

Other extras include ``facility_formats``, ``hyperspy``, ``plot_styles``,
``PrettyPrint``, ``mimetype_detection``, ``numba``, ``cv2``, ``mayavi`` and ``qt``.
Their declarations are in ``pyproject.toml``. In Conda environments, install
optional packages from the Conda channels where available.

**OCR is optional.** The ``ocr`` extra installs the Python wrapper
``pytesseract``; text recognition also needs the separate Tesseract executable.
For a Conda environment, both can be installed with::

    conda install -c conda-forge pytesseract tesseract

.. installation-end

A small example
---------------

Create a measurement, label its columns and attach experimental metadata::

    import numpy as np
    from Stoner import Data

    data = Data(np.array([[0.0, 1.0], [2.0, 5.0], [1.0, 3.0]]),
                column_headers=["Field", "Signal"], setas="xy")
    data["Temperature"] = 4.2
    ordered = data.clone.sort("Field")
    ordered.plot()

For loading real measurements and fitting models, see the `user guide`_ and
`example scripts`_.

Development and contributions
-----------------------------

The maintained source is on the ``stable`` branch of the `repository`_.
To work on the code with its test dependencies::

    git clone --branch stable https://github.com/stonerlab/Stoner-PythonCode.git
    cd Stoner-PythonCode
    conda env create -f tests/test-env.yml
    conda run -n test-environment python -m pip install --no-deps -e .
    conda run -n test-environment python -m pytest

Read the `contributor guide`_ for documentation builds and extension points,
`AGENTS.md`_ for repository conventions, and the `maintenance plan`_ for current
work. Report reproducible problems through the `issue tracker`_. Release tags
and earlier source are available in the `repository`_.

Contact, licence and citation
--------------------------------

The lead developer is Gavin Burnell (g.burnell@leeds.ac.uk), with contributions
from current and former members of the Leeds Condensed Matter Physics group.
See the user guide for contributor credits.

Copyright University of Leeds and contributors, except where individual files
state otherwise. Stoner is licensed under the GNU General Public License v3;
see `LICENSE.md`_. Please cite the package using its `Zenodo DOI`_.

.. _user guide: https://stoner-pythoncode.readthedocs.io/en/stable/UserGuide/ugindex.html
.. _API reference: https://stoner-pythoncode.readthedocs.io/en/stable/Stoner.html
.. _repository: https://github.com/stonerlab/Stoner-PythonCode
.. _example scripts: https://github.com/stonerlab/Stoner-PythonCode/tree/stable/doc/samples
.. _sample data: https://github.com/stonerlab/Stoner-PythonCode/tree/stable/sample-data
.. _contributor guide: https://stoner-pythoncode.readthedocs.io/en/stable/UserGuide/developer.html
.. _AGENTS.md: https://github.com/stonerlab/Stoner-PythonCode/blob/stable/AGENTS.md
.. _maintenance plan: https://github.com/stonerlab/Stoner-PythonCode/blob/stable/MAINTENANCE_PLAN.md
.. _issue tracker: https://github.com/stonerlab/Stoner-PythonCode/issues
.. _LICENSE.md: https://github.com/stonerlab/Stoner-PythonCode/blob/stable/LICENSE.md
.. _Zenodo DOI: https://zenodo.org/badge/latestdoi/10057055
