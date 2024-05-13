import os
from setuptools import setup, find_packages
import re
import sys, io
from os import environ as env


def get_version():
    init_name=os.path.join(os.path.dirname(__file__),"Stoner","__init__.py")
    with open(init_name,"r") as init:
        for line in init:
            line=line.strip()
            if line.startswith("__version_info__"):
                parts=line.split("=")
                __version_info__=eval(parts[1].strip())
                return '.'.join(__version_info__)
    raise ValueError(f"Failed to get version info from {init_name}")

def yield_sphinx_only_markup(lines):
    """
    :param file_inp:     a `filename` or ``sys.stdin``?
    :param file_out:     a `filename` or ``sys.stdout`?`

    """
    substs = [
        ## Selected Sphinx-only Roles.
        #
        (r':abbr:`([^`]+)`',        r'\1'),
        (r':ref:`([^`]+)`',         r'`\1`_'),
        (r':term:`([^`]+)`',        r'**\1**'),
        (r':dfn:`([^`]+)`',         r'**\1**'),
        (r':(samp|guilabel|menuselection):`([^`]+)`',        r'``\2``'),
        (r':py:[a-z]+:`([^`]+)`',        r'\1'),



        ## Sphinx-only roles:
        #        :foo:`bar`   --> foo(``bar``)
        #        :a:foo:`bar` XXX afoo(``bar``)
        #
        #(r'(:(\w+))?:(\w+):`([^`]*)`', r'\2\3(``\4``)'),
        (r':(\w+):`([^`]*)`', r'\1(``\2``)'),


        ## Sphinx-only Directives.
        #
        (r'\.\. doctest',           r'code-block'),
        (r'\.\. plot::',            r'.. '),
        (r'\.\. seealso',           r'info'),
        (r'\.\. glossary',          r'rubric'),
        (r'\.\. figure::',          r'.. '),


        ## Other
        #
        (r'\|version\|',              r'x.x.x'),
    ]

    regex_subs = [ (re.compile(regex, re.IGNORECASE), sub) for (regex, sub) in substs ]

    def clean_line(line):
        try:
            for (regex, sub) in regex_subs:
                line = regex.sub(sub, line)
        except Exception as ex:
            print("ERROR: %s, (line(%s)"%(regex, sub))
            raise ex

        return line

    for line in lines:
        yield clean_line(line)

def read(fname):
    mydir=os.path.dirname(__file__)
    with io.open(os.path.join(mydir, fname)) as fd:
        return fd.readlines()

def requires(fname):
    mydir=os.path.dirname(__file__)
    with io.open(os.path.join(mydir, fname)) as fd:
        entries=fd.readlines()
        entries=[entry for entry in entries if entry[0] not in " #\n\t"]
        return entries

if "READTHEDOCS" in env:
    requyirements="doc/requirements.txt"
else:
    requyirements="requirements.txt"

setup(
    name = "Stoner",
    python_requires='>3.7',
    version = str(get_version()),
    author = "Gavin Burnell",
    author_email = "g.burnell@leeds.ac.uk",
    description = "Library to help write data analysis tools for experimental condensed matter physics.",
    license = "GPLv3",
    keywords = "Data-Analysis Physics",
    url = "http://github.com/~gb119/Stoner-PythonCode",
    packages=find_packages(),
    package_dir={'Stoner': 'Stoner'},
    package_data={'Stoner':['stylelib/*.mplstyle']},
    test_suite="tests",
    # setup_requires=['pytest-runner'],
    # tests_require=['pytest'],
    install_requires=requires(requyirements),
    extras_require = { "PrettyPrint":["tabulate>=0.7.5"],
                       "mimetype_detection":["magic"],
                       "TDMS":["nptdms"],
                       "numba":["numba"],
                       "cv2":["cv2"],
                       "image_alignment":["imreg_dft","image_registration"]},
    long_description= ''.join(yield_sphinx_only_markup(read('README.rst'))),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
