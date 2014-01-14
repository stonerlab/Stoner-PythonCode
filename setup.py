import os
from setuptools import setup
import Stoner
reload(Stoner)

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "Stoner",
    version = str(Stoner.__version__),
    author = "Gavin Burnell",
    author_email = "g.burnell@leeds.ac.uk",
    description = ("Classes to represent simple scientific data sets and write analysis codes, developed for the University of Leeds Condensed Matter Physics Group"),
    license = "GPLv3",
    keywords = "Data-Analysis Physics",
    url = "http://github.com/~gb119/Stoner-PythonCode",
    packages=['Stoner'],
    test_suite="tests",
    install_requires=["numpy>=1.4","scipy>=0.12","matplotlib>=1.1","h5py"],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
