import os
from setuptools import setup, find_packages
import Stoner

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
    packages=find_packages(),
    package_dir={'Stoner': 'Stoner'},
    package_data={'Stoner':['stylelib/*.mplstyle']},
    test_suite="tests",
    install_requires=["numpy>=1.7","scipy>=0.14","matplotlib>=1.5","h5py","lmfit","filemagic","pillow","scikit-image"],
    long_description=read('doc/readme.md'),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
