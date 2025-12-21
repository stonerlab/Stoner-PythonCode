"""Provides the subclasses for loading different file formats into :py:class:`Stoner.Data` objects.

You do not need to use these classes directly, they are made available to :py:class:`Stoner.Data` which
will load each of them in turn when asked to load an unknown data file.

Each class has a :py:attr:`Stoner.Core.DataFile.priority` attribute that is used to determine the order in which
they are tried by :py:class:`Stoner.Data` and friends where trying to load data.
Larger priority index classes are run last (so is a bit of a misnomer!).

Each class should implement a :py:meth:`Stoner.Core.DataFile._load` method and optionally a
:py:meth:`Stoner.Core.DataFile.save` method. Classes should make every effort to
positively identify that the file is one that they understand and throw a
:py:exception:Stoner.cpre.exceptions.StonerLoadError` if not.

Classes may also provide :py:attr:`Stoner.Core.DataFile.patterns` attribute which is a list of filename glob patterns
(e.g.  ['*.data','*.txt']) which is used in the file dialog box to filter the list of files. Finally, classes can
provide a :py:attr:`Stoner.Core.DataFile.mime_type` attribute which gives a list of mime types that this class might
be able to open. This helps identify classes that could be use to load particular file types.
"""

# __all__ = ["instruments", "generic", "rigs", "facilities", "simulations", "attocube", "maximus"]
# from . import instruments, generic, rigs, facilities, simulations, attocube, maximus
__all__ = ["data", "image"]
from . import data, image
