# -*- coding: utf-8 -*-
"""
Created on Mon May 23 12:05:59 2016

@author: phyrct
"""

from .core import ImageArray
from Stoner import Data
from Stoner.Folders import DiskBssedFolder, baseFolder
from Stoner.compat import string_types
from Stoner.tools import isiterable,islike_list
from Stoner.Image import ImageFile,ImageArray

from skimage.viewer import CollectionViewer
import numpy as np

def _load_ImageArray(f, **kargs):
    """Simple meothd to load an image array."""
    kargs.pop("img_num",None)
    return ImageArray(f, **kargs)

class _generator(object):

    """A helper class to iterator over ImageFolder yet remember it's own length."""

    def __init__(self,fldr):
        self.fldr=fldr
        self.len=len(fldr)

    def __len__(self):
        return self.len

    def __iter__(self):
        self.ix=0
        return self

    def __next__(self):
        if self.ix<len(self):
            ret=self[self.ix]
            self.ix+=1
            return ret
        else:
            raise StopIteration("Finished iterating Folder.")

    def __getitem__(self,index):
        ret=self.fldr[index]
        if hasattr(ret,"image"):
            ret=ret.image
        return ret

    def next(self):
        return self.__next__()

class ImageFolderMixin(object):

    """Mixin to provide a folder object for images.

    ImageFolderMixin is designed to behave pretty much like DataFolder but with
    functions and loaders appropriate for image based files.

        Attributes:
        type (:py:class:`Stoner.Image.core.ImageArray`) the type ob object to sotre in the folder (defaults to :py:class:`Stoner.Cire.Data`)

        extra_args (dict): Extra arguments to use when instantiatoing the contents of the folder from a file on disk.

        pattern (str or regexp): A filename globbing pattern that matches the contents of the folder. If a regular expression is provided then
            any named groups are used to construct additional metadata entryies from the filename. Default is *.* to match all files with an extension.

        read_means (bool): IF true, additional metatdata keys are added that return the mean value of each column of the data. This can hep in
            grouping files where one column of data contains a constant value for the experimental state. Default is False

        recursive (bool): Specifies whether to search recurisvely in a whole directory tree. Default is True.

        flatten (bool): Specify where to present subdirectories as spearate groups in the folder (False) or as a single group (True). Default is False.
            The :py:meth:`DiskBasedFolder.flatten` method has the equivalent effect and :py:meth:`DiskBasedFolder.unflatten` reverses it.

        directory (str): The root directory on disc for the folder - by default this is the current working directory.

        multifile (boo): Whether to select individual files manually that are not (necessarily) in  a common directory structure.

        readlist (bool): Whether to read the directory immediately on creation. Default is True
    """

    def __init__(self, *args, **kargs):
        """nitialise the ImageFolder.

        Mostly a pass through to the :py:class:`Stoner.Folders.baseFolder` class.
        """
        kargs["pattern"]=kargs.get("pattern","*.png")
        kargs["type"]=kargs.get("type",ImageArray)
        if "flat" in self._defaults:
            del self._defaults["flat"]
        super(ImageFolderMixin,self).__init__(*args,**kargs)

    @property
    def images(self):
        """A generator that iterates over just the images in the Folder."""
        return _generator(self)

    def loadgroup(self):
        """Load all files from this group into memory"""
        for _ in self:
            pass

    def as_stack(self):
        """Return a ImageStack of the images in the current group."""
        from Stoner.Image import ImageStack
        k = ImageStack(self)
        return k

    def mean(self):
        """Calculate the mean value of all the images in the stack."""
        total=np.zeros_like(self[0])
        for i in self:
            total+=i.data
        total/=len(self)
        if issubclass(self._type,np.ndarray):
            ret= total.view(type=self._type)
            ret.metadata.update(self[0].metadata)
        elif issubclass(self._type,ImageFile):
            ret=self._type()
            ret.image=total
            ret.metadata.update(self[0].metadata)
        else:
            ret=total
        return ret


    def view(self):
        """Create a matplotlib animated view of the contents.

        """
        cv=CollectionViewer(self.images)
        cv.show()
        return cv

class ImageFolder(ImageFolderMixin,DiskBssedFolder,baseFolder):

    """Folder object for images.

    ImageFolder is designed to behave pretty much like DataFolder but with
    functions and loaders appropriate for image based files.

        Attributes:
        type (:py:class:`Stoner.Image.core.ImageArray`) the type ob object to sotre in the folder (defaults to :py:class:`Stoner.Cire.Data`)

        extra_args (dict): Extra arguments to use when instantiatoing the contents of the folder from a file on disk.

        pattern (str or regexp): A filename globbing pattern that matches the contents of the folder. If a regular expression is provided then
            any named groups are used to construct additional metadata entryies from the filename. Default is *.* to match all files with an extension.

        read_means (bool): IF true, additional metatdata keys are added that return the mean value of each column of the data. This can hep in
            grouping files where one column of data contains a constant value for the experimental state. Default is False

        recursive (bool): Specifies whether to search recurisvely in a whole directory tree. Default is True.

        flatten (bool): Specify where to present subdirectories as spearate groups in the folder (False) or as a single group (True). Default is False.
            The :py:meth:`DiskBasedFolder.flatten` method has the equivalent effect and :py:meth:`DiskBasedFolder.unflatten` reverses it.

        directory (str): The root directory on disc for the folder - by default this is the current working directory.

        multifile (boo): Whether to select individual files manually that are not (necessarily) in  a common directory structure.

        readlist (bool): Whether to read the directory immediately on creation. Default is True
    """
    pass
