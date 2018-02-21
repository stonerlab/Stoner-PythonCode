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
        return self

    def __next__(self):
        for i in self.fldr:
            if hasattr(i,"image"):
                ret=i.image
            else:
                ret=i
            yield ret

    def __getitem__(self,index):
        ret=self.fldr[index]
        if hasattr(ret,"image"):
            ret=ret.image
        return ret

    next=__next__


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

    def slice_metadata(self, key=None, values_only=False,output=None):  # pylint: disable=arguments-differ
        """Return a list of the metadata dictionaries for each item/file in the top level group

        Keyword Arguments:
            key(string or list of strings):
                if given then only return the item(s) requested from the metadata
            values_only(bool):
                if given amd *output* not set only return tuples of the dictionary values. Mostly useful
                when given a single key string
            output (str or type):
                Controls the output format from slice_metadata. Possible values are

                - "dict" or dict - return a list of dictionary subsets of the metadata from each image
                - "list" or list - return a list of values of each item pf the metadata
                - "array" or np.array - return a single array - like list above, but returns as a numpy array. This can create a 2D array from multiple keys
                - "Data" or Stoner.Data - returns the metadata in a Stoner.Data object where the column headers are the metadata keys.

        Returns:
            ret(list of dict, tuple or values):
                depending on values_only returns the sliced dictionaries or tuples/
                values of the items

        To do:
            this should probably be a func in baseFolder and should use have
            recursive options (build a dictionary of metadata values). And probably
            options to extract other parts of objects (first row or whatever).
        """
        if output is None:
            output="dict" if not values_only else "list"
        if output not in ["dict","list","array","Data",dict,list,np.ndarray,Data]:
            raise SyntaxError("output of slice metadata must be either dict, list, or array not {}".format(output))
        metadata=[k.metadata for k in self] #this can take some time if it's loading in the images
        if isinstance(key, string_types):
            key=metadata[0].__lookup__(key,multiple=True)
        elif isiterable(key):
            newkey=[]
            for k in key:
                newkey.extend(metadata[0].__lookup__(k,multiple=True))
            key=newkey
        if isinstance(key, list):
            for i,met in enumerate(metadata):
                assert all([k in met for k in key]), 'key requested not in item {}'.format(i)
                metadata[i]={k:v for k,v in metadata[i].items() if k in key}
        if output in ["list","array","Data",list,np.ndarray,Data]:
            cols=[]
            for k in metadata[0]:
                if islike_list(metadata[0][k]):
                    for i,_ in enumerate(metadata[0][k]):
                        cols.append("{}_{}".format(k,i))
                else:
                    cols.append(k)

            for i,met in enumerate(metadata):
                metadata[i]=[]
                for k,v in met.items():
                    if islike_list(v):
                        metadata[i].extend(v)
                    else:
                        metadata[i].append(v)
            if len(metadata[0])==1: #single key
                metadata=[m[0] for m in metadata]
            if output in ["array",np.ndarray]:
                metadata=np.array(metadata)
            if output in ["Data",Data]:
                tmp=Data()
                tmp.data=np.array(metadata)
                tmp.column_headers=cols
                metadata=tmp
        return metadata

    def as_stack(self):
        """Return a ImageStack of the images in the current group."""
        from Stoner.Image import ImageStack
        k = ImageStack(self)
        return k

    def mean(self):
        """Calculate the mean value of all the images in the stack."""
        total=np.zeros_like(self[0])
        for i in self:
            total+=i
        total/=len(self)
        if isinstance(self._type,np.ndarray):
            ret= total.view(type=self._type)
            ret.metadata.update(self[0].metadata)
        elif isinstance(self._type,ImageFile):
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
