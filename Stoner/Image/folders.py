# -*- coding: utf-8 -*-
"""
Created on Mon May 23 12:05:59 2016

@author: phyrct
"""

from .core import ImageArray
from Stoner.Folders import DiskBssedFolder, baseFolder
from Stoner.compat import string_types

from collections import Iterable


def _load_ImageArray(f, **kargs):
    """Simple meothd to load an image array."""
    kargs.pop("img_num",None)
    return ImageArray(f, **kargs)

class ImageFolder(DiskBssedFolder,baseFolder):
    
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

    def __init__(self, *args, **kargs):
        """nitialise the ImageFolder. 
        
        Mostly a pass through to the :py:class:`Stoner.Folders.baseFolder` class.
        """
        kargs["pattern"]=kargs.get("pattern","*.png")
        kargs["type"]=kargs.get("type",ImageArray)
        if "flat" in self._defaults:
            del self._defaults["flat"]
        super(ImageFolder,self).__init__(*args,**kargs)
        
    
    def loadgroup(self):
        """Load all files from this group into memory"""
        for _ in self:
            pass
        
    def slice_metadata(self, key=None, values_only=False):  # pylint: disable=arguments-differ
        """Return a list of the metadata dictionaries for each item/file in the top level group

        Keyword Arguments:
            key(string or list of strings):
                if given then only return the item(s) requested from the metadata
            values_only(bool):
                if given only return tuples of the dictionary values. Mostly useful
                when given a single key string
        Returns:
            ret(list of dict, tuple or values):
                depending on values_only returns the sliced dictionaries or tuples/
                values of the items
            
        To do: 
            this should probably be a func in baseFolder and should use have
            recursive options (build a dictionary of metadata values). And probably
            options to extract other parts of objects (first row or whatever).
        """
        metadata=[k.metadata for k in self] #this can take some time if it's loading in the images
        if isinstance(key, string_types):
            key=metadata[0].__lookup__(key,multiple=True)
        elif isinstance(key,Iterable):
            newkey=[]
            for k in key:
                newkey.extnd(metadata[0].__lookup__(k,multiple=True))
            key=newkey
        if isinstance(key, list):
            for i,met in enumerate(metadata):
                assert all([k in met for k in key]), 'key requested not in item {}'.format(i)
                metadata[i]={k:v for k,v in metadata[i].items() if k in key}
        if values_only:
            for i,met in enumerate(metadata):
                metadata[i]=[v for k,v in met.items()]
            if len(metadata[0])==1: #single key
                metadata=[m[0] for m in metadata]
        return metadata
    
    def stack(self):
        """Return a ImageStack of the images in the current group."""
        from Stoner.Image import ImageStack
        k = ImageStack(self)
        return k
      
