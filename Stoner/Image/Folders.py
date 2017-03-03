# -*- coding: utf-8 -*-
"""
Created on Mon May 23 12:05:59 2016

@author: phyrct
"""

from .core import ImageArray
from .core import dtype_range
from Stoner.Core import metadataObject
from Stoner.Util import Data
import numpy as np
from os import path
import copy

from skimage.viewer import CollectionViewer
from Stoner.Folders import DiskBssedFolder, baseFolder
from Stoner.compat import *
from Stoner.compat import string_types


def _load_ImageArray(f,img_num=0, **kargs):
    return ImageArray(f, **kargs)

class ImageFolder(DiskBssedFolder,baseFolder):
    """ImageFolder is designed to behave pretty much like DataFolder but with
    functions and loaders appropriate for image based files.
    
    TO DO: need a function for converting groups into and out of ImageStack
    """

    #_type=ImageArray
    #read_means=False #Only to make grouping work

    def __init__(self, *args, **kargs):
        """
        Initialise the ImageFolder. A list of images to manipulate. Mostly a pass
        through to the :py:class:`Stoner.Folders.baseFolder` class.
        """
        kargs["pattern"]=kargs.get("pattern","*.png")
        kargs["type"]=kargs.get("type",ImageArray)
        super(ImageFolder,self).__init__(*args,**kargs)
        
    
    def loadgroup(self):
        """Load all files from this group into memory"""
        [None for _ in self]
    
    def slice_metadata(self, key=None, values_only=False):
        """Return a list of the metadata dictionaries for each item/file in the
        top level group

        Parameters
        ----------
        key: string or list of strings
            if given then only return the item(s) requested from the metadata
        values_only: bool
            if given only return tuples of the dictionary values. Mostly useful
            when given a single key string
        Returns
        ------
        ret: list of dict, tuple or values
            depending on values_only returns the sliced dictionaries or tuples/
            values of the items
            
        TO DO: this should probably be a func in baseFolder and should use have
        recursive options (build a dictionary of metadata values). And probably
        options to extract other parts of objects (first row or whatever).
        """

        metadata=[k.metadata for k in self] #this can take some time if it's loading in the images
        if isinstance(key, string_types):
            key=[key]
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
        from Stoner.Image import KerrStack
        k = KerrStack(self)
        return k
      
