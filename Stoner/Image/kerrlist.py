# -*- coding: utf-8 -*-
"""
Created on Mon May 23 12:05:59 2016

@author: phyrct
"""

from .core import KerrArray
import numpy as np
import os, sys, time
from os import path
from copy import copy
import PIL #check we have python imaging library plugin
import skimage
from skimage.io import ImageCollection
from skimage.io import imread
from skimage import filters, feature
#from skimage import draw,exposure,feature,io,measure,\
#                    filters,util,restoration,segmentation,\
#                    transform
#from skimage.viewer import ImageViewer,CollectionViewer
from Stoner.Folders import objectFolder
from Stoner.compat import *


GRAY_RANGE=(0,65535)  #2^16
IM_SIZE=(512,672)
AN_IM_SIZE=(554,672) #Kerr image with annotation not cropped

StringTypes=(str,unicode)


def _load_KerrArray(f,img_num=0, **kwargs):
    return KerrArray(f, **kwargs)

class KerrList(objectFolder):
    """KerrList groups functions that can be applied to a group of KerrImages.
    In general it is designed to behave pretty much like a normal python list.

    This version inherits from Stoner.Folders.objectFolder to provide grouping and iterator functions.
    """

    _type=KerrArray # class attribute to keep things happy

    _listfuncs_proxy=None

    def __init__(self, *args,**kargs):
        """
        Initialise a KerrList. A list of images to manipulate. Mostly a pass
        through to the :py:class:`Stoner.Folders.objectFolder` class.
        """
        kargs["pattern"]=kargs.get("pattern","*.png")
        super(KerrList, self).__init__(*args,**kargs)
        self.altdata=[None for i in self.files] #altdata for if arrays are changed
                                               #from those loaded from memory


    def __dir__(self):
        """dir(KerrList) returns attributes from files as well as listfuncs."""
        listfuncs=set(dir(self._listfuncs))
        selfdir=set(super(KerrList,self).__dir__())
        return list(listfuncs|selfdir)

    @property
    def _listfuncs(self):
        """Cache imported listfuncs."""
        if self._listfuncs_proxy is None:
            import listfuncs
            self._listfuncs_proxy=listfuncs
        return self._listfuncs_proxy

    def __getattr__(self,name):
        """run when asking for an attribute that doesn't exist yet. It
        looks in listfuncs for a match. If
        it finds it it returns a copy of the function that automatically adds
        the KerrList as the first argument."""

        ret=None
        if name in dir(self._listfuncs):
            workingfunc=getattr(self._listfuncs,name)
            ret=self._func_generator(workingfunc)
        else:
            ret=super(KerrList,self).__getattr__(name)
        return ret


    def _func_generator(self,workingfunc):
        """generate a function that adds self as the first argument"""

        def gen_func(*args, **kwargs):
            r=workingfunc(self, *args, **kwargs) #send copy of self as the first arg
            return r
        gen_func.__name__=workingfunc.__name__
        gen_func.__doc__=workingfunc.__doc__

        return gen_func

    def all_arrays(self):
        """Load all files and return a KerrList with only arrays in it. Better
        if we're going to append and delete items"""
        x=[None for i in self] # iterating over self causes everything to be loaded
        return self

    def apply_all(self, func, *args, **kwargs):
        """Apply a function to all images in list"""
        if isinstance(func,string_types):
            f=getattr(self,func) # objectFolder handles this for us.
            f(*args,**kwargs)
            retval=self
        elif callable(func):
            retval=[]
            for ix,f in enumerate(self):
                ret=func(f, *args, **kwargs)
                if isinstance(ret,np.ndarray):
                    ret=ret.view(type=self._type)
                    ret.metadata=f.metadata.copy()
                if isinstance(ret,self._type):
                    self[i]=ret
                else:
                    retval.append(ret)
            if len(retval)==0:
                retval=self
        else:
            raise ValueError('func must be a string or function')
        return retval

    def slice_metadata(self, key=None, values_only=False):
        """Return a list of the metadata dictionaries for each item/file

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
        """
        metadata=[k.metadata for k in self]
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


