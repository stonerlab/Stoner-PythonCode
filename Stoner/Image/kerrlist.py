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

from skimage.viewer import CollectionViewer
from Stoner.Folders import objectFolder
from Stoner.compat import *
from Stoner.compat import string_types


GRAY_RANGE=(0,65535)  #2^16
IM_SIZE=(512,672)
AN_IM_SIZE=(554,672) #Kerr image with annotation not cropped


def _load_KerrArray(f,img_num=0, **kwargs):
    return KerrArray(f, **kwargs)

class KerrList(objectFolder):
    """KerrList groups functions that can be applied to a group of KerrImages.
    In general it is designed to behave pretty much like a normal python list.

    This version inherits from Stoner.Folders.objectFolder to provide grouping and iterator functions.
    """

    _type=KerrArray # class attribute to keep things happy
    read_means=False #Only to make grouping work

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
            for i,ka in enumerate(self):
                f=getattr(self[i],func) #check for KerrArray function
                self[i]=f(*args,**kwargs)
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

class KerrStack(object):
    """
    This is used to deal with a stack of images from the Kerr software presented
    as a 3d numpy array. The final axis is the number of images.    
    """
    
    def __init__(self, imagearray, fieldlist=None):
        """3d array stack of images
        Parameters
        ----------
        imagearray ndarray:
            the last axis is taken as the stack number
        fieldlist ndarray:
            list of field values
        """
        if isinstance(imagearray, KerrStack):
            self.imagearray=imagearray.imagearray
            self.fields=imagearray.fields
        else:
            self.imagearray=imagearray
            self.convert_float()
            if fieldlist is not None:
                fieldlist=np.array(fieldlist)
                assert len(fieldlist.shape)==1, 'fieldlist must be 1d'
                assert fieldlist.shape[0]==imagearray.shape[2], 'fieldlist shape incompatible with image array'
            else:
                fieldlist=np.arange(imagearray.shape[2])
            self.fields=fieldlist

    @property
    def shape(self):
        """call through to ndarray.shape
        """
        return self.imagearray.shape
    
    def __len__(self):
        return self.imagearray.shape[2]
        
    def __getitem__(self, i):
        return self.imagearray[:,:,i].view(type=KerrArray)
        
    def __setitem__(self, i, value):
        self.imagearray[:,:,i]=value  
    
    def convert_float(self):
        """convert array to floats between 0 and 1"""
        for i,im in enumerate(self):
            k=KerrArray(im)
            k=k.convert_float(clip_negative=True)
            self[i]=np.array(k)
    
    def clone(self):
        return KerrStack(np.copy(self.imagearray), fieldlist=np.copy(self.fields))        
    
    def subtract(self, background, contrast=16, clip_intensity=True):
        """subtract a background image from all images in the stack.
        If clip_intensity then clip negative intensities to 0"""
        for i,im in enumerate(self):
            new=contrast*(im-background)+0.5
            if clip_intensity:
                new=new.clip_intensity()
            self[i]=new
    
    def apply_all(self, func, quiet=True, *args, **kwargs):
        """apply function func to all images in the stack
        Parameters
        ----------
        func: string or callable
            if string it must be a function reachable by KerrArray
        quiet: bool
            if False print '.' for every iteration
        *args, **kwargs
            arguments for the function
        """
        if isinstance(func, string_types):
            for i, im in enumerate(self):
                k=KerrArray(im)
                f=getattr(k,func)
                self[i]=f(*args,**kwargs)
                if not quiet:
                    print('.')
        elif hasattr(func, '__call__'):
            for i, im in enumerate(self):
                self[i]=func(im, *args, **kwargs)
            if not quiet:
                print('.')
    
    def show(self):
        """show a stack of images in a CollectionViewer window"""
        stackims=[self[i] for i in range(len(self))]
        cv=CollectionViewer(stackims)
        cv.show()
        return cv
    
    def hysteresis(self, mask=None):
        """Make a hysteresis loop of the average intensity in the given images
    
        Parameters
        ----------
        mask: boolean array of same size as an image, or list of masks for each image
            if True then don't include that area in the hysteresis
    
        Returns
        -------
        hyst: nx2 np.ndarray
            fields, intensities, 2 column numpy array
        """
        ia=self.imagearray
        hys_length=len(self)       
        hyst=np.column_stack((self.fields,np.zeros(hys_length)))
        for i in range(hys_length):
            im=ia[:,:,i]
            if isinstance(mask, np.ndarray) and len(mask.shape)==2:
                hyst[i,1] = np.average(im[np.invert(mask.astype(bool))])
            elif isinstance(mask, np.ndarray) and len(mask.shape)==3:
                hyst[i,1] = np.average(im[np.invert(mask[:,:,i].astype(bool))])
            elif isinstance(mask, (tuple,list)):
                hyst[i,1] = np.average(im[np.invert(mask[i])])
            else:
                hyst[i,1] = np.average(im)
        return hyst
        
    
    def reverse(self):
        """reverse the stack order"""
        for i in reversed(range(len(self))):
            self.imagearray = self.imagearray[:,:,::-1]
            
    def index_to_field(self, index_map):
        """Convert an image of index values into an image of field values
        (needs fieldlist to be defined for the stack)
        """
        if self.fields is None:
            raise AttributeError('Field list is not defined in MaskStack')
        fieldvals=np.take(self.fields, index_map)
        return KerrArray(fieldvals)
        
class MaskStack(KerrStack):
    """Similar to KerrStack but made for stacks of boolean or binary images
    """
    def __init__(self, imagearray, field_list=None):
        super(MaskStack,self).__init__(imagearray, field_list)
        self.imagearray=self.imagearray.astype(bool)
    
    def switch_index(self, saturation_end=True, saturation_value=True):
        """Given a stack of boolean masks representing a hystersis loop find
        the stack index of the saturation field for each pixel.
        
        Take the final mask as all switched (or the first mask if saturation_end
        is False). Work back through the masks taking the first time a pixel 
        switches as its coercive field (ie the last time it switches before
        reaching saturation).
        Elements that start switched at the lowest measured field or never
        switch are given a zero index.
        
        At the moment it's set up to expect masks to be false when the sample is saturated
        at a high field
        
        Parameters
        ----------
        saturation_end: bool
            True if the last image is closest to the fully saturated state. 
            False if you want the first image
        saturation_value: bool
            if True then a pixel value True means that switching has occured
            (ie magnetic saturation would be all True)
            
        Returns
        -------
        switch_ind: MxN ndarray of int
            index that each pixel switches at
        switch_progession: MxNx(P-1) ndarray of bool
            stack of masks showing when each pixel saturates
        
        """
        ms=self.clone()
        if not saturation_end:
            ms=ms.reverse()
        switch_ind=np.zeros(ms[0].shape, dtype=int)
        switch_prog=self.clone()
        switch_prog.imagearray=np.zeros(
                    (self.shape[0],self.shape[1],self.shape[2]-1), dtype=bool)
        switch_prog.fields=switch_prog.fields[0:-1]
        for m in reversed(range(len(ms)-1)):
            already_done=np.copy(switch_ind).astype(dtype=bool) #only change switch_ind if it hasn't already
            condition=np.logical_and( ms[m]!=saturation_value, 
                                      ms[m+1]==saturation_value )
            condition=np.logical_and(condition, np.invert(already_done))
            condition=[condition, np.logical_not(condition)]
            choice=[np.ones(switch_ind.shape)*m, switch_ind] #index or leave as is
            switch_ind = np.select(condition, choice)       
            switch_prog[m]=already_done
        if not saturation_end:
            switch_ind=-switch_ind + len(self)-1 #should check this!
            switch_prog.reverse()
        switch_ind=KerrArray(switch_ind.astype(int))
        return switch_ind, switch_prog
        
