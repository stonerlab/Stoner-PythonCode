# -*- coding: utf-8 -*-
"""
Created on Mon May 23 12:05:59 2016

@author: phyrct
"""

from .core import ImageArray
from .core import dtype_range
from .folders import ImageFolder
from Stoner.Core import metadataObject
from Stoner.Util import Data
import numpy as np
from os import path
import copy

from skimage.viewer import CollectionViewer
from Stoner.compat import *
from Stoner.compat import string_types


def _load_ImageArray(f,img_num=0, **kargs):
    return ImageArray(f, **kargs)

def _average_list(listob):
    """Average a list of items picking an appropriate average given the type.
    If no appropriate average is found None will be returned.
    if listob contains nested lists or dicts of numbers then try to average
    individual items within the lists/keys.
    """
    if len(listob)==0:
        return None
    if not all([type(i)==type(listob[0]) for i in listob]):
        return None #all of the list isn't the same type
    typex = listob[0]
    if isinstance(typex, (int,long,float)):
            ret = sum(listob)/float(len(listob))
    elif isinstance(typex, np.ndarray):
        try:
            ret = np.average(tuple(listob))
        except: #probably incompatible array sizes
            ret = None
    elif isinstance(typex, (tuple,list)): #recursively go through sub lists averaging values
        nl = zip(*listob)
        ret = [_average_list(list(i)) for i in nl]
        if isinstance(typex, tuple):
            ret = tuple(ret)
    elif isinstance(typex, dict): #recursively go through dictionary keys averaging values
        ret = {}
        for k in typex.keys():
            ret[k] = _average_list([listob[i][k] for i in listob])
    elif isinstance(typex, string_types):
        if all(i==typex for i in listob):
            ret = listob[0] #all the same text return that string
        else:
            ret = None
    else:
        return None
    return ret
    

class ImageStack(metadataObject):
    """
    This is used to deal with a stack of images with identical dimensions
    as a 3d numpy array - more efficient than the folder based methods in ImageFolder
    for iterating etc. The first axis is the number of images. Images are added
    and deleted through list type methods ImageStack.append, ImageStack.insert
    """
        
    def __init__(self, *args, **kargs):
        """Implements the basic logic for the ImageStack
        kargs:
            metadata: extra info to retain beyond the image metadata
            metadata_info: descriptor of how to add the individual image
                      metadata to the main metadata parameter.
                      'list': lists the metadata for each of the common keys
                      'average': attempts to save the average value of each meta key
            copyarray: whether to copy the image given in args[0]
        Attributes:
            imarray: the 3d stack of images
            allmeta: list of metadata for each image
            zipallmeta: a dictionary of all common metadata items zipped into lists
            clone: copy of self
            shape: pass through to imarray.shape
                      
        """
        super(ImageStack, self).__init__(  \
                    metadata=kargs.pop('metadata',None)) #initialise metadata
        self._len = 0 #lock on adhoc imarray adjustments
        self.imarray = np.zeros((0,0,0))
        self.allmeta = [] #A list of metadata dicts extracted from the input images,
                          #if just a 3d numpy array is given this will be empty.
                          #this is for easy reconstruction if we want to get images 
                          #back out of ImageStack
        self.zipallmeta = {} #dict of allmeta in list form (only keys that 
                                #are common to all images are retained)
        self.metadata_info = kargs.pop('metadata_info', 'list')
        copyarray = kargs.pop('copyarray', False)
        self._commonkeys = [] #metadata fields that are common to all images        
        self['metadata_info'] = self.metadata_info
        print(isinstance(args[0], ImageFolder))
        if len(args)==0:
            pass  #add arrays later
        else:
            images=args[0]
            if isinstance(images, ImageStack):
                if copyarray:
                    images = images.clone
                for k, v in images.__dict__.items():
                    setattr(self, k, v)
            elif isinstance(images, string_types): #try for a folder name
                ims = ImageFolder(images)
                for i in ims:
                    self.append(i)
            elif isinstance(images, np.ndarray):
                if images.shape not in [2,3]:
                    raise ValueError('ImageStack takes a 2 or 3d numpy array. Array of shape {} given.'.format(images.shape))
                if images.shape==2: #one image given
                    self.append(images)
                elif images.shape==3:
                    if copyarray:
                        images = np.copy(images)
                    for i in range(images.shape[0]):
                        self.append(images[i,:,:].view(type=ImageArray))
            elif isinstance(args[0], (list, ImageFolder)):
                if isinstance(images, ImageFolder):
                    images=[i for i in images] #take the images from the top group
                types = [isinstance(i, np.ndarray) for i in images] #this covers ImageArray too
                if all(types):        
                    for i in images:
                        self.append(i.view(type=ImageArray)) #new memory space used for array 
                else:
                    raise ValueError('Bad input for ImageStack {}'.format(type(images)))
            else:
               raise ValueError('Bad input for ImageStack {}'.format(type(images)))         
             
    @property
    def shape(self):
        """call through to ndarray.shape
        """
        return self.imarray.shape
    
    @property
    def imarray(self):
        return self._imarray
    
    @imarray.setter
    def imarray(self, arr):
        if len(arr.shape)!=3:
            raise ValueError('Bad shape {} for imarray'.format(arr.shape))
        if arr.shape[0] != self._len:
            raise ValueError('Update imarray using append and insert methods')
        self._imarray = arr
    
    @property
    def allmeta(self):
        return self._allmeta
    
    @allmeta.setter
    def allmeta(self, value):
        self._allmeta=value
        self._update_metadata
        
    @property
    def clone(self):
        """Produce a copy of self
        """
        return copy.deepcopy(self)
    
    def __iter__(self):
        return self.__next__()
    
    def __next__(self):
        for i in range(len(self)):
            yield self[i]
    
    def _update_commonkeys(self):
        """update self._commonkeys
        common keys in self.allmeta
        """
        keys = set(self.allmeta[0].keys())
        for m in self.allmeta:
            self._commonkeys = keys & set(m.keys()) #intersection
    
    def _update_zipallmeta(self):
        """list of metadata items for each common key
        """
        self._update_commonkeys()
        for k in self._commonkeys:
            self.zipallmeta[k] = [i[k] for i in self.allmeta]
        
    def _update_metadata(self):     
        """update the metadata from allmeta according to the specifications in 
        metadata_info
        """
        self._update_zipallmeta()
        mi = self.metadata_info
        for cm in self._commonkeys:
            if mi=='list':
                self[cm] = self.zipallmeta[cm]
            elif mi=='average':
                self[cm] = _average_list(self.zipallmeta[cm])
            elif mi=='none':
                try:
                    del(self[cm])
                except KeyError:
                    pass
            else:
                raise NotImplementedError("metadata_info can't take all keywords yet")
        
    def __getitem__(self,index):
        """Patch indexing of strings to metadata and an index to a single 
        ImageArray"""
        if isinstance(index,string_types):
            return self.metadata[index]
        elif isinstance(index, int):
            ret = self.imarray[index,:,:].view(type=ImageArray)
            ret.metadata = self.allmeta[index]
            return ret
        elif isinstance(index, slice):
            ret = [self[i] for i in range(index.start, index.stop)]
            return ret
        return super(ImageStack,self).__getitem__(index)
        
    def __setitem__(self,index,value):
        """Patch string index through to metadata. All other set operations
        to be handled by insert, append and del."""
        if isinstance(index,string_types):
            self.metadata[index]=value
        elif isinstance(index,int):
            if not isinstance(value, np.ndarray) and len(value.shape)==2:
                raise ValueError('setitem recieved type {}, expected 2d array type.'.format(type(value)))
            self.imarray[index,:,:] = value
            self.allmeta[index] = value.view(ImageArray).metadata
            self._update_metadata()
        else:
            super(ImageStack,self).__setitem__(index,value)

    def __delitem__(self,index):
        """Patch indexing of strings to metadata and index to single image
        and slice to multiple image"""
        if isinstance(index,string_types):
            del self.metadata[index]
        elif isinstance(index, int):
            del(self.allmeta[index])
            self._update_metadata()
            self._len -= 1
            self.imarray = np.delete(self.imarray, index, axis=0)
        elif isinstance(index, slice):
            for i in range(index.start,index.stop):
                del(self[i])
        else:
            super(ImageStack,self).__delitem__(index)
            
    def __len__(self):
        return self.imarray.shape[0]

    def __deepcopy__(self, memo):
        cls = self.__class__
        ret = cls.__new__(cls)
        memo[id(self)] = ret
        for k,v in self.__dict__.items():
            setattr(ret, k, copy.deepcopy(v, memo))
        return ret
        
    def slice_metadata(self, key, values_only=True):
        """return a list of metadata from each image given the key or a list of 
        keys. If values_only then return a list of values, else return a dictionary entry"""
        if isinstance(key, string_types):
            if values_only:
                ret = self.zipallmeta[key]
            else:
                ret = {key:self.zipallmeta[key]}
        elif isinstance(key, (list,tuple)):
            ret = []
            for k in key:
                if values_only:
                    ret.append(self.zipallmeta[k])
                else:
                    ret.append({k:self.zipallmeta[k]})
        else:
            raise ValueError('slice_metadata does not except input of type {}'.format(type(key)))
        return ret
    
    def append(self, item):
        """append an image array"""
        if not isinstance(item, np.ndarray):
            raise ValueError('append expects array type, {} given.'.format(type(item)))
        if len(self)>0 and item.shape!=self[0].shape:
            raise ValueError('append accepts an array of shape {}, array of shape {} given.'.format(self[0].shape, item.shape))
        self.allmeta.append(item.view(type=ImageArray).metadata)
        self._update_metadata()
        self._len += 1
        if len(self)==0: #array empty so far
            self.imarray = np.array([item])  
        else:
            self.imarray = np.append(self.imarray, np.array([item]), axis=0)                
    
    def insert(self, index, item):
        """insert an image array at index"""
        if not isinstance(item, np.ndarray):
            raise ValueError('append expects array type, {} given.'.format(type(item)))
        if len(self)>0 and item.shape!=self[0].shape:
            raise ValueError('append accepts an array of shape {}, array of shape {} given.'.format(self[0].shape, item.shape))
        self.allmeta.insert(index, item.view(type=ImageArray).metadata)
        self._update_metadata()
        self._len += 1
        self.imarray = np.insert(self.imarray, index, np.array([item]), axis=0)

    def dtype_limits(self, clip_negative=True):
        """Return intensity limits, i.e. (min, max) tuple, of imarray dtype.
        Parameters
        ----------
        clip_negative : bool, optional
            If True, clip the negative range (i.e. return 0 for min intensity)
            even if the image dtype allows negative values.
            The default behavior (None) is equivalent to True.
        Returns
        -------
        imin, imax : tuple
            Lower and upper intensity limits.
        """
        imin, imax = dtype_range[self.imarray.dtype.type]
        if clip_negative:
            imin = 0
        return imin, imax
    
    def clip_intensity(self):
        """clip intensity that lies outside the range allowed by dtype.
        Most useful for float where pixels above 1 are reduced to 1.0 and -ve pixels
        are changed to 0. (Numpy should limit the range on arrays of int dtypes"""
        dl=self.dtype_limits(clip_negative=True)
        np.clip(self.imarray, dl[0], dl[1], out=self.imarray)
    
    def convert_float(self, clip_negative=True):
        """convert the imarray to floating point type normalised to -1 
        to 1. If clip_negative then clip intensities below 0 to 0.
        """
        if self.imarray.dtype.kind=='f': #already float
            pass
        else:
            dl=self.dtype_limits(clip_negative=False)
            new=self.imarray.astype(np.float64)
            new=new/float(dl[1])
            self.imarray = new
        if clip_negative:
            self.clip_intensity()

    def apply_all(self, *args, **kargs):
        """apply function func to all images in the stack
        Parameters
        ----------
        func: string or callable
            if string it must be a function reachable by ImageArray
        quiet: bool
            if False print '.' for every iteration
        funcargs, funckargs
            arguments for the function
        """
        args = list(args)
        func = args.pop(0)
        quiet = kargs.pop('quiet', False)
        if isinstance(func, string_types):
            for i, im in enumerate(self):
                f=getattr(im,func)
                self[i]=f(*args,**kargs)
                if not quiet:
                    print('.')
        elif hasattr(func, '__call__'):
            for i, im in enumerate(self):
                self[i]=func(im, *args, **kargs)
            if not quiet:
                print('.')
    
    def show(self):
        """show a stack of images in a skimage CollectionViewer window"""
        #stackims=[self[i] for i in range(len(self))]
        cv=CollectionViewer(self)
        cv.show()
        return cv  

    def save(self, fname):
        """probably should save in hdf5 format"""
        raise NotImplemented

class KerrStack(ImageStack):
    """add some functionality to ImageStack particular to Kerr images
    """
        
    def __init__(self, *args, **kargs):
        """
        Attributes:
            fields: if fields is in metadata this is a slice of the fields values
                     in a numpy array otherwise it is just arange(len(self))
        """
        super(KerrStack, self).__init__(*args, **kargs)
        self.convert_float()
        if 'fields' in self.zipallmeta.keys():
            self.fields = np.array(self.zipallmeta('field'))
        else:
            self.fields = np.arange(len(self))
    
    def subtract(self, background, contrast=16, clip_intensity=True):
        """subtract a background image (or index) from all images in the stack.
        If clip_intensity then clip negative intensities to 0
        """
        self.convert_float(clip_negative=False)
        if isinstance(background, int):
            bg=self[background]
        bg = bg.view(ImageArray).convert_float(clip_negative=False)
        bg=np.tile(bg, (len(self),1,1))
        self.imarray = contrast * (self.imarray - bg) + 0.5
        if clip_intensity:
            self.clip_intensity()
#    
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
        hyst=np.column_stack((self.fields,np.zeros(len(self))))
        for i in range(len(self)):
            im=self[i]
            if isinstance(mask, np.ndarray) and len(mask.shape)==2:
                hyst[i,1] = np.average(im[np.invert(mask.astype(bool))])
            elif isinstance(mask, np.ndarray) and len(mask.shape)==3:
                hyst[i,1] = np.average(im[np.invert(mask[i,:,:].astype(bool))])
            elif isinstance(mask, (tuple,list)):
                hyst[i,1] = np.average(im[np.invert(mask[i])])
            else:
                hyst[i,1] = np.average(im)
        d = Data(hyst, setas='xy')
        d.column_headers = ['Field', 'Intensity']
        return d
            
    def index_to_field(self, index_map):
        """Convert an image of index values into an image of field values
        """
        fieldvals=np.take(self.fields, index_map)
        return ImageArray(fieldvals)
    
    def correct_drifts(self, refindex, threshold=0.005, upsample_factor=50, box=None):
        """Align images to correct for image drift.
        
        Parameters
        ----------
        refindex: int or str
            index or name of the reference image to use for zero drift
        Other parameters see ImageArray.correct_drift
        """
        ref=self[refindex]
        self.apply_all('correct_drift', ref, threshold=threshold,
                     upsample_factor=upsample_factor, box=box)
                               
    
class MaskStack(KerrStack):
    """Similar to ImageStack but made for stacks of boolean or binary images
    """
    def __init__(self, *args, **kargs):
        super(MaskStack,self).__init__(*args, **kargs)
        self.imarray=self.imarray.astype(bool)
    
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
        ms=self.clone
        if not saturation_end:
            ms=reversed(ms)
        switch_ind=np.zeros(ms[0].shape, dtype=int)
        switch_prog=self.clone
        switch_prog.imarray=np.zeros(
                    (self.shape[0]-1,self.shape[1],self.shape[2]), dtype=bool)
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
        switch_ind=ImageArray(switch_ind.astype(int))
        return switch_ind, switch_prog
        
