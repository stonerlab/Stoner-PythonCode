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

IM_SIZE=(512,672) #Standard Kerr image size
AN_IM_SIZE=(554,672) #Kerr image with annotation not cropped

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
    """:py:class:`Stoner.Image.stack.ImageStack` is a 3d numpy array stack of images.

    This is used to deal with a stack of images with identical dimensions
    as a 3d numpy array - more efficient than the folder based methods in ImageFolder
    for iterating etc. The first axis is the number of images. Images are added
    and deleted through list type methods ImageStack.append, ImageStack.insert.
    
    Images can be accessed through ImageStack[i]
    Metadata through ImageStack['key']
    images should be added or removed using the list like methods append, insert, del
    
    Attributes:
        imarray(np.ndarray): 
            the 3d stack of images
        allmeta(list): 
            list of metadata stored for each image
        zipallmeta(dict): 
            a dictionary of all common metadata items zipped into lists
        clone(ImageStack): 
            copy of self
        shape(tuple): 
            pass through to imarray.shape
    """
        
    def __init__(self, *args, **kargs):
        """Constructor for :py:class:`Stoner.Image.stack.ImageStack`
        Args:
            (str, ndarray (3d or 2d), ImageFolder, ImageStack, list of array type):
                Various forms recognised. Will try to create as expected!
        Keyword Arguments:
            metadata: extra info to retain beyond the image metadata
            metadata_info: descriptor of how to add the individual image
                      metadata to the main metadata parameter.
                      'list': lists the metadata for each of the common keys
                      'average': attempts to save the average value of each meta key
            copyarray: whether to copy the image given in args[0]
                              
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
                pattern = kargs.get('pattern', '*.png')
                ims = ImageFolder(images, pattern=pattern)
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
        """The main array storing the images
        """
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
        """List of complete metadata for each image in ImageStack"""
        return self._allmeta
    
    @allmeta.setter
    def allmeta(self, value):
        self._allmeta=value
        self._update_metadata
        
    @property
    def clone(self):
        """Return a copy of self
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
        """List of metadata values for each image given key.
        
        Return a list of metadata from each image given the key or a list of 
        keys. If values_only then return a list of values, else return a dictionary entry
        
        Arg:
            key(str, list, tuple):
                the key or list of keys to return values for
        Keyword Arguments:
            values_only(bool):
                whether to return values only (list, default) or a dictionary type
        Returns:
            (list,dict):
                sliced metadata. Form depending on values_only
        """
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
        """Append an image array
        
        Append an image to the end of the stack and update allmeta
        Arg:
            item(ndarray or ImageArray):
                array of same shape as other images in stack
        """
                
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
        """Insert an image array at index
        
        Insert an image into the stack at index and update allmeta.
        
        Args:
            index(int):
                index for insertion
            item(ndarray or ImageArray):
                array of same shape as other images in stack
        """
        
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
        
        Keyword Arguments:
            clip_negative(bool):
                If True, clip the negative range (i.e. return 0 for min intensity)
                even if the image dtype allows negative values.
        Returns:
            (imin,imax) (tuple):
                Lower and upper intensity limits.
        """
        imin, imax = dtype_range[self.imarray.dtype.type]
        if clip_negative:
            imin = 0
        return imin, imax
    
    def clip_intensity(self):
        """Clip intensity that lies outside the range allowed by dtype.
        
        Most useful for float where pixels above 1 are reduced to 1.0 and -ve pixels
        are changed to 0. (Numpy should limit the range on arrays of int dtypes."""
        dl=self.dtype_limits(clip_negative=True)
        np.clip(self.imarray, dl[0], dl[1], out=self.imarray)
    
    def convert_float(self, clip_negative=True):
        """Convert the imarray to floating point type normalised to -1 to 1. 
        If clip_negative then clip intensities below 0 to 0.
        
        Keyword Arguments:
            clip_negative(bool):
                whether to clip intensities
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
        """apply function to all images in the stack
        
        Args:
            func(string or callable):
                if string it must be a function reachable by ImageArray
            quiet(bool):
                if False print '.' for every iteration
        
        Note:
            Further args, kargs are passed through to the function
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
    
    def subtract(self, background, contrast=16, clip_intensity=True):
        """Subtract a background image (or index) from all images in the stack.
        
        The formula used is new = (ImageArray - background) * contrast + 0.5
        If clip_intensity then clip negative intensities to 0. Array is always
        converted to float for this method.
        
        Arg:
            background(int or 2d np.ndarray):
                the background image to subtract. If int is given this is used
                as an index on the stack.
        Keyword Arguments:
            contrast(int):
                Default 16. Magnifies the subtraction
            clip_intensity(bool):
                whether to clip the image intensities in range (0,1) after subtraction
        """
        self.convert_float(clip_negative=False)
        if isinstance(background, int):
            bg=self[background]
        bg = bg.view(ImageArray).convert_float(clip_negative=False)
        bg=np.tile(bg, (len(self),1,1))
        self.imarray = contrast * (self.imarray - bg) + 0.5
        if clip_intensity:
            self.clip_intensity()
    
    def crop_stack(self, box):
        """Crop the imagestack.
        Crops to the box given

        Args:
            box(array or list of type int):
                [xmin,xmax,ymin,ymax]

        Returns:
            (ImageStack):
                cropped images
        """
        self.imarray = self.imarray[:,box[2]:box[3],box[0]:box[1]]
        
    def show(self):
        """Show the stack of images in a skimage CollectionViewer window"""
        #stackims=[self[i] for i in range(len(self))]
        cv=CollectionViewer(self)
        cv.show()
        return cv  

    def save(self, fname):
        """probably should save in hdf5 format"""
        raise NotImplemented
        
    def stddev(self, weights=None):
        """calculate weighted standard deviation for stack
        This is a biased standard deviation, may not be appropriate for small sample sizes
        """
        avs = self.stack_average(weights=weights)
        avs = np.tile(avs, (len(self),1,1)) #make it the same size as imarray
        sumsqrdev = np.sum(weights*(self.imarray - avs)**2,axis=0)
        result = np.sqrt(sumsqrdev/(np.sum(weights, axis=0))) 
        return result.view(ImageArray)
    
    def stderr(self, weights=None):
        """Standard error in the stack average
        """
        serr = self.stddev(weights=weights)/np.sqrt(self.shape[0])
        return serr

    def average(self, weights=None):
        """Get an array of average pixel values for the stack.
        Pass through to numpy average        
        Returns:
            average(ImageArray):
                average values
        """                   
        average = np.average(self.imarray, axis=0, weights=weights)
        return average.view(ImageArray)

    def correct_drifts(self, refindex, threshold=0.005, upsample_factor=50, box=None):
        """Align images to correct for image drift.
        
        Pass through to ImageArray.corret_drift.
        
        Arg:
            refindex: int or str
                index or name of the reference image to use for zero drift
        Keyword Arguments:
            threshold(float): see ImageArray.correct_drift
            upsample_factor(int): see ImageArray.correct_drift
            box: see ImageArray.correct_drift
            
        """
        ref=self[refindex]
        self.apply_all('correct_drift', ref, threshold=threshold,
                     upsample_factor=upsample_factor, box=box)        


class KerrStack(ImageStack):
    """:py:class:`Stoner.Image.stack.KerrStack is similar to ImageStack but adds
    some functionality particular to Kerr images.
    
    Attributes:
        fields(list):
            list of applied fields in stack. This is the most important metadata
            for things like hysteresis.    
    """
        
    def __init__(self, *args, **kargs):
        super(KerrStack, self).__init__(*args, **kargs)
        self.convert_float()
        if 'field' in self.zipallmeta.keys():
            self.fields = np.array(self.zipallmeta['field'])
        else:
            self.fields = np.arange(len(self))
#    
    def hysteresis(self, mask=None):
        """Make a hysteresis loop of the average intensity in the given images
    
        Keyword Argument:
            mask(ndarray or list):
                boolean array of same size as an image or imarray or list of 
                masks for each image. If True then don't include that area in 
                the intensity averaging.
    
        Returns
        -------
        hyst(Data):
            'Field', 'Intensity', 2 column array
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
    
    def reverse(self):
        """Reverse the image order
        """
        self.imarray = self.imarray[::-1,:,:]
        self.fields = self.fields[::-1]
                              
    def denoise_thresh(self, denoise_weight=0.1, thresh=0.5, invert=False):
        """apply denoise then threshold images.
        Return a new MaskStack.
        True for values greater than thresh, False otherwise
        else return True for values between thresh and 1"""
        masks=self.clone
        masks.apply_all('denoise', weight=0.1)
        masks.apply_all('threshold_minmax', threshmin=thresh, 
                        threshmax=np.max(masks.imarray))        
        masks=MaskStack(masks)
        if invert:
            masks.imarray = np.invert(masks.imarray)
        return masks
    
    def find_threshold(self, testim=None, mask=None):
        """Try to find the threshold value at which the image switches. Takes
        it as the median value of the testim. Masks values
        where the difference is less than tolerance in case part of the image is
        irrelevant.
        """
        if testim is None:
            testim = self[len(self)/2]
        else:
            testim = self[testim]
        if mask is None:
            med = np.median(testim)
        else:
            med = np.median(np.ravel(testim[np.invert(mask)]))
        return med
        
    def stable_mask(self, tolerance=1e-2, comparison = None):
        """Produce a mask of areas of the image that are changing little over the
        stack. comparison is an optional tuple that gives the index of two images
        to compare, otherwise first and last used. tolerance is the difference
        tolerance"""
        mask = np.zeros(self[0].shape, dtype=bool)
        mask[abs(self[-1]-self[0])<tolerance] = True
        return mask
    
    def crop_text(self, copy=False):
        """Crop the bottom text area from a standard Kermit image stack
        Returns:
        (self):
            cropped image
        """

        assert self[0].shape==AN_IM_SIZE or self[0].shape==IM_SIZE, \
                'Need a full sized Kerr image to crop' #check it's a normal image
        crop=(0,IM_SIZE[1],0,IM_SIZE[0])
        self.crop_stack(box=crop)
    
    def HcMap(self, threshold=0.5, correct_drift=False, baseimage=0, quiet=False, 
              saturation_end=True, saturation_white=True, extra_info=False):
        """produce a map of the switching field at every pixel in the stack.
        It needs the stack to start saturated one way and end saturated the other way.
        
        Keyword Arguments:
            threshold(float):
                the threshold value for the intensity switching. This will need to 
                be tuned for each stack
            correct_drift(bol):
                whether to correct drift on the image stack before proceding
            baseimage(int or ImageArray):
                we use drift correction from the baseimage.
            saturation_end(bool):
                last image in stack is closest to saturation 
            saturation_white(bool):
                bright pixels are saturated dark pixels are not yet switched
            quiet: bool
                choose wether to output status updates as print messages
            extra_info(bool):
                choose whether to return intermediate calculation steps as an extra dictionary        
        Returns:
            (ImageArray): The map of field values for switching of each pixel in the stack
        """
        ks=self.clone
        if isinstance(baseimage,int):
            baseimage = self[baseimage].clone
        elif isinstance(baseimage,np.ndarray):
            baseimage = baseimage.view(ImageArray)
        if correct_drift:
            ks.apply_all('correct_drift', ref=baseimage, quiet=quiet)
            if not quiet: print('drift correct done')
        masks = self.denoise_thresh(denoise_weight=0.1, thresh=threshold, invert=not(saturation_white))
        if not quiet: print 'thresholding done'  
        si,sp = masks.switch_index(saturation_end=saturation_end)       
        Hcmap=ks.index_to_field(si)
        Hcmap[Hcmap==ks.fields[0]]=0 #not switching does not give us a Hc value
        if extra_info:
            ei={'switch_index':si, 'switch_array':sp, 'masks':masks}
            return Hcmap, ei
        return Hcmap

    def average_Hcmap(self, weights=None, ignore_zeros=False):
        """Get an array of average pixel values for the stack.
        Return average of pixel values in the stack.
        
        Keyword arguments:
            ignore zeros(bool):
                Weight zero values in an image as 0 in the averaging.
        
        Returns:
            average(ImageArray):
                average values
        """
        if ignore_zeros:
            weights=self.clone
            weights.imarray = weights.imarray.astype(bool).astype(int) #1 if Hc isn't zero, zero otherwise
            condition=np.sum(weights,axis=0)==0 #stop zero division error
            for m in range(self.shape[0]):
                weights[m] = np.select([condition, np.logical_not(condition)],
                                 [np.ones_like(weights[m]),weights[m]])
            #weights means we only account for non-zero values in average                     
        average = np.average(self.imarray, axis=0, weights=weights)
        return average.view(ImageArray)    
    
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
        
        Keyword Arguments:
            saturation_end(bool):
                True if the last image is closest to the fully saturated state. 
                False if you want the first image
            saturation_value(bool):
                if True then a pixel value True means that switching has occured
                (ie magnetic saturation would be all True)
            
        Returns:
            switch_ind: MxN ndarray of int
                index that each pixel switches at
            switch_progession: MxNx(P-1) ndarray of bool
                stack of masks showing when each pixel saturates
        
        """
        ms=self.clone
        if not saturation_end:
            ms = ms.reverse()
        #arr1 = ms[0].astype(float) #find out whether True is at begin or end
        #arr2 = ms[-1].astype(float)
        #if np.average(arr1)>np.average(arr2): #OK so it's bright at the start
        if not saturation_value:
            self.imarray = np.invert(ms.imarray) #Now it's bright (True) at end
        switch_ind=np.zeros(ms[0].shape, dtype=int)
        switch_prog=self.clone
        switch_prog.imarray=np.zeros(self.shape, dtype=bool)
        del(switch_prog[-1])
        for m in reversed(range(len(ms)-1)): #go from saturation backwards
            already_done=np.copy(switch_ind).astype(dtype=bool) #only change switch_ind if it hasn't already
            condition=np.logical_and( ms[m]!=True, ms[m+1]==True )
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
