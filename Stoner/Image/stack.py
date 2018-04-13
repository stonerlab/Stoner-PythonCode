# -*- coding: utf-8 -*-
"""Provide variants of :class:`Stoner.Image.ImageFolder` that store images efficiently in 3D numpy arrays."""
__all__ = ["ImageStackMixin","ImageStack2","ImageStack"]
import numpy as np
import copy
import numbers
import warnings

from skimage.viewer import CollectionViewer
from Stoner.compat import  string_types,int_types
from Stoner.tools import all_type
from .core import ImageArray,dtype_range, ImageFile
from .folders import ImageFolder,ImageFolderMixin
from Stoner.Core import regexpDict,metadataObject,typeHintedDict
from Stoner.Folders import DiskBasedFolder, baseFolder
from Stoner.Image.util import convert

IM_SIZE=(512,672) #Standard Kerr image size
AN_IM_SIZE=(554,672) #Kerr image with annotation not cropped

def _load_ImageArray(f, **kargs):
    kargs.pop("Img_num",None) # REemove img_num if it exists
    return ImageArray(f, **kargs)

def _average_list(listob):
    """Average a list of items picking an appropriate average given the type.

    If no appropriate average is found None will be returned.
    if listob contains nested lists or dicts of numbers then try to average
    individual items within the lists/keys.
    """
    if len(listob)==0:
        return None
    if not all_type(listob,type(listob[0])):
        return None #all of the list isn't the same type
    typex = listob[0]
    if isinstance(typex, numbers.Number):
        ret = sum(listob)/float(len(listob))
    elif isinstance(typex, np.ndarray):
        try:
            ret = np.average(tuple(listob))
        except Exception: #probably incompatible array sizes
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

class ImageStackMixin(object):

    """Implement an interface for a baseFolder to store images in a 3D numpy array for faster access."""


    def __init__(self,*args,**kargs):
        """Initialise an ImageStack's pricate data and provide a type argument."""
        self._stack=np.atleast_3d(np.ma.MaskedArray([]))
        self._metadata=regexpDict()
        self._names=list()
        self._sizes=np.array([],dtype=int).reshape(0,2)

        kargs["type"]=ImageFile
        
        if not len(args):
            super(ImageStackMixin,self).__init__(**kargs)
            return None # No further initialisation
        other=args[0]
        if isinstance(other,ImageStackMixin):
            super(ImageStackMixin,self).__init__(*args[1:],**kargs)
            self._stack=other._stack
            self._metadata=other._metadata
            self._names=other._names
            self._sizes=other._sizes
        elif isinstance(other,ImageFolder): #ImageFolder can already init from itself
            super(ImageStackMixin,self).__init__(*args,**kargs)
        elif isinstance(other,np.ndarray) and len(other.shape)==3: #Initialise with 3D numpy array, first coordinate is number of images
            super(ImageStackMixin,self).__init__(*args[1:],**kargs)
            self.imarray=other
            self.imarray.shape
            self._sizes=np.ones((other.shape[0],2), dtype=int)*other.shape[1:]
            self._names=["Untitled-{}".format(d) for d in range(other.shape[0])]
            for n in self._names:
                self._metadata[n]=typeHintedDict()
        elif isinstance(other,list):
            try:
                other = [ImageFile(i) for i in other]
            except:
                raise ValueError('Failed to initialise ImageStack with list input')
            super(ImageStackMixin,self).__init__(*args[1:],**kargs)
            for ot in other:
                self.append(ot)
            del(self[-1]) #Bit of a hack to get rid of initialised zeros data - 
                          #this poss needs changing in the append method
        else:
            super(ImageStackMixin,self).__init__(*args,**kargs)
            
    def __lookup__(self,name):
        """Stub for other classes to implement.

        Parameters:
            name(str): Name of an object

        Returns:
            A key in whatever form the :py:meth:`baseFolder.__getter__` will accept.

        Note:
            We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!
        """
        if isinstance(name,int_types):
            try:
                _=self._stack[:,:,name]
            except IndexError:
                raise KeyError("{} is out of range for accessing the ImageStack.".format(name))
            return name
        elif name not in self.__names__():
            name=self._metadata.__lookup__(name)
        return list(self._metadata.keys()).index(name) #return the matching index of the name

    def __names__(self):
        """Stub method to return a list of names of all objects that can be indexed for __getter__.

        Note:
            We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!
        """
        return self._names

    def __getter__(self,name,instantiate=True):
        """Stub method to do whatever is needed to transform a key to a metadataObject.

        Parameters:
            name (key type): The canonical mapping key to get the dataObject. By default
                the baseFolder class uses a :py:class:`regexpDict` to store objects in.

        Keyword Arguments:
            instatiate (bool): If True (default) then always return a metadataObject. If False,
                the __getter__ method may return a key that can be used by it later to actually get the
                metadataObject. If None, then will return whatever is helf in the object cache, either instance or name.

        Returns:
            (metadataObject): The metadataObject

            Note:
            We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!


        """
        try:
            idx=self.__lookup__(name)
        except KeyError: # If we don't seem to have the name then see if we can fall back to something else like a DiskBasedFolder
            return super(ImageStackMixin,self).__getter__(name,instantiate)
        if isinstance(instantiate,bool) and not instantiate:
            return self.__names__()[idx]
        else:
            instance= self._instantiate(idx)
            return self._update_from_object_attrs(instance)

    def __setter__(self,name,value,force_insert=False):
        """Stub to setting routine to store a metadataObject.

        Parameters:
            name (string) the named object to write - may be an existing or new name
            value (metadataObject) the value to store.

        Note:
            We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!
        """
        if isinstance(name,int_types):
            try:
                name=self.__names__()[name]
            except IndexError:
                name=self.make_name(value)
        if name is None:
            name=self.make_name(value)
        try:
            if force_insert:
                raise KeyError("Fake force insert")
            idx=self.__lookup__(name)
        except KeyError: #Ok we're appending here
            if isinstance(value,string_types): # Append with a filename, call __getter__
                value=self.__getter__(value,instantiate=True) # self.__getter__ will also insert if necessary
                return None
            else: # Append with real value
                idx=len(self)
                return self.__inserter__(idx,name,value)
        else:
            value = self.type(value) #ensure type if a bare numpy array was given
            self._sizes[idx]=value.shape
        self._metadata[name]=value.metadata
        if hasattr(value,"image"):
            value=value.image
        row,col=value.shape
        pag=len(self._sizes)
        new_size=self.max_size+(pag,)
        self._resize_stack(new_size)
        self._stack[:row,:col,idx]=value

    def __inserter__(self,ix,name,value):
        """Provide an efficient insert into the stack.

        The default implementation is rather slow about inserting since it has to clear the data folder and then rebuild it entry by entry. This does
        a simple insert."""
        value = ImageFile(value) #ensure we have some metadata
        self._names.insert(ix,name)
        self._metadata[name]=value.metadata
        self._sizes=np.insert(self._sizes,ix,value.shape,axis=0)
        new_size=self.max_size+(len(self._names),)
        self._resize_stack(new_size)       
        self._stack=np.insert(self._stack,ix,np.zeros(self.max_size),axis=2)
        row,col = value.shape
        self._stack[:row,:col,ix] = value.data


    def __deleter__(self,ix):
        """Deletes an object from the baseFolder.

        Parameters:
            ix(str): Index to delete, should be within +- the lengthe length of the folder.

        Note:
            We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!

        """
        idx=self.__lookup__(ix)
        name=list(self.__names__())[idx]
        del self._metadata[name]
        self._stack=np.delete(self._stack,idx,axis=2)
        del self._names[idx]
        self._sizes=np.delete(self._sizes,ix,axis=0)

    def __clear__(self):
        """"Clears all stored :py:class:`Stoner.Core.metadataObject` instances stored.

        Note:
            We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!

        """
        self._metadata=regexpDict()
        self._stack=np.atleast_3d(np.ma.MaskedArray([]))

    def __clone__(self,other=None,attrs_only=False):
        """Do whatever is necessary to copy attributes from self to other.

        Note:
            We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!
        """
        if other is None:
            other=self.__class__()
        if not attrs_only:
            other._metadata=copy.deepcopy(self._metadata)
            other._stack=copy.deepcopy(self._stack)
            other._names=copy.deepcopy(self._names)
            other._sizes=np.copy(self._sizes)
        return super(ImageStackMixin,self).__clone__(other=other,attrs_only=attrs_only)

    ###########################################################################
    ###################      Private methods     ##############################

    def _instantiate(self,idx):
        """Reconstructs the data type."""
        tmp=self.type()
        r,c=self._sizes[idx]
        tmp.data=self._stack[:r,:c,idx]
        tmp.metadata=self._metadata[self.__names__()[idx]]
        tmp._fromstack = True
        return tmp

    def _resize_stack(self,new_size):
        """Create a new stack with a new size."""
        old_size=self._stack.shape
        if old_size==new_size:
            return new_size
        row,col,pag=tuple([min(o,n) for o,n in zip(old_size,new_size)])
            
        new=np.ma.zeros(new_size)
        new[:row,:col,:pag]=self._stack[:row,:col,:pag]
        self._stack=new
        return row,col,pag        

    ###########################################################################
    ################### Properties of ImageStack ##############################

    @property
    def imarray(self):
        """"Produce the 3D stack of images - as [image,x,y]"""
        return np.transpose(self._stack,(2,0,1))

    @imarray.setter
    def imarray(self,value):
        value=np.ma.MaskedArray(np.atleast_3d(value))
        self._stack=np.transpose(value,(1,2,0))

    @property
    def max_size(self):
        if np.prod(self._sizes.shape)==0:
            return(0,0)
        return (self._sizes[:,0].max(),self._sizes[:,1].max())

    @property
    def shape(self):
        x,y,z=self._stack.shape
        return (z,x,y)

    ###########################################################################
    ###################         Public  methods         #######################


    def asfloat(self, normalise=True, clip=False, clip_negative=False):
        """Convert stack to floating point type.
        Analagous behaviour to ImageFile.asfloat()

        If currently an int type and normalise then floats will be normalised 
        to the maximum allowed value of the int type.
        If currently a float type then no change occurs. 
        If clip_negative then clip values outside the range 0,1

        Keyword Arguments:
            normalise(bool):
                normalise the image to the max value of current int type
            clip(bool):
                clip resulting range to values between -1 and 1
            clip_negative(bool):
                clip range further to 0,1
        """
        if self.imarray.dtype.kind=='f':
            pass
        else:
            self._stack = convert(self._stack, dtype=np.float64, normalise=normalise)
        if clip or clip_negative:
            self.clip_intensity(clip_negative=clip_negative)
    
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
        imin, imax = dtype_range[self._stack.dtype.type]
        if clip_negative:
            imin = 0
        return imin, imax

    ###########################################################################
    ################### Depricated Compaibility methods #######################

    @property
    def allmeta(self):
        """List of complete metadata for each image in ImageStack"""
        warnings.warn("allmeta is depricated in favour of ImageStack.metadata.all")
        return self.metadata.all

    @allmeta.setter
    def allmeta(self, value):
        """List of complete metadata for each image in ImageStack"""
        warnings.warn("allmeta is depricated in favour of ImageStack.metadata.all")
        self.metadata.all=value
        
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
        warnings.warn("correct_drift is a depricated method for an image stack - consider using align.")
        ref=self[refindex]
        self.apply_all('correct_drift', ref, threshold=threshold,
                     upsample_factor=upsample_factor, box=box)

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
        warnings.warn("crop_stack is depricated - sam effect can be achieved with crop(box)")
        self.crop(box)

    def show(self):
        """Pass through to :py:meth:`Stoner.Image.ImageFolder.view`"""
        warnings.weanr("show() is depricated in favour of ImageFolder.view()")
        return self.view()

class StackAnalysisMixin(object):
    """Add some analysis capability to ImageStack. These functions may override
       ImageFile functions but do them efficiently for a numpy stack of 
       images.
       """
    
    def subtract(self, background, contrast=16, clip_intensity=True):
        """Subtract a background image (or index) from all images in the stack.

        The formula used is new = (ImageArray - background) * contrast + 0.5
        If clip_intensity then clip negative intensities to 0. Array is always
        converted to float for this method.

        Arg:
            background(int or np.ndarray or ImageFile):
                the background image to subtract. If int is given this is used
                as an index on the stack.
        Keyword Arguments:
            contrast(float):
                Determines contrast of resulting image
            clip_intensity(bool):
                whether to clip the image intensities in range (0,1) after subtraction
        """
        self.asfloat(normalise=True, clip_negative=False)
        if isinstance(background, int):
            bg=self[background]
        if isinstance(bg.ImageFile):
            bg=bg.image
        bg = bg.view(ImageArray).asfloat(normalise=True, clip_negative=False)
        bg=np.tile(bg, (1,1,len(self)))
        self._stack = contrast * (self._stack - bg) + 0.5
        if clip_intensity:
            self.clip_intensity()
            
    

class ImageStack2(StackAnalysisMixin, ImageStackMixin,ImageFolderMixin,DiskBasedFolder,baseFolder):

    """An akternative implementation of an image stack based on baseFolder."""

    pass

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
        warnings.warn("""ImageStack is a depricated class.
                      
        Please conider using ImageStack2 which is based on ImageFolder and DataFolder and has a number of advantages.
        
        Please report any missing functionality that is in ImageStack and not ImageStack2 as an issue on github. """)
        
        super(ImageStack, self).__init__(  \
                    metadata=kargs.pop('metadata',None)) #initialise metadata
        self._len = 0 #lock on adhoc imarray adjustments
        self.imarray = np.zeros((0,0,0))
        self.metadata_info = kargs.pop('metadata_info', 'list')
        self.allmeta = [] #A list of metadata dicts extracted from the input images,
        #                  if just a 3d numpy array is given this will be empty.
        #                  this is for easy reconstruction if we want to get images
        #                  back out of ImageStack
        self.zipallmeta = {} #dict of allmeta in list form (only keys that
        #                       are common to all images are retained)
        copyarray = kargs.pop('copyarray', False)
        self._commonkeys = [] #metadata fields that are common to all images
        self['metadata_info'] = self.metadata_info
        if len(args)!=0:
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
        """call through to ndarray.shape"""
        return self.imarray.shape

    @property
    def imarray(self):
        """The main array storing the images"""
        return self._imarray

    @imarray.setter
    def imarray(self, arr):
        """Set the main array for the image."""
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
        self._update_metadata()

    @property
    def clone(self):
        """Return a copy of self"""
        return copy.deepcopy(self)

    def __iter__(self):
        """Iterating method."""
        return self.__next__()

    def __next__(self):
        """Python 3 style iterator interface."""
        for i in range(len(self)):
            yield self[i]

    def _update_commonkeys(self):
        """update self._commonkeys common keys in self.allmeta"""
        if len(self._allmeta)>0:
            keys = set(self.allmeta[0].keys())
            for m in self.allmeta:
                self._commonkeys = keys & set(m.keys()) #intersection
        else:
            self._commonkeys = set()

    def _update_zipallmeta(self):
        """list of metadata items for each common key"""
        self._update_commonkeys()
        for k in self._commonkeys:
            self.zipallmeta[k] = [i[k] for i in self.allmeta]

    def _update_metadata(self):
        """update the metadata from allmeta according to the specifications in  metadata_info"""
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
        """Patch indexing of strings to metadata and an index to a single ImageArray"""
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
        """Patch string index through to metadata. All other set operations to be handled by insert, append and del."""
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
        """Patch indexing of strings to metadata and index to single image and slice to multiple image"""
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
        """Define a length for the array."""
        return self.imarray.shape[0]

    def __deepcopy__(self, memo):
        """Support copy.deepcopy."""
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
        are changed to 0. (Numpy should limit the range on arrays of int dtypes.
        """
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
        quiet = kargs.pop('quiet', True)
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
        raise NotImplementedError

    def stddev(self, weights=None):
        """Calculate weighted standard deviation for stack

        This is a biased standard deviation, may not be appropriate for small sample sizes
        """
        avs = self.stack_average(weights=weights)
        avs = np.tile(avs, (len(self),1,1)) #make it the same size as imarray
        sumsqrdev = np.sum(weights*(self.imarray - avs)**2,axis=0)
        result = np.sqrt(sumsqrdev/(np.sum(weights, axis=0)))
        return result.view(ImageArray)

    def stderr(self, weights=None):
        """Standard error in the stack average"""
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



