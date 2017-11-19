
# -*- coding: utf-8 -*-
"""
Created on Tue May 03 14:31:14 2016

@author: phyrct

kermit.py

Implements ImageArray, a class for using and manipulating two tone images such
as those produced by Kerr microscopy (with particular emphasis on the Evico
software images).

Images from the Evico software are 2D arrays of 16bit unsigned integers.

"""

import numpy as np
import os
from copy import copy, deepcopy
from skimage import color,exposure,feature,io,measure,\
                    filters,graph,util,restoration,morphology,\
                    segmentation,transform,viewer, draw
from PIL import Image
from PIL import PngImagePlugin #for saving metadata
import matplotlib.pyplot as plt
from Stoner.Core import typeHintedDict,metadataObject,regexpDict
from Stoner.Image.util import convert
from Stoner import Data
from Stoner.tools import istuple,fix_signature
from Stoner.compat import python_v3,string_types,get_filedialog # Some things to help with Python2 and Python3 compatibility
import inspect
from functools import wraps
if python_v3:
    from io import BytesIO as StreamIO
else:
    from cStringIO import StringIO as StreamIO



dtype_range = {np.bool_: (False, True), 
               np.bool8: (False, True),
               np.uint8: (0, 255),
               np.uint16: (0, 65535),
               np.int8: (-128, 127),
               np.int16: (-32768, 32767),
               np.int64: (-2**63, 2**63 - 1),
               np.uint64: (0, 2**64 - 1),
               np.int32: (-2**31, 2**31 - 1),
               np.uint32: (0, 2**32 - 1),
               np.float16: (-1, 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}


    
class ImageArray(np.ma.MaskedArray,metadataObject):
    
    """:py:class:`Stoner.Image.core.ImageArray` is a numpy array like class with a metadata parameter and pass through to skimage methods. 
    
    ImageArray is for manipulating images stored as a 2d numpy array.
    It is built to be almost identical to a numpy array except for one extra
    parameter which is the metadata. This stores information about the image
    in a dictionary object for later retrieval.
    All standard numpy functions should work as normal and casting two types
    together should yield a ImageArray type (ie. ImageArray+np.ndarray=ImageArray)

    In addition any function from skimage should work and return a ImageArray.
    They can be called as eg. im=im.gaussian(sigma=2). Don't include the module
    name, just the function name (ie not filters.gaussian). Also omit the first
    image argument required by skimage.
    
    Attributes:
        metadata (dict):
            dictionary of metadata for the image
        clone (self):
            copy of self
        max_box (tuple):
            coordinate extent (xmin,xmax,ymin,ymax)
            
        
    For clarity it should be noted that any function will not alter the current
    instance, it will clone it first then return the clone after performing the
    function on it.

    A note on coordinate systems:
    For arrays the indexing is (row, column). However the normal way to index
    an image would be to do (horizontal, vert), which is the opposite.
    In ImageArray the coordinate system is chosen similar to skimage. y points
    down x points right and the origin is in the top left corner of the image.
    When indexing the array therefore you need to give it (y,x) coordinates
    for (row, column).

     ----> x (column)
    |
    |
    v
    y (row)

    eg I want the 4th pixel in the horizontal direction and the 10th pixel down
    from the top I would ask for ImageArray[10,4]

    but if I want to translate the image 4 in the x direction and 10 in the y
    I would call im=im.translate((4,10))

    """

    #Proxy attributess for storing imported functions. Only do the import when needed
    _func_proxy=None

    #extra attributes for class beyond standard numpy ones
    _extra_attributes_default = {'metadata': typeHintedDict({}),
                                 'filename': ''}
    
    #Default values for when we can't find the attribute already
    _defaults={"debug":False, "_hardmask":False}

    #now initialise class

    def __new__(cls, *args, **kargs):
        """Construct an ImageArray object.
        
        We're using __new__ rather than __init__ to imitate a numpy array as 
        close as possible.
        """
        
        
        if len(args) not in [0,1]:
            raise ValueError('ImageArray expects 0 or 1 arguments, {} given'.format(len(args)))
            
        ### Deal with kwargs
        array_arg_keys = ['dtype','copy','order','subok','ndmin',"mask"] #kwargs for array setup
        array_args = {k:kargs.pop(k) for k in array_arg_keys if k in kargs.keys()}
        user_metadata = kargs.pop('metadata',{})
        asfloat = kargs.pop('asfloat', False) or kargs.pop('convert_float',False) #convert_float for back compatability
        _debug = kargs.pop('debug', False)
                           
        ### 0 args initialisation
        if len(args)==0:
            ret = np.empty((0,0), dtype=float).view(cls)
        else:
            
            ### 1 args initialisation
            arg = args[0]
            loadfromfile=False
            if isinstance(arg, cls):
                ret = arg           
            elif isinstance(arg, np.ndarray):
                # numpy array or ImageArray)
                if len(arg.shape)==2:
                    ret = arg.view(ImageArray)
                elif len(arg.shape)==1:
                    arg = np.array([arg]) #force 2d array
                    ret = arg.view(ImageArray)
                else:
                    raise ValueError('Array dimension 0 must be at most 2. Got {}'.format(len(arg.shape)))
                ret.metadata=getattr(arg,"metadata",typeHintedDict())
            elif isinstance(arg, bool) and not arg:
                patterns=(('png', '*.png'), ('npy','*.npy'))
                arg = get_filedialog(what='r',filetypes=patterns)
                if len(arg)==0:
                    raise ValueError('No file given')
                else:
                    loadfromfile=True
            elif isinstance(arg, string_types) or loadfromfile:
                # Filename- load datafile
                if not os.path.exists(arg):
                    raise ValueError('File path does not exist {}'.format(arg))
                ret = cls._load(arg, **array_args)
            elif isinstance(arg, ImageFile):
                #extract the image
                ret = arg.image
            else:
                try:  #try converting to a numpy array (eg a list type)
                    ret = np.asarray(arg, **array_args).view(cls)
                    if ret.dtype=='O': #object dtype - can't deal with this
                        raise ValueError
                except ValueError: #ok couldn't load from iterable, we're done
                    raise ValueError("No constructor for {}".format(type(arg)))
            if asfloat and ret.dtype.kind!='f': #convert to float type in place
                dl = dtype_range[ret.dtype.type]
                ret = np.array(ret, dtype=np.float64).view(cls)
                ret = ret / float(dl[1])
                
        #all constructors call array_finalise so metadata is now initialised
        if 'Loaded from' not in ret.metadata.keys():
            ret.metadata['Loaded from']=''
        ret.filename = ret.metadata['Loaded from']
        ret.metadata.update(user_metadata) 
        ret.debug = _debug
        return ret

    def __array_finalize__(self, obj):
        """__array_finalize__ and __array_wrap__ are necessary functions when subclassing numpy.ndarray to fix some behaviours. 
        
        See http://docs.scipy.org/doc/numpy-1.10.1/user/basics.subclassing.html for
        more info and examples
        Defaults below are only set when constructing an array using view
        eg np.arange(10).view(ImageArray). Otherwise filename and metadata
        attributes are just copied over (plus any other attributes set in 
        _optinfo).
        """
        if getattr(self, 'debug',False):
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
        _extra_attributes = getattr(obj, '_optinfo', 
                                deepcopy(ImageArray._extra_attributes_default))
        setattr(self, '_optinfo', copy(_extra_attributes))
        for k,v in _extra_attributes.items():
            setattr(self, k, getattr(obj, k, v))
        super(ImageArray,self).__array_finalize__(obj)

    def __array_prepare__(self,arr, context=None):
        return super(ImageArray,self).__array_prepare__(arr,context)

    def __array_wrap__(self, out_arr, context=None):
        """Part of the numpy array machinery.
        
        see __array_finalize__ for info. This is for if ImageArray is called 
        via ufuncs. array_finalize is called after.
        """
        ret=super(ImageArray,self).__array_wrap__(out_arr, context)
        return ret

    def __init__(self, *args, **kwargs):
        """Constructor method for :py:class:`ImageArray`.

        various forms are recognised

        .. py:function:: ImageArray('filename')
            :noindex:

            Creates the new ImageArray object and then loads data from the 
            given *filename*.

        .. py:function:: ImageArray(array)
            :noindex:

            Creates a new ImageArray object and assigns the *array*.
            
        .. py:function:: ImageArray(ImageFile)
            :noindex:

            Creates a new ImageArray object and assigns the *array* using
            the ImageFile.image property.


        .. py:function:: ImageArray(ImageArray)
            :noindex:

            Creates the new ImageArray object and initialises all data from the
            existing :py:class:`ImageArray` instance. This on the face of it does the same as
            the assignment operator, but is more useful when one or other of the
            ImageArray objects is an instance of a sub - class of ImageArray
        
        .. py:function:: ImageArray(bool)
            :noindex:

           if arg[0] is bool and False then open a file dialog to locate the
           file to open.
            
        Args:
            arg (positional arguments): an argument that matches one of the
                definitions above
        Keyword Arguments: All keyword arguments that match public attributes are
                used to set those public attributes eg metadata.
                
                asfloat(bool):
                    if True  and loading the image from file, convert the image 
                    to float values between 0 and 1 (necessary for some forms 
                    of processing)
        """
        super(ImageArray,self).__init__(*args,**kwargs)

    @classmethod
    def _load(cls,filename,**kargs):
        """Load an image from a file and return as a ImageArray."""
        fmt=kargs.pop("fmt",os.path.splitext(filename)[1][1:])
        handlers={"npy":cls._load_npy, "png":cls._load_png,"tiff":cls._load_tiff,"tif":cls._load_tiff}
        if fmt not in handlers:
            raise ValueError("{} is not a recognised format for loading.".format(fmt))
        ret=handlers[fmt](filename,**kargs)
        return ret
    
    @classmethod
    def _load_npy(cls,filename,**kargs):
        """Load image data from a numpy file."""
        image = np.load(filename)
        image = np.array(image, **kargs).view(cls)
        image.metadata["Loaded from"]=os.path.realpath(filename)
        image.filename=os.path.realpath(filename)
        return image

    @classmethod
    def _load_png(cls,filename,**kargs):
        with Image.open(filename,'r') as img:
            image=np.asarray(img).view(cls)
            # Since skimage.img_as_float() looks at the dtype of the array when mapping ranges, it's important to make
            # sure that we're not using too many bits to store the image in. This is a bit of a hack to reduce the bit-depth...
            if np.issubdtype(image.dtype,np.integer):
                bits=np.ceil(np.log2(image.max()))
                if bits<=8:
                    image=image.astype("uint8")
                elif bits<=16:
                    image=image.astype("uint16")
                elif bits<=32:
                    image=image.astype("uint32")
            for k in img.info:
                v=img.info[k]
                if "b'" in v: v=v.strip(" b'")    
                image.metadata[k]=v
        image.metadata["Loaded from"]=os.path.realpath(filename)
        image.filename=os.path.realpath(filename)
        return image

    @classmethod
    def _load_tiff(cls,filename,**kargs):
        with Image.open(filename,'r') as img:
            image=np.asarray(img).view(cls)
            # Since skimage.img_as_float() looks at the dtype of the array when mapping ranges, it's important to make
            # sure that we're not using too many bits to store the image in. This is a bit of a hack to reduce the bit-depth...
            if np.issubdtype(image.dtype,np.integer):
                bits=np.ceil(np.log2(image.max()))
                if bits<=8:
                    image=image.astype("uint8")
                elif bits<=16:
                    image=image.astype("uint16")
                elif bits<=32:
                    image=image.astype("uint32")
            tags=img.tag_v2
            if 270 in tags:
                from json import loads
                try:
                    metadata_string=tags[270]
                    metadata=loads(metadata_string)
                except Exception:
                    metadata=[]
            else:
                metadata=[]
            for line in metadata:
                parts=line.split("=")
                k=parts[0]
                v="=".join(parts[1:])
                image.metadata[k]=v
        image.metadata["Loaded from"]=os.path.realpath(filename)
        image.filename=os.path.realpath(filename)
        return image

        
#==============================================================
# Propety Accessor Functions
#==============================================================r
    @property
    def aspect(self):
        """Return the aspect ratio (width/height) of the image."""
        return float(self.shape[1])/self.shape[0]
    
    @property
    def centre(self):
        return tuple(np.array(self.shape)/2.0)

    @property
    def clone(self):
        """return a copy of the instance"""
        ret = ImageArray(np.copy(self))
        for k,v in self._optinfo.items():
            try:
                setattr(ret, k, deepcopy(v))
            except Exception:
                setattr(ret,k,copy(v))
        return ret

    @property
    def flat(self):
        """MaskedArray.flat doesn't work the same as array.flat."""
        return np.asarray(self).flat        

    @property
    def max_box(self):
        """return the coordinate extent (xmin,xmax,ymin,ymax)"""
        box=(0,self.shape[1],0,self.shape[0])
        return box

#    @property
#    def data(self):
#        """alias for image[:]. Equivalence to Stoner.data behaviour"""
#        return self[:]
    
    @property
    def CW(self):
        """Rotate clockwise by 90 deg."""
        return self.T[:,::-1]
    
    @property
    def CCW(self):
        """Rotate counter-clockwise by 90 deg."""
        return self.T[::-1,:]

    @property
    def _funcs(self):
        """Return an index of possible callable functions in other modules, caching result if not alreadty there.
        
        Look in Stoner.Image.imagefuncs, scipy.ndimage.* an d scikit.* for functions. We assume that each function
        takes a first argument that is an ndarray of image data, so with __getattrr__ and _func_generator we
        can make a bound method through duck typing."""
        if self._func_proxy is None: #Buyild the cache
            func_proxy=regexpDict() # Cache is a regular expression dictionary - keys matched directly and then by regular expression
            
            # Get the Stoner.Image.imagefuncs mopdule first
            from .import imagefuncs
            for d in dir(imagefuncs):
                if not d.startswith("_"):
                    func=getattr(imagefuncs,d)
                    if callable(func) and func.__module__==imagefuncs.__name__:
                        name="{}__{}".format(func.__module__,d).replace(".","__")
                        func_proxy[name]=func
                        
            #Add scipy.ndimage functions
            import scipy.ndimage as ndi
            _sp_mods=[ndi.interpolation,ndi.filters,ndi.measurements,ndi.morphology,ndi.fourier]
            for mod in _sp_mods:
                for d in dir(mod):
                    if not d.startswith("_"):
                        func=getattr(mod,d)
                        if callable(func) and func.__module__==mod.__name__:
                            func.transpose=True
                            name="{}__{}".format(func.__module__,d).replace(".","__")
                            func_proxy[name]=func
            #Add the scikit images modules
            _ski_modules=[color,exposure,feature,io,measure,filters,filters.rank, graph,
                    util, restoration, morphology, segmentation, transform,
                    viewer]
            for mod in _ski_modules:
                for d in dir(mod):
                    if not d.startswith("_"):
                        func=getattr(mod,d)
                        if callable(func):
                            name="{}__{}".format(func.__module__,d).replace(".","__")
                            func_proxy[name]=func
            self._func_proxy=func_proxy # Store the cache
        return self._func_proxy

#==============================================================
#function generator
#==============================================================
    def __dir__(self):
        """Implement code for dir()"""
        proxy=set(list(self._funcs.keys()))
        parent=set(dir(super(ImageArray,self)))
        mine=set(dir(ImageArray))
        return sorted(list(proxy|parent|mine))
    
    def __getattr__(self,name):
        """Magic attribute access method. 
        
        Tries first to get the attribute via a superclass call, if this fails
        checks for some well known attribute names and supplies missing defaults.
        
        To handle magic calls into other modules, we have a regular expression dicitonary
        that stores and index of callables by full name where the . are changed to __
        If we can get a match to __<name> then we get that callable from our index.
        
        The callable is then passed to self._func_generator for wrapping into a
        'on the fly' method of this class.

        TODO:
            An alternative nested attribute system could be something like
            http://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
            might be cool sometime.
        """
        ret=None
        try:
            ret=getattr(super(ImageArray,self),name)
        except AttributeError as e:
        #first check kermit funcs
            if name.startswith('_') or name in ["debug"]:
                if name=="_hardmask":
                    ret=False
            elif ".*__{}$".format(name) in self._funcs:
                ret=self._funcs[".*__{}$".format(name)]
                ret=self._func_generator(ret)
            if ret is None:
                raise AttributeError('No attribute found of name {}'.format(name))
        return ret
    
    def _func_generator(self,workingfunc):
        """Used by __getattr__ to wrap an arbitary callbable to make it a bound method of this class.
        
        Args:
            workingfunc (callable): The callable object to be wrapped.
            
        Returns:
            (function): A function with enclosure that holds additional information about this object.
            
        The function returned from here will call workingfunc with the first argument being a clone of this
        ImageArray. If the meothd returns an ndarray, it is wrapped back to our own class and the metadata dictionary
        is updated. If the function returns a :py:class:`Stoner.Data` object then this is also updated with our metadata.
        
        This method also updates the name and documentation strings for the wrapper to match the wrapped function - 
        thus ensuring that Spyder's help window can generate useful information.
        
        """
        @wraps(workingfunc)
        def gen_func(*args, **kwargs):
            transpose = getattr(workingfunc,"transpose",False)
            if transpose:
                change=self.clone.T
            else:
                change=self.clone
            r=workingfunc(change, *args, **kwargs) #send copy of self as the first arg
            if isinstance(r,Data):
                pass #Data return is ok
            elif isinstance(r,np.ndarray) and np.prod(r.shape)==np.max(r.shape): #1D Array
                r=Data(r)
                r.metadata=self.metadata.copy()
                r.column_headers[0]=workingfunc.__name__
            elif isinstance(r,np.ndarray): #make sure we return a ImageArray
                if transpose:
                    r=r.view(type=self.__class__).T
                else:
                    r=r.view(type=self.__class__)
                sm=self.metadata.copy() #Copy the currenty metadata
                sm.update(r.metadata) # merge in any new metadata from the call
                r.metadata=sm  # and put the returned metadata as the merged data
            #NB we might not be returning an ndarray at all here !
            return r
        return fix_signature(gen_func,workingfunc)

    @property
    def draw(self):
        """DrawProxy is an opbject for accessing the skimage draw sub module."""
        return DrawProxy(self)

#==============================================================================
# OTHER SPECIAL METHODS
#==============================================================================
    
    def __setattr__(self, name, value):
        """Set an attribute on the object."""
        super(ImageArray, self).__setattr__(name, value)
        #add attribute to those for copying in array_finalize. use value as
        #defualt.
        circ = ['_optinfo'] #circular references
        proxy = ['_funcs'] #can be reloaded for cloned arrays
        if name in circ + proxy: 
            #Ignore these in clone
            pass
        else:
            self._optinfo.update({name:value})
        
    def __getitem__(self,index):
        """Patch indexing of strings to metadata."""
        if getattr(self,"debug",False):
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
        if isinstance(index,ImageFile) and index.image.dtype==bool:
            index=index.image
        if isinstance(index,string_types):
            return self.metadata[index]
        else:
            return super(ImageArray,self).__getitem__(index)


    def __setitem__(self,index,value):
        """Patch string index through to metadata."""
        if isinstance(index,ImageFile) and index.dtype==bool:
            index=index.image
        if isinstance(index,string_types):
            self.metadata[index]=value
        else:
            super(ImageArray,self).__setitem__(index,value)

    def __delitem__(self,index):
        """Patch indexing of strings to metadata."""
        if isinstance(index,string_types):
            del self.metadata[index]
        else:
            super(ImageArray,self).__delitem__(index)

#==============================================================
#Now any other useful bits
#==============================================================

    
    def box(self, *args, **kargs):
        """alias for crop"""
        return self.crop(*args, **kargs)
    
    def crop_image(self, *args, **kargs):
        """back compatability"""
        return self.crop(*args, **kargs)

    def crop(self, *args, **kargs):
        """Crop the image. 
        
        This is essentially like taking a view onto the array
        but uses image x,y coords (x,y --> col,row)
        Returns a view according to the coords given. If box is None it will
        allow the user to select a rectangle. If a tuple is given with None
        included then max extent is used for that coord (analagous to slice).
        If copy then return a copy of self with the cropped image.
        
        Args:
            box(tuple) or 4 separate args or None:
                (xmin,xmax,ymin,ymax)
                If None image will be shown and user will be asked to select
                a box (bit experimental)
        Keyword Arguments:
            copy(bool):
                If True return a copy of ImageFile with the cropped image
        Returns:
            (ImageArray):
                view or copy of array asked for
        
        Example:
            a=ImageFile(np.arange(12).reshape(3,4))
            
            a.crop(1,3,None,None)
        """
        if len(args)==0 and 'box' in kargs.keys():
            args = kargs['box'] #back compatability
        elif len(args) not in (1,4):
            raise ValueError('crop accepts 1 or 4 arguments, {} given.'.format(len(args)))
        if len(args)==1:
            box = args[0]
            if box is None: #experimental
                print('Select crop area')
                box = self.draw_rectangle(box)
            elif isinstance(box, (tuple,list)) and len(box)==4:
                pass
            elif isinstance(box,int):
                box=(box,self.shape[1]-box,box,self.shape[0]-box)
            elif isinstance(box,float):
                box=[round(self.shape[1]*box/2),round(self.shape[1]*(1-box/2)),round(self.shape[1]*box/2),round(self.shape[1]*(1-box/2))]
                box=tuple([int(x) for x in box])
            else:
                raise ValueError('crop accepts tuple of length 4, {} given.'.format(len(box)))
        else:
            box = tuple(args)
        for i,item in enumerate(box): #replace None with max extent
            if item is None:
                box[i]=self.max_box[i]
        ret = self[box[2]:box[3],box[0]:box[1]]   
        if 'copy' in kargs.keys() and kargs['copy']:
            ret = ret.clone
        return ret
    
    def dtype_limits(self, clip_negative=True):
        """Return intensity limits, i.e. (min, max) tuple, of the image's dtype.
        
        Args:
            image(ndarray):
                Input image.
            clip_negative(bool):
                If True, clip the negative range (i.e. return 0 for min intensity)
                even if the image dtype allows negative values.
        Returns:
            (imin, imax : tuple)
                Lower and upper intensity limits.
        """
        if clip_negative is None:
            clip_negative = True
        imin, imax = dtype_range[self.dtype.type]
        if clip_negative:
            imin = 0
        return imin, imax
        
    def asfloat(self, normalise=False, clip_negative=False):
        """Return the image converted to floating point type.
        
        If currently an int type then floats will be automatically normalised.
        If currently a float type then will normalise if normalise.
        If currently an unsigned int type then image will be in range 0,1        
        Keyword Arguments:
            normalise(bool):
                normalise the image to -1,1
            clip_negative(bool):
                clip negative intensity to 0
        """
        ret = ImageArray(convert(self, dtype=np.float64))
        if normalise:
            ret = ret.normalise()
        if clip_negative:
            ret = ret.clip_negative()
        return ret
    
    def clip_intensity(self):
        """prefer ImageArray.normalise 
        
        clip_intensity for back compatibility
        """
        ret = self.asfloat(normalise=True, clip_negative=True)
        return ret
    
    def convert_float(self, clip_negative=True):
        """back compatability. asfloat preferred"""
        self.asfloat(normalise=False, clip_negative=clip_negative)
              
    def clip_negative(self):
        """Clip negative pixels to 0.
        
        Most useful for float where pixels above 1 are reduced to 1.0 and -ve pixels
        are changed to 0.
        """
        self[self<0] = 0
    
    def asint(self, dtype=np.uint16):
        """convert the image to unsigned integer format. 
        
        May raise warnings about loss of precision. Pass through to skiamge img_as_uint
        """
        if self.dtype.kind=='f' and (np.max(self)>1 or np.min(self)<-1):
            self=self.normalise()
        ret = convert(self, dtype)
        return ret
    
    def convert_int(self):
        """back compatability. asuint preferred"""
        self.asint()

    def save(self, filename, **kargs):
        """Saves the image into the file 'filename'. 
        
        Metadata will be preserved in .png format.
        fmt can be 'png' or 'npy' or 'both' which will save the file in that format.
        metadata is lost in .npy format but data is converted to integer format
        for png so that definition can be lost and negative values are clipped.

        Args:
            filename (string, bool or None): Filename to save data as, if this is
                None then the current filename for the object is used
                If this is not set, then then a file dialog is used. If 
                filename is False then a file dialog is forced.
            
        """
        fmt=kargs.pop("fmt","png")
        self.filename=getattr(self,"filename","")
        if fmt not in ['png','npy','both']:
            raise ValueError('fmt must be "png" or "npy" or "both"')
        if filename is None:
            filename = self.filename
        if filename in (None, '') or (isinstance(filename, bool) and not filename):
            # now go and ask for one
            filename = self.__file_dialog('w')
        if fmt in ['png', 'both']:
            pngname = os.path.splitext(filename)[0] + '.png'
            meta=PngImagePlugin.PngInfo()
            info=self.metadata.export_all()
            info=[i.split('=') for i in info]
            for k,v in info:        
                meta.add_text(k,v)
            s=self.asint(dtype=np.uint16)
            im=Image.fromarray(s.astype('uint32'),mode='I')
            im.save(pngname,pnginfo=meta)
        if fmt in ['npy', 'both']:
            npyname = os.path.splitext(filename)[0] + '.npy'
            np.save(npyname, self)
    
    def __file_dialog(self, mode):
        """Creates a file dialog box for loading or saving ~b ImageFile objects.

        Args:
            mode(string): The mode of the file operation  'r' or 'w'

        Returns:
            A filename to be used for the file operation.
        """
        patterns=(('png', '*.png'))

        if self.filename is not None:
            filename = os.path.basename(self.filename)
            dirname = os.path.dirname(self.filename)
        else:
            filename = ""
            dirname = ""
        if "r" in mode:
            mode = "file"
        elif "w" in mode:
            mode = "save"
        else:
            mode = "directory"
        dlg = get_filedialog(what=mode, initialdir=dirname, initialfile=filename, filetypes=patterns)
        if len(dlg) != 0:
            self.filename = dlg
            return self.filename
        else:
            return None


            
class ImageFile(metadataObject):
    
    """An Image file type that is analagous to DataFile.
    
    This contains metadata and an image attribute which
    is an ImageArray type which subclasses numpy ndarray and adds lots of extra
    image specific processing functions. 
    
    The ImageFile owned attribute is image. All other calls including metadata
    are passed through to ImageArray (so no need to inherit from metadataObject).
    
    Almost all calls to ImageFile are passed through to the underlying ImageArray
    logic and ImageArray can be used as a standalone class.
    However because ImageArray subclasses an ndarray it is not possible to enter
    it in place. All attributes return an array instance which needs to be reassigned.
    ImageFile owns image and so can change in place.
    The penalty is that numpy ufuncs don't return ImageFile type
    
    so can do:
    imfile.asfloat() #imagefile.image is updated to float type
    however need to do:
    imfile.image = np.abs(imfile.image)
    
    whereas for imarray:
    need to do:
    imarray = imagearray.asfloat()
    but:
    np.abs(imarray) #returns ImageArray type
    """
    
    _image = None
    
    fmts=["png","npy","tiff"]
    
    def __init__(self, *args, **kargs):
        """Mostly a pass through to ImageArray constructor.
        
        Local attribute is image. All other attributes and calls are passed
        through to image attribute.
        """
        super(ImageFile,self).__init__(*args,**kargs)
        if len(args)==0:
            self._image=ImageArray()
        elif len(args)>0 and isinstance(args[0],string_types):
            self._image = ImageArray(*args, **kargs)
        elif len(args)>0 and isinstance(args[0],ImageFile): # Fixing type
            self._image=args[0].image
        elif len(args)>0 and isinstance(args[0],np.ndarray): # Fixing type
            self._image=ImageArray(*args,**kargs)


    #####################################################################################################################################
    ############################# Properties #### #######################################################################################                        
    
    @property
    def clone(self):
        """Make a copy of this ImageFile."""
        new=self.__class__(self.image.clone)
        for attr in self.__dict__:
            if callable(getattr(self,attr)) or attr in ["image","metadata"]:
                continue
            try:
                setattr(new,attr,deepcopy(getattr(self,attr)))
            except NotImplementedError: # Deepcopying failed, so just copy a reference instead
                setattr(new,attr,getattr(self,attr))
        return new

    @property
    def data(self):
        """alias for image[:]. Equivalence to Stoner.data behaviour"""
        return self.image
    
    @data.setter
    def data(self,value):
        """Simple minded pass through."""
        self.image=value
    
    @property
    def CW(self):
        """Rotate clockwise by 90 deg."""
        ret=self.clone
        ret.image=ret.image.CW
        return ret
    
    @property
    def CCW(self):
        """Rotate counter-clockwise by 90 deg."""
        ret=self.clone
        ret.image=ret.image.CCW
        return ret

    @property
    def image(self):
        """Access the image data."""
        return self._image
    
    @image.setter
    def image(self, v):
        """Ensure stored image is always an ImageArray."""
        filename=self.filename
        self._image = ImageArray(v)
        self.filename=filename
        
    @property
    def filename(self):
        """Pass through to image attribute."""
        if self._image is not None:
            return self.image.filename
        else:
            return ""
    
    @filename.setter
    def filename(self,value):
        """Pass through to image attribute."""
        if self._image is None:
            self._image=ImageArray()
        self.image.filename=value
        
    @property
    def mask(self):
        """Get the mask of the underlying IamgeArray."""
        return MaskProxy(self)
    
    @mask.setter
    def mask(self,value):
        """Set the underlying ImageArray's mask."""
        if isinstance(value,ImageFile):
            value=value.image
        if isinstance(value,MaskProxy):
            value=value._mask
        self.image.mask=value
        
    @property
    def metadata(self):
        """Intercept metadata and redirect to image.metadata."""
        return self.image.metadata
    
    @metadata.setter
    def metadata(self,value):
        self.image.metadata=value

    #####################################################################################################################################
    ############################# Special methods #######################################################################################            

    def __dir__(self):
        """Implement code for dir()"""
        proxy=set(dir(self.image))
        parent=set(dir(super(ImageFile,self)))
        mine=set(dir(ImageFile))
        return sorted(list(proxy|parent|mine))

    
    def __getitem__(self, n):
        """A pass through to ImageArray."""
        try:
            return self.image.__getitem__(n)
        except KeyError:
            if n not in self.metadata and n in self._image.metadata:
                self.metadata[n]=self._image.metadata[n]
            return self.metadata.__getitem__(n)
   
    def __setitem__(self, n, v):
        """A Pass through to ImageArray."""
        if isinstance(n,string_types):
            self.metadata.__setitem__(n,v)
        else:
            self.image.__setitem__(n,v)
    
    def __delitem__(self, n):
        """A Pass through to ImageArray."""
        try:
            self.image.__delitem__(n)
        except KeyError:
            self.metadata.__delitem__(n)
    
    def __getattr__(self, n):
        """"Handles attriobute access."""
        try:
            ret=super(ImageFile,self).__getattr__(n)
        except AttributeError:
            ret = getattr(self.image, n)
            if callable(ret): #we have a method
                ret = self._func_generator(ret) #modiy so that we can change image in place
        return ret

    def __setattr__(self, n, v):
        """Handles setting attributes."""
        if not hasattr(self,n):
            setattr(self._image,n,v)
        else:
            super(ImageFile,self).__setattr__(n, v)

    def __add__(self,other):
        """Implement the subtract operator"""
        result=self.clone
        result=self.__add_core__(result,other)
        return result

    def __iadd__(self,other):
        """Implement the inplace subtract operator"""
        result=self
        result=self.__add_core__(result,other)
        return result
    
    def __add_core__(self,result,other):
        """Actually do result=result-other."""
        if isinstance(other,result.__class__) and result.shape==other.shape:
            result.image+=other.image
        elif isinstance(other,np.ndarray) and other.shape==result.shape:
            result.image+=other            
        elif isinstance(other,(int,float)):
            result.image+=other
        else:
            return NotImplemented
        return result
    
    def __truediv__(self,other):
        """Implement the divide operator"""
        result=self.clone
        result=self.__div_core__(result,other)
        return result

    def __itruediv__(self,other):
        """Implement the inplace divide operator"""
        result=self
        result=self.__div_core__(result,other)
        return result

    def __div_core__(self,result,other):
        """Actually do result=result/other."""
        #Cheat and pass through to ImageArray
        
        if isinstance(other,ImageFile):
            other=other.image
        
        result.image=result.image/other
        return result

    def __sub__(self,other):
        """Implement the subtract operator"""
        result=self.clone
        result=self.__sub_core__(result,other)
        return result

    def __isub__(self,other):
        """Implement the inplace subtract operator"""
        result=self
        result=self.__sub_core__(result,other)
        return result
    
    def __sub_core__(self,result,other):
        """Actually do result=result-other."""
        if isinstance(other,result.__class__) and result.shape==other.shape:
            result.image-=other.image
        elif isinstance(other,np.ndarray) and other.shape==result.shape:
            result.image-=other            
        elif isinstance(other,(int,float)):
            result.image-=other
        else:
            return NotImplemented
        return result
            
    def __neg__(self):
        """Intelliegent negate function that handles unsigned integers."""
        ret=self.clone
        if self._image.dtype.kind=="u":
            l,h=dtype_range[self._image.dtype]
            ret.image=h-self.image
        else:
            ret.image=-self.image
        return ret
    
    def __invert__(self):
        """Equivalent to clockwise rotation"""
        return self.CW

    def __floordiv__(self,other):
        """Calculate and XMCD ratio on the images."""
        if not isinstance(other,ImageFile):
            return NotImplemented
        ret=self.clone
        ret.image=(self.image-other.image)/(self.image+other.image)
        return ret
    
    #####################################################################################################################################
    ############################# Private methods #######################################################################################            
        
    def _func_generator(self, workingfunc):
        """ImageFile generator. 
        
        If function called returns ImageArray type 
        of the same shape as our current image then
        use that to update image attribute, otherwise return given output.
        """
        @wraps(workingfunc)
        def gen_func(*args, **kargs):

            if len(args)>0:
                args=list(args)
                for ix,a in enumerate(args):
                    if isinstance(a,ImageFile):
                        args[ix]=a.image

            force=kargs.pop("_",False)
            r = workingfunc(*args, **kargs)
            if isinstance(r,ImageArray) and (force or r.shape==self.image.shape):
                #Enure that we've captured any metadata added inside the working function
                self.metadata.update(r.metadata)
                #Now swap the iamge in, but keep the metadata
                r.metadata=self.metadata
                filename=self.filename
                self.image = r
                self.filename=filename
                
                return self
            else:
                return r
        return fix_signature(gen_func,workingfunc)
                
    def _repr_png_(self):
        """Provide a display function for iPython/Jupyter."""
        fig=self.image.imshow()
        plt.title(self.filename)
        data=StreamIO()
        fig.savefig(data,format="png")
        plt.close(fig)
        data.seek(0)
        ret=data.read()
        data.close()
        return ret
        
    #####################################################################################################################################
    ############################## Public methods #######################################################################################            

    def save(self, filename=None, **kargs):
        """Saves the image into the file 'filename'. 
        
        Metadata will be preserved in .png format.
        fmt can be 'png' or 'npy' or 'both' which will save the file in that format.
        metadata is lost in .npy format but data is converted to integer format
        for png so that definition can be lost and negative values are clipped.

        Args:
            filename (string, bool or None): Filename to save data as, if this is
                None then the current filename for the object is used
                If this is not set, then then a file dialog is used. If 
                filename is False then a file dialog is forced.
            
        """
        #Standard filename block
        if filename is None:
            filename = self.filename
        if filename is None or (isinstance(filename, bool) and not filename):
            # now go and ask for one
            filename = self.__file_dialog('w')
            
        
        def_fmt=os.path.splitext(filename)[1][1:] # Get a default format from the filename
        if def_fmt not in self.fmts: # Default to png if nothing else
            def_fmt="png"
        fmt=kargs.pop("fmt",[def_fmt])
        
        if not isinstance(fmt,list):
            fmt=[fmt]
        if set(fmt)&set(self.fmts)==set([]):
            raise ValueError('fmt must be {}'.format(",".join(self.fmts)))
        self.filename=filename
        for fm in fmt:
            saver=getattr(self,"save_{}".format(fm),"save_png")
            saver(filename)
        
    def save_png(self,filename):
        """Save the ImageFile with metadata in a png file."""
        pngname = os.path.splitext(filename)[0] + '.png'
        meta=PngImagePlugin.PngInfo()
        info=self.metadata.export_all()
        info=[(i.split('=')[0],"=".join(i.split('=')[1:])) for i in info]
        for k,v in info:        
            meta.add_text(k,v)
        s=(self.image-self.image.min())*256/(self.image.max()-self.image.min())
        im=Image.fromarray(s.astype('uint8'),mode='L')
        im.save(pngname,pnginfo=meta)

    def save_npy(self,filename):
        """Save the ImageFile as a numpy array."""
        npyname = os.path.splitext(filename)[0] + '.npy'
        np.save(npyname, self)

    def save_tiff(self,filename):
        """Save the ImageFile as a tiff image with metadata."""
        from PIL.TiffImagePlugin import ImageFileDirectory_v2
        import json
        im=Image.fromarray(self.image.asint( np.uint32),mode="I")
        ifd = ImageFileDirectory_v2()
        ifd[270]=json.dumps(self.metadata.export_all())
        tiffname = os.path.splitext(filename)[0] + '.tiff'
        im.save(tiffname,tiffinfo=ifd)
            
class DrawProxy(object):
    """Provides a wrapper around scikit-image.draw to allow easy drawing of objects onto images."""
    
    def __init__(self,*args,**kargs):
        """Grab the parent image from the constructor."""
        self.img = args[0]
        
    def __getattr__(self,name):
        """Retiurn a callable function that will carry out the draw operation requested."""
        func=getattr(draw,name)
        @wraps(func)
        def _proxy(*args,**kargs):
            value=kargs.pop("value",np.ones(1,dtype=self.img.dtype)[0])
            coords=func(*args,**kargs)
            self.img[coords]=value
            return self.img
        
        return fix_signature(_proxy,func)
    
    def __dir__(self):
        """Pass through to the dir of skimage.draw."""
        return draw.__dir__()
    
    def annulus(self,r,c,radius1,radius2,shape=None,value=1.0):
        """Use a combination of two circles to draw and annulus.
        
        Args:
            r,c (float): Centre co-ordinates
            radius1,radius2 (float): Inner and outer radius.
            
        Keyword Arguments:
            shape (2-tuple, None): Confine the co-ordinates to staywith shape
            value (float): value to draw with
        Returns:
            A copy of the image with the annulus drawn on it.
        
        Notes:
            If radius2<radius1 then the sense of the whole shape is inverted
            so that the annulus is left clear and the filed is filled. 
        """
        if shape is None:
            shape=self.img.shape
        invert=radius2<radius1
        if invert:
            buf=np.ones(shape)
            fill=0.0
            bg=1.0
        else:
            buf=np.zeros(shape)
            fill=1.0
            bg=2.0
        radius1,radius2=min(radius1,radius2),max(radius1,radius2)
        rr,cc=draw.circle(r,c,radius2,shape=shape)
        buf[rr,cc]=fill
        rr,cc=draw.circle(r,c,radius1,shape=shape)
        buf[rr,cc]=bg
        self.img=np.where(buf==1,value,self.img)
        return self.img
    
    def rectangle(self,r,c,w,h,angle=0.0,shape=None,value=1.0):
        """Draw a rectangle on an image.
        
        Args:
            r,c (float): Centre co-ordinates
            w,h (float): Lengths of the two sides of the rectangle
            
        Keyword Arguments:
            angle (float): Angle to rotate the rectangle about
            shape (2-tuple or None): Confine the co-ordinates to this shape.
            value (float): The value to draw with.
            
        Returns:
            A copy of the current image with a rectangle drawn on.
        """
        if shape is None:
            shape=self.img.shape

        x1=r-h/2
        x2=r+h/2
        y1=c-w/2
        y2=c+w/2
        co_ords=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
        if angle!=0:
            centre=np.array([r,c])
            c,s,m=np.cos,np.sin,np.matmul
            r=np.array([[c(angle),-s(angle)],[s(angle),c(angle)]])
            co_ords=np.array([centre+m(r,xy-centre) for xy in co_ords])
        rr,cc=draw.polygon(co_ords[:,0],co_ords[:,1],shape=shape)
        self.img[rr,cc]=value
        return self.img

    def square(self,r,c,w,angle=0.0,shape=None,value=1.0):
        """Draw a square on an image.
        
        Args:
            r,c (float): Centre co-ordinates
            w (float): Length of the side of the square
            
        Keyword Arguments:
            angle (float): Angle to rotate the rectangle about
            shape (2-tuple or None): Confine the co-ordinates to this shape.
            value (float): The value to draw with.
            
        Returns:
            A copy of the current image with a rectangle drawn on.
        """
        return self.rectangle(r,c,w,w,angle=angle,shape=shape,value=value)


class MaskProxy(object):
    
    @property
    def _IA(self):
        return self._IF.image
    
    @property
    def _mask(self):
        self._IA.mask=np.ma.getmaskarray(self._IA)
        return self._IA.mask   
    
    @property
    def draw(self):
        return DrawProxy(self._mask)
    
    def __init__(self,*args,**kargs):
        """Keep track of the underlying objects."""
        self._IF=args[0]
        
    def __getitem__(self,index):
        """Proxy through to mask index."""
        return self._mask.__getitem__(index)
    
    def __setitem__(self,index,value):
        """Proxy through to underlying mask."""
        self._IA.mask.__setitem__(index,value)
        
    def __delitem__(self,index):
        """Proxy through to underyling mask."""
        self._IA.mask.__delitem__(index)
        
    def __getattr__(self,name):
        """Checks name against self._IA._funcs and constructs a method to edit the mask as an image."""
        if hasattr(self._IA.mask,name):
            return getattr(self._IA.mask,name)
        if not ".*__{}$".format(name) in self._IA._funcs:
            raise AttributeError("{} not a callable mask method.".format(name))
        func=self._IA._funcs[".*__{}$".format(name)]
        @wraps(func)
        def _proxy_call(*args,**kargs):
            r=func(self._mask.astype(int),*args,**kargs)
            if isinstance(r,np.ndarray) and r.shape==self._IA.shape:
                self._IA.mask=r
            return r
        _proxy_call.__doc__=func.__doc__
        _proxy_call.__name__=func.__name__
        return fix_signature(_proxy_call,func)
    
    def __rep__(self):
        return repr(self._mask)
    
    def __str__(self):
        return repr(self._mask)
    
    def __neg__(self):
        return -self._mask
    
    def _repr_png_(self):
        """Provide a display function for iPython/Jupyter."""
        fig=self._IA._funcs[".*imshow"](self._mask.astype(int))
        data=StreamIO()
        fig.savefig(data,format="png")
        plt.close(fig)
        data.seek(0)
        ret=data.read()
        data.close()
        return ret
    
    def clear(self):
        """Clear a mask."""
        self._IA.mask=np.zeros_like(self._IA)
        
    def invert(self):
        """Invert the mask."""
        self._IA.mask=-self._IA.mask
        
        
        
        
        