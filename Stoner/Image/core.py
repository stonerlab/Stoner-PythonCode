
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
                    segmentation,transform,viewer
from PIL import Image
from PIL import PngImagePlugin #for saving metadata
from Stoner.Core import typeHintedDict,metadataObject
from Stoner.Image.util import convert
from Stoner import Data
from Stoner.compat import string_types,get_filedialog # Some things to help with Python2 and Python3 compatibility
import inspect



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


    
class ImageArray(np.ndarray,metadataObject):
    
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
    _ski_funcs_proxy=None
    _kfuncs_proxy=None
    #extra attributes for class beyond standard numpy ones
    _extra_attributes_default = {'metadata': typeHintedDict({}),
                                 'filename': ''}

    #now initialise class

    def __new__(cls, *args, **kargs):
        """Construct an ImageArray object.
        
        We're using __new__ rather than __init__ to imitate a numpy array as 
        close as possible.
        """
        
        
        if len(args) not in [0,1]:
            raise ValueError('ImageArray expects 0 or 1 arguments, {} given'.format(len(args)))
            
        ### Deal with kwargs
        array_arg_keys = ['dtype','copy','order','subok','ndmin'] #kwargs for array setup
        array_args = {k:kargs.pop(k) for k in array_arg_keys if k in kargs.keys()}
        user_metadata = kargs.pop('metadata',{})
        asfloat = kargs.pop('asfloat', False) or kargs.pop('convert_float',False) #convert_float for back compatability
        _debug = kargs.pop('_debug', False)
                           
        ### 0 args initialisation
        if len(args)==0:
            ret = np.empty((0,0), dtype=float).view(cls)
            ret.metadata.update(user_metadata)
            return ret       
        
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
        ret._debug = _debug
        return ret

    def __array_finalize__(self, obj):
        """__array_finalize__ and __array_wrap__ are necessary functions when subclassing numpy.ndarray to fix some behaviours. 
        
        See http://docs.scipy.org/doc/numpy-1.10.1/user/basics.subclassing.html for
        more info and examples
        Defaults below are only set when constructing an array using view
        eg np.arange(10).view(ImageArray). Otherwise filename and metadata
        attributes are just copied over (plus any other attributes set in 
        _extra_attributes).
        """
        if hasattr(self, '_debug') and self._debug:
            print(type(self), type(obj))
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            print('caller name:', calframe[1][3])
        _extra_attributes = getattr(obj, '_extra_attributes', 
                                deepcopy(ImageArray._extra_attributes_default))
        setattr(self, '_extra_attributes', copy(_extra_attributes))
        for k,v in _extra_attributes.items():
            setattr(self, k, getattr(obj, k, v))       

    def __array_wrap__(self, out_arr, context=None):
        """Part of the numpy array machinery.
        
        see __array_finalize__ for info. This is for if ImageArray is called 
        via ufuncs. array_finalize is called after.
        """
        ret=np.ndarray.__array_wrap__(self, out_arr, context)
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
        pass #already sorted metadata and ndarray setup through __new__

    @classmethod
    def _load(cls,filename,**array_args):
        """Load an image from a file and return as a ImageArray."""
        if filename.endswith('.npy'):
            image = np.load(filename)
            image = np.array(image, **array_args).view(cls)
        else:        
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
        return image
        
#==============================================================
# Propety Accessor Functions
#==============================================================r
    @property
    def clone(self):
        """return a copy of the instance"""
        ret = ImageArray(np.copy(self))
        for k,v in self._extra_attributes.items():
            setattr(ret, k, deepcopy(v))
        return ret        

    @property
    def max_box(self):
        """return the coordinate extent (xmin,xmax,ymin,ymax)"""
        box=(0,self.shape[1],0,self.shape[0])
        return box

    @property
    def data(self):
        """alias for image[:]. Equivalence to Stoner.data behaviour"""
        return self[:]
    
    @property
    def _kfuncs(self):
        """Provide an attribtute that caches the imported ImageArray functions."""
        if self._kfuncs_proxy is None:
            from . import imagefuncs
            self._kfuncs_proxy=imagefuncs
        return self._kfuncs_proxy

    @property
    def _ski_funcs(self):
        """Provide an attribute that cahces the import sckit-image function names."""
        if self._ski_funcs_proxy is None:
            _ski_modules=[color,exposure,feature,io,measure,filters,filters.rank, graph,
                    util, restoration, morphology, segmentation, transform,
                    viewer]
            self._ski_funcs_proxy={}
            for mod in _ski_modules:
                self._ski_funcs_proxy[mod.__name__] = (mod, dir(mod))
        return self._ski_funcs_proxy
        


#==============================================================
#function generator
#==============================================================
    def __dir__(self):
        """Implement code for dir()"""
        kfuncs=set(dir(self._kfuncs))
        skimage=set()
        mods=list(self._ski_funcs.keys())
        mods.reverse()
        for k in mods:
            skimage|=set(self._ski_funcs[k][1])
        parent=set(dir(super(ImageArray,self)))
        mine=set(dir(ImageArray))
        return list(skimage|kfuncs|parent|mine)
    
    def __getattr__(self,name):
        """run when asking for an attribute that doesn't exist yet. 
        
        It looks first in imagefuncs.py then in skimage functions for a match. If
        it finds it it returns a copy of the function that automatically adds
        the image as the first argument.
        skimage functions can be called by module_func notation
        with the underscore sep eg im.exposure_rescale_intensity, or it can simply
        be the function with no module given in which case the entire directory
        is searched and the first hit is returned im.rescale_intensity.
        Note that numpy is already subclassed so numpy funcs are highest on the
        heirarchy, followed by imagefuncs, followed by skimage funcs
        Also note that the function named must take image array as the
        first argument


        An alternative nested attribute system could be something like
        http://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
        might be cool sometime.
        """
        ret=None
        #first check kermit funcs
        if name.startswith('_'):
            pass
        elif name in dir(self._kfuncs):
            workingfunc=getattr(self._kfuncs,name)
            ret=self._func_generator(workingfunc)
        elif '_' in name: #ok we might have a skimage module_function request
            t=name.split('_')
            t[1]='_'.join(t[1:]) #eg rescale_intensity needs stitching back together
            t = ['skimage.'+t[0],t[1]]
            if t[0] in self._ski_funcs.keys():
                if t[1] in self._ski_funcs[t[0]][1]:
                    workingfunc=getattr(self._ski_funcs[t[0]][0],t[1])
                    ret=self._func_generator(workingfunc)        
        if ret is None: #Ok maybe just a request for an skimage func, no module
            for key in self._ski_funcs.keys(): #now look in skimage funcs
                if name in self._ski_funcs[key][1]:
                    workingfunc=getattr(self._ski_funcs[key][0],name)
                    ret=self._func_generator(workingfunc)
                    break

        if ret is None:
            raise AttributeError('No attribute found of name {}'.format(name))
        return ret
    
    def _func_generator(self,workingfunc):
        """generate a function that adds self as the first argument"""

        def gen_func(*args, **kwargs):
            r=workingfunc(self.clone, *args, **kwargs) #send copy of self as the first arg
            if isinstance(r,Data):
                pass #Data return is ok
            elif isinstance(r,np.ndarray) and np.prod(r.shape)==np.max(r.shape): #1D Array
                r=Data(r)
                r.metadata=self.metadata.copy()
                r.column_headers[0]=workingfunc.__name__
            elif isinstance(r,np.ndarray) and not isinstance(r,ImageArray): #make sure we return a ImageArray
                r=r.view(type=ImageArray)
                r.metadata=self.metadata.copy()
            #NB we might not be returning an ndarray at all here !
            return r
        gen_func.__doc__=workingfunc.__doc__
        gen_func.__name__=workingfunc.__name__
        return gen_func 
    
    def __setattr__(self, name, value):
        """Set an attribute on the object."""
        super(ImageArray, self).__setattr__(name, value)
        #add attribute to those for copying in array_finalize. use value as
        #defualt.
        circ = ['_extra_attributes'] #circular references
        proxy = ['_kfuncs_proxy', '_ski_funcs_proxy'] #can be reloaded for cloned arrays
        if name in circ + proxy: 
            #Ignore these in clone
            pass
        else:
            self._extra_attributes.update({name:value})
        
    def __getitem__(self,index):
        """Patch indexing of strings to metadata."""
        if self._debug:
            curframe = inspect.currentframe()
            calframe = inspect.getouterframes(curframe, 2)
            print('caller name:', calframe[1][3])
        if isinstance(index,string_types):
            return self.metadata[index]
        else:
            return super(ImageArray,self).__getitem__(index)


    def __setitem__(self,index,value):
        """Patch string index through to metadata."""
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
    
    def normalise(self):
        """Normalise the image to -1,1.
        """
        if self.dtype.kind != 'f':
            ret  = self.asfloat(normalise=False) #bit dodgy here having normalise in asfloat
        else:
            ret = self
        norm=np.linalg.norm(ret)
        if norm==0: 
            raise RuntimeError('Attempting to normalise an array with only 0 values')
        ret = ret / norm
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


            
class ImageFile(object):
    
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
    
    def __init__(self, *args, **kargs):
        """Mostly a pass through to ImageArray constructor.
        
        Local attribute is image. All other attributes and calls are passed
        through to image attribute.
        """
        self.image = ImageArray(*args, **kargs)
    
    @property
    def image(self):
        """Access the image data."""
        return self._image
    
    @image.setter
    def image(self, v):
        """Ensure stored image is always an ImageArray."""
        self._image = ImageArray(v)
    
    def __getitem__(self, n):
        """A pass through to ImageArray."""
        return self.image.__getitem__(n)
   
    def __setitem__(self, n, v):
        """A Pass through to ImageArray."""
        self.image.__setitem__(n,v)
    
    def __delitem__(self, n):
        """A Pass through to ImageArray."""
        self.image.__delitem__(n)
    
    def __getattr__(self, n):
        """"Handles attriobute access."""
        ret = getattr(self.image, n)
        if hasattr(ret, '__call__'): #we have a method
            ret = self._func_generator(ret) #modiy so that we can change image in place
        return ret
        
    def _func_generator(self, workingfunc):
        """ImageFile generator. 
        
        If function called returns ImageArray type 
        of the same shape as our current image then
        use that to update image attribute, otherwise return given output.
        """
        def gen_func(*args, **kargs):
            r = workingfunc(*args, **kargs)
            if isinstance(r,ImageArray) and r.shape==self.image.shape:
                self.image = r
            else:
                return r
        
        gen_func.__doc__=workingfunc.__doc__
        gen_func.__name__=workingfunc.__name__
        return gen_func            
    
    def __setattr__(self, n, v):
        """Handles setting attributes."""
        if n in ['image', '_image']:
            super(ImageFile,self).__setattr__(n, v)
        else:
            setattr(self.image, n, v)
