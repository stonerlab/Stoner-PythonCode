
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
import tempfile
import warnings
import subprocess #calls to command line
from copy import deepcopy
from skimage import color,exposure,feature,io,measure,\
                    filters,graph,util,restoration,morphology,\
                    segmentation,transform,viewer
from skimage import img_as_float,convert,img_as_uint
from PIL import Image
from PIL import PngImagePlugin #for saving metadata
from Stoner.Core import typeHintedDict,metadataObject
from Stoner import Data
from Stoner.compat import * # Some things to help with Python2 and Python3 compatibility



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
    """:py:class:`Stoner.Image.core.ImageArray` is a numpy array like class
    with a metadata parameter and pass through to skimage methods. 
    
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
    _extra_attributes_default = {'metadata': TypeHintedDict({}),
                                 'filename': ''}

    #now initialise class

    def __new__(cls, *args, **kargs):
        """
        Construct an ImageArray object. 
        We're using __new__ rather than __init__ to imitate a numpy array as 
        close as possible.
        """
        
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
            image.metadata["Loaded from"]=filename
            return image
        
        if len(args) not in [0,1]:
            raise ValueError('ImageArray expects 0 or 1 arguments, {} given'.format(len(args)))
        array_arg_keys = ['dtype','copy','order','subok','ndmin'] #kwargs for array setup
        array_args = {k:kargs.pop(k) for k in array_args_keys if k in kargs.keys()}
        user_metadata = kargs.pop('metadata',{})
        asfloat = kargs.pop('asfloat', False) or kargs.pop('convert_float',False) #convert_float for back compatability
        if len(args)==0:
            ret = np.empty((0,0), dtype=float).view(cls)
            ret.metadata.update(metadata)
            return ret       
        arg = args[0]
        loadfromfile=False
        if isinstance(arg, np.ndarray):
            # numpy array or ImageArray
            ret = arg.view(ImageArray)
        elif isinstance(arg, bool) and not arg:
            patterns=(('png', '*.png'), ('npy','*.npy'))
            arg = get_filedialog(what='r',filetypes=patterns)
            if len(arg)==0:
                raise ValueError('No file given')
            else:
                loadfromfile=True
        elif isinstance(arg, string_types) or loadfromfile:
            # Filename- load datafile
            ret = _load(cls, arg, **array_args)
            if asfloat and ret.dtype.kind!='f': #convert to float type in place
                dl = dtype_range[ret.dtype.type]
                ret = np.array(ret, dtype=np.float64).view(cls)
                ret = ret / float(dl[1])
        elif isinstance(arg, ImageFile):
            #extract the image
            ret = arg.image
        else:
            try:  #try converting to a numpy array (eg a list type)
                ret = np.asarray(arg, **array_args).view(cls)
            except ValueError: #ok couldn't load from iterable, we're done
                raise SyntaxError("No constructor for {}".format(type(arg)))
        #all constructors call array_finalise so metadata is now initialised
        if 'Loaded from:' not in ret.metadata.keys():
            ret.metadata['Loaded from']=''
        ret.filename = ret.metadata['Loaded from']
        ret.metadata.update(user_metadata)        
        return ret

    def __array_finalize__(self, obj):
        """__array_finalize__ and __array_wrap__ are necessary functions when
        subclassing numpy.ndarray to fix some behaviours. See
        http://docs.scipy.org/doc/numpy-1.10.1/user/basics.subclassing.html for
        more info and examples
        Defaults below are only set when constructing an array using view
        eg np.arange(10).view(ImageArray). Otherwise filename and metadata
        attributes are just copied over (plus any other attributes set in 
        _extra_attributes).
        """
        self._extra_attributes = getattr(obj, '_extra_attributes', 
                                    ImageArray._extra_attributes_default)
        for k,v in _extra_attributes:
            if hasattr(obj, k):
                setattr(self, k, obj.k)
            else:
                setattr(self, k, v)

    def __array_wrap__(self, out_arr, context=None):
        """see __array_finalize__ for info"""
        ret=np.ndarray.__array_wrap__(self, out_arr, context)
        return ret

#==============================================================
# Propety Accessor Functions
#==============================================================r
    @property
    def clone(self):
        """return a copy of the instance"""
        ret = ImageArray(np.copy(self))
        for k,v in self._extra_attributes:
            setattr(ret, k, deepcopy(v))
        return ret        

    @property
    def max_box(self):
        """return the coordinate extent (xmin,xmax,ymin,ymax)"""
        box=(0,self.shape[1],0,self.shape[0])
        return (0,self.shape[1],0,self.shape[0])

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
        """run when asking for an attribute that doesn't exist yet. It
        looks first in imagefuncs.py then in skimage functions for a match. If
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
            t = [t[0],t[1]]
            if t[0] in self._ski_funcs.keys():
                if t[1] in self._ski_funcs[t[0]][1]:
                    workingfunc=getattr(self._ski_funcs[t[0]][0],t[1])
                    ret=self._func_generator(workingfunc)
        else: #Ok maybe just a request for an skimage func, no module
            for key in self._ski_funcs.keys(): #now look in skimage funcs
                if name in self._ski_funcs[key][1]:
                    workingfunc=getattr(self._ski_funcs[key][0],name)
                    ret=self._func_generator(workingfunc)
                    break

        if ret is None:
            raise AttributeError('No attribute found of name {}'.format(name))
        return ret
    
    def __setattr__(self, name, value):
        super(ImageArray, self).__setattr__(name, value)
         #add attribute to those for copying in array_finalize. use value as
         #defualt.
        _extra_attributes.update({name:value})
        
    def __getitem__(self,index):
        """Patch indexing of strings to metadata."""
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
#==============================================================
#Now any other useful bits
#==============================================================

    
    def box(self, *args, **kargs):
        """alias for crop
        """
        return self.crop(*args, **kargs)
    
    def crop_image(self, *args, **kargs):
        """back compatability"""
        return self.crop(*args, **kargs)

    def crop(self, *args, **kargs):
        """Crop the image. This is essentially like taking a view onto the array
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
    
    def convert_int(self):
        """convert the image to uint16 (the format used by Evico)
        Returns:
            :py:class:`Stoner.Image.core.ImageArray`
        """
        return self.img_as_uint()
        
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
        self.image = ImageArray(img_as_float(self))
        if normalise:
            self.normalise()
        if clip_negative:
            self.clip_negative()
    
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
    
    def asuint(self):
        """convert the image to unsigned integer format. May raise warnings 
        about loss of precision. Pass through to skiamge img_as_uint
        """
        ret = ImageArray(img_as_uint(self))
        return ret
    
    def convert_int(self):
        """back compatability. asuint preferred"""
        self.asuint()

    def save(self, filename=None, fmt='png'):
        """Saves the image into the file 'filename'. Metadata will be preserved 
        in .png format.
        fmt can be 'png' or 'npy' or 'both' which will save the file in that format.
        metadata is lost in .npy format but data is converted to integer format
        for png so that definition can be lost and negative values are clipped.

        Args:
            filename (string, bool or None): Filename to save data as, if this is
                None then the current filename for the object is used
                If this is not set, then then a file dialog is used. If 
                filename is False then a file dialog is forced.
            
        """
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
            s=self.convert_int()
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
            A filename to be used for the file operation."""
        # Wildcard pattern to be used in file dialogs.

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

class KerrArray(ImageArray):
    """A mixin class to work with ImageFile for Kerr specific image functions.
    """
    GRAY_RANGE=(0,65535)  #2^16
    IM_SIZE=(512,672) #Standard Kerr image size
    AN_IM_SIZE=(554,672) #Kerr image with annotation not cropped
    #useful_keys are metadata keys that we'd usually like to keep from a 
    #standard kerr output.
    useful_keys = ['X-B-2d','field: units','MicronsPerPixel','Comment:',
        'Contrast Shift','HorizontalFieldOfView','Images to Average',
        'Lens','Magnification','Substraction Std']
    _test_keys = ['X-B-2d','field: units'] #minimum keys in data to assert that
                                           #it is a standard file output
               
    def __init__(self,*args,**kargs):
        """
        Keyword Arguments:
            reduce_metadata(bool):
                if True reduce the metadata to useful bits and do some processing on it          
            asfloat(bool)
                if True convert the image to float values between 0 and 1 (necessary 
                for some forms of processing)
            crop_text(bool):
                whether to crop the bottom text area from the image
            ocr_metadata(bool):
                whether to try to use optical character recognition to get the 
                metadata from the image (necessary for images taken pre 06/2016
                and so far field from hysteresis images)
            field_only(bool):
                if ocr_metadata is true, get field only (bit faster)
        """
        kerrdefaults = {'ocr_metadata':False,
                        'field_only':False,
                        'reduce_metadata':True,
                        'asfloat':True,
                        'crop_text':True}
        self.kerrargs={}
        for k, v in kerrdefaults:
            self.kerrargs[k] = kargs.pop(k, v) 
        super(KerrMixin,self).__init__(*args,**kargs)
        if self.kerrargs['reduce_metadata']:
            self.reduce_metadata()
        if self.kerrargs['ocr_metadata']:
            self.ocr_metadata(field_only=self.kerrargs['field_only']) 
        if self.kerrargs['asfloat']:
            self.asfloat()
        if self.kerrargs['crop_text']:
            self.crop_text()
    
        def __init__(self, image=[],
                     ocr_metadata=False, field_only=False,
                                 metadata=None, **kwargs):
        """Constructor for :py:class:`Stoner.Image.core.ImageArray`.
        Create a ImageArray instance with metadata attribute

        Args:
            image: string or numpy array initiator
                If a filename is given it will try to load the image from memory
                Otherwise it will call np.array(image) on the object so an array or
                list is suitable
        Keyword Arguments:
            reduce_metadata: bool
                if True reduce the metadata to useful bits and do some processing on it          
            convert_float: bool
                if True convert the image to float values between 0 and 1 (necessary 
                for some forms of processing)
            ocr_metadata: bool
                whether to try to use optical character recognition to get the 
                metadata from the image (necessary for images taken pre 06/2016)
            field_only: bool
                if ocr_metadata is true, get field only (bit faster)
            metadata: dict
                dictionary of extra metadata items you would like adding to your array
        """
        
        if kwargs.get("reduce_metadata",self._reduce_metadata):
            self.reduce_metadata()
        if metadata is not None and isinstance(metadata,(dict,typeHintedDict)):
            for key in metadata.keys():
                self.metadata[key] = metadata[key]
        if isinstance(image,string_types):
            self.metadata['filename']=image
            self.filename=image
        if ocr_metadata:
            self.ocr_metadata(field_only=field_only) #update metadat 
            
    @property
    def tesseractable(self):
        """Do a test call to tesseract to see if it is there and cache the result."""
        if hasattr(self,"_tesseractable"):
            return self._tesseractable
        try:
            ret=subprocess.call(["tesseract","-v"])
        except:
            ret = -1
        self._tesseractable=ret==0
        return ret==0
    
    def crop_text(self, copy=False):
        """Crop the bottom text area from a standard Kermit image

        KeywordArguments:
            copy(bool):
                Whether to return a copy of the data or the original data

        Returns:
        (ImageArray):
            cropped image
        """

        assert self.shape==AN_IM_SIZE or self.shape==IM_SIZE, \
                'Need a full sized Kerr image to crop' #check it's a normal image
        crop=(0,IM_SIZE[1],0,IM_SIZE[0])
        return self.crop_image(box=crop, copy=copy)

    def reduce_metadata(self):
        """Reduce the metadata down to a few useful pieces and do a bit of 
        processing.
        
        Returns:
            (:py:class:`typeHintedDict`): the new metadata 
        """
        
        newmet={}
        useful_keys=['X-B-2d','field: units','MicronsPerPixel','Comment:',
                    'Contrast Shift','HorizontalFieldOfView','Images to Average',
                    'Lens','Magnification','Substraction Std']
        if not all([k in self.keys() for k in ['X-B-2d','field: units']]):
            return self.metadata #we've not got a standard Labview output, not safe to reduce
        for key in useful_keys:
            if key in self.keys():
                newmet[key]=self[key]
        newmet['field']=newmet.pop('X-B-2d') #rename
        if 'Substraction Std' in self.keys():
            newmet['subtraction']=newmet.pop('Substraction Std')
        if 'Averaging' in self.keys():
            if self['Averaging']: #averaging was on
                newmet['Averaging']=newmet.pop('Images to Average')
            else:
                newmet['Averaging']=1
                newmet.pop('Images to Average')
        self.metadata=typeHintedDict(newmet)
        return self.metadata
                
    def _parse_text(self, text, key=None):
        """Attempt to parse text which has been recognised from an image
        if key is given specific hints may be applied"""
        #print '{} before processsing: \'{}\''.format(key,data)

        #strip any internal white space
        text=[t.strip() for t in text.split()]
        text=''.join(text)

        #replace letters that look like numbers
        errors=[('s','5'),('S','5'),('O','0'),('f','/'),('::','x'),('Z','2'),
                         ('l','1'),('\xe2\x80\x997','7'),('?','7'),('I','1'),
                         ('].','1'),("'",'')]
        for item in errors:
            text=text.replace(item[0],item[1])

        #apply any key specific corrections
        if key in ['ocr_field','ocr_scalebar_length_microns']:
            try:
                text=float(text)
            except:
                pass #leave it as string
        #print '{} after processsing: \'{}\''.format(key,data)

        return text

    def _tesseract_image(self, im, key):
        """ocr image with tesseract tool.
        im is the cropped image containing just a bit of text
        key is the metadata key we're trying to find, it may give a
        hint for parsing the text generated."""

        #first set up temp files to work with
        tmpdir=tempfile.mkdtemp()
        textfile=os.path.join(tmpdir,'tmpfile.txt')
        stdoutfile=os.path.join(tmpdir,'logfile.txt')
        imagefile=os.path.join(tmpdir,'tmpim.tif')
        with open(textfile,'w') as tf:#open a text file to export metadata to temporarily
            pass

        #process image to make it easier to read
        i=1.0*im / np.max(im) #change to float and normalise
        i=exposure.rescale_intensity(i,in_range=(0.49,0.5)) #saturate black and white pixels
        i=exposure.rescale_intensity(i) #make sure they're black and white
        i=transform.rescale(i, 5.0) #rescale to get more pixels on text
        io.imsave(imagefile,(255.0*i).astype("uint8"),plugin='pil') #python imaging library will save according to file extension

        #call tesseract
        if self.tesseractable:
            with open(stdoutfile,"w") as stdout:
                subprocess.call(['tesseract', imagefile, textfile[:-4]],stdout=stdout,stderr=subprocess.STDOUT) #adds '.txt' extension itself
            os.unlink(stdoutfile)
        with open(textfile,'r') as tf:
            data=tf.readline()

        #delete the temp files
        os.remove(textfile)
        os.remove(imagefile)
        os.rmdir(tmpdir)

        #parse the reading
        if len(data)==0:
            print('No data read for {}'.format(key))
        data=self._parse_text(data, key=key)
        return data

    def _get_scalebar(self):
        """Get the length in pixels of the image scale bar"""
        box=(0,419,519,520) #row where scalebar exists
        im=self.crop_image(box=box, copy=True)
        im=im.astype(float)
        im=(im-im.min())/(im.max()-im.min())
        im=exposure.rescale_intensity(im,in_range=(0.49,0.5)) #saturate black and white pixels
        im=exposure.rescale_intensity(im) #make sure they're black and white
        im=np.diff(im[0]) #1d numpy array, differences
        lim=[np.where(im>0.9)[0][0],
             np.where(im<-0.9)[0][0]] #first occurance of both cases
        assert len(lim)==2, 'Couldn\'t find scalebar'
        return lim[1]-lim[0]

    def ocr_metadata(self, field_only=False):
        """Use image recognition to try to pull the metadata numbers off the image

        Requirements: This function uses tesseract to recognise the image, therefore
        tesseract file1 file2 must be valid on your command line.
        Install tesseract from
        https://sourceforge.net/projects/tesseract-ocr-alt/files/?source=navbar

        KeywordArguments:
            field_only(bool):
                only try to return a field value

        Returns:
            metadata: dict
                updated metadata dictionary
        """
        if self.shape!=AN_IM_SIZE:
            pass #can't do anything without an annotated image

        #now we have to crop the image to the various text areas and try tesseract
        elif field_only:
            fbox=(110,165,527,540) #(This is just the number area not the unit)
            im=self.crop_image(box=fbox,copy=True)
            field=self._tesseract_image(im,'ocr_field')
            self.metadata['ocr_field']=field
        else:
            text_areas={'ocr_field': (110,165,527,540),
                        'ocr_date': (542,605,512,527),
                        'ocr_time': (605,668,512,527),
                        'ocr_subtract': (237,260,527,540),
                        'ocr_average': (303,350,527,540)}
            try:
                sb_length=self._get_scalebar()
            except AssertionError:
                sb_length=None
            if sb_length is not None:
                text_areas.update({'ocr_scalebar_length_microns': (sb_length+10,sb_length+27,514,527),
                                   'ocr_lens': (sb_length+51,sb_length+97,514,527),
                                    'ocr_zoom': (sb_length+107,sb_length+149,514,527)})

            metadata={}   #now go through and process all keys
            for key in text_areas.keys():
                im=self.crop_image(box=text_areas[key], copy=True)
                metadata[key]=self._tesseract_image(im,key)
            metadata['ocr_scalebar_length_pixels']=sb_length
            if type(metadata['ocr_scalebar_length_microns'])==float:
                metadata['ocr_microns_per_pixel']=metadata['ocr_scalebar_length_microns']/sb_length
                metadata['ocr_pixels_per_micron']=1/metadata['ocr_microns_per_pixel']
                metadata['ocr_field_of_view_microns']=np.array(IM_SIZE)*metadata['ocr_microns_per_pixel']
            self.metadata.update(metadata)
        if 'ocr_field' in self.metadata.keys() and not isinstance(self.metadata['ocr_field'],(int,float)):
            self.metadata['ocr_field']=np.nan  #didn't read the field properly
        return self.metadata
            
class ImageFile(metadataObject):
    def __init__(self, *args, **kargs):
            #for image file
        self._public_attrs={
                "metadata": typeHintedDict,
                "filename": string_types,
        self.metadata["Stoner.class"] = self.__class__.__name__
        
        if len(kargs) > 0:  # set public attributes from keywords
                myattrs = self._public_attrs
                to_go=[]
                for k in kargs:
                    if k in myattrs:
                        if isinstance(kargs[k], myattrs[k]):
                            self.__setattr__(k, kargs[k])
                        else:
                            if isinstance(myattrs[k], tuple):
                                typ = "one of " + ",".join([str(type(t)) for t in myattrs[k]])
                            else:
                                typ = "a {}".format(type(myattrs[k]))
                            raise TypeError("{} should be {} not a {}".format(k, typ, type(kargs[k])))
                        to_go.append(k)
                for k in to_go:
                    del kargs[k]