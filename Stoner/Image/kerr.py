# -*- coding: utf-8 -*-
"""Kerr Image Processing Module.

Created on Fri Apr 21 17:29:08 2017
Derivatives of ImageArray and ImageStack specific to processing Kerr images.

@author: phyrct
"""
from Stoner import Data
from Stoner.Core import typeHintedDict
from Stoner.Image import ImageArray, ImageStack
import numpy as np
import subprocess,tempfile,os
from skimage import exposure,io,transform

GRAY_RANGE=(0,65535)  #2^16
IM_SIZE=(512,672) #Standard Kerr image size
AN_IM_SIZE=(554,672) #Kerr image with annotation not cropped


class KerrArray(ImageArray):
    
    """A subclass for Kerr microscopy specific image functions."""
    
    #useful_keys are metadata keys that we'd usually like to keep from a 
    #standard kerr output.
    _useful_keys = ['X-B-2d','field: units','MicronsPerPixel','Comment:',
        'Contrast Shift','HorizontalFieldOfView','Images to Average',
        'Lens','Magnification','Substraction Std']
    _test_keys = ['X-B-2d','field: units'] #minimum keys in data to assert that it is a standard file output
               
    def __init__(self,*args,**kargs):
        """Constructor for KerrArray which subclasses ImageArray. 
        
        Extra keyword arguments accepted are given below.
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
        kerrargs={}
        for k, v in kerrdefaults:
            kerrargs[k] = kargs.pop(k, v) 
        super(KerrArray,self).__init__(*args,**kargs)
        if kerrargs['reduce_metadata']:
            self.reduce_metadata()
        if kerrargs['ocr_metadata']:
            self.ocr_metadata(field_only=kerrargs['field_only']) 
        if kerrargs['asfloat']:
            self.asfloat()
        if kerrargs['crop_text']:
            self.crop_text()
            
    @property
    def tesseractable(self):
        """Do a test call to tesseract to see if it is there and cache the result."""
        try:
            self.__tesseractable=getattr(self,"_tesseractable")
        except AttributeError:
            self._tesseractable=subprocess.call(["tesseract","-v"])==0
        except Exception:
            self._tesseractable=False
        return self._tesseractable
    
    def crop_text(self, copy=False):
        """Crop the bottom text area from a standard Kermit image

        KeywordArguments:
            copy(bool):
                Whether to return a copy of the data or the original data

        Returns:
        (ImageArray):
            cropped image
        """
        if self.shape!=AN_IM_SIZE and self.shape!=IM_SIZE:
            raise ValueError('Need a full sized Kerr image to crop') #check it's a normal image
        return self.crop(None, None, None, IM_SIZE[0],copy=copy)

    def reduce_metadata(self):
        """Reduce the metadata down to a few useful pieces and do a bit of processing.
        
        Returns:
            (:py:class:`typeHintedDict`): the new metadata 
        """
        newmet={}
        if not all([k in self.keys() for k in KerrArray._test_keys]):
            return self.metadata #we've not got a standard Labview output, not safe to reduce
        for key in KerrArray._useful_keys:
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
        
        if key is given specific hints may be applied
        """
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
            except Exception:
                pass #leave it as string
        #print '{} after processsing: \'{}\''.format(key,data)

        return text

    def _tesseract_image(self, im, key):
        """ocr image with tesseract tool.
        
        im is the cropped image containing just a bit of text
        key is the metadata key we're trying to find, it may give a
        hint for parsing the text generated.
        """
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
    
    
class KerrStack(ImageStack):
    
    """:py:class:`Stoner.Image.stack.KerrStack is similar to ImageStack but adds some functionality particular to Kerr images.
    
    Attributes:
        fields(list):
            list of applied fields in stack. This is the most important metadata
            for things like hysteresis.    
    """
        
    def __init__(self, *args, **kargs):
        """Constructor."""
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
        """Convert an image of index values into an image of field values"""
        fieldvals=np.take(self.fields, index_map)
        return ImageArray(fieldvals)
    
    def reverse(self):
        """Reverse the image order"""
        self.imarray = self.imarray[::-1,:,:]
        self.fields = self.fields[::-1]
                              
    def denoise_thresh(self, denoise_weight=0.1, thresh=0.5, invert=False):
        """apply denoise then threshold images.
        
        Return a new MaskStack.
        True for values greater than thresh, False otherwise
        else return True for values between thresh and 1
        """
        masks=self.clone
        masks.apply_all('denoise', weight=0.1)
        masks.apply_all('threshold_minmax', threshmin=thresh, 
                        threshmax=np.max(masks.imarray))        
        masks=MaskStack(masks)
        if invert:
            masks.imarray = np.invert(masks.imarray)
        return masks
    
    def find_threshold(self, testim=None, mask=None):
        """Try to find the threshold value at which the image switches. 
        
        Takes it as the median value of the testim. Masks values
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
        """Produce a mask of areas of the image that are changing little over the stack. 
        
        comparison is an optional tuple that gives the index of two images
        to compare, otherwise first and last used. tolerance is the difference
        tolerance
        """
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
    
    def HcMap(self, threshold=0.5, correct_drift=False, baseimage=0, quiet=True, 
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
        if not quiet: print('thresholding done')  
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
    
    """Similar to ImageStack but made for stacks of boolean or binary images"""
    
    def __init__(self, *args, **kargs):
        """Constructor."""
        super(MaskStack,self).__init__(*args, **kargs)
        self.imarray=self.imarray.astype(bool)
    
    def switch_index(self, saturation_end=True, saturation_value=True):
        """Given a stack of boolean masks representing a hystersis loop find the stack index of the saturation field for each pixel.
        
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
            condition=np.logical_and(not ms[m], ms[m+1] )
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