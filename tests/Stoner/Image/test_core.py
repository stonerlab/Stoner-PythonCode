# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:09:04 2016

@author: phyrct
"""

from Stoner.Image import ImageArray, KerrArray, ImageFile
from Stoner.Core import typeHintedDict
import numpy as np
import unittest
import sys
from os import path
import os

import warnings

#data arrays for testing - some useful small images for tests

thisdir=path.dirname(__file__)

def share_memory(arr1, arr2):
    """Check if two numpy arrays share memory"""
    return arr1.base is arr2 or arr2.base is arr1

class ImageArrayTest(unittest.TestCase):

    def setUp(self):
        #random array shape 3,4
        self.arr = np.array([[ 0.41674764,  0.66100043,  0.91755303,  0.33796703],
                                  [ 0.06017535,  0.1440342 ,  0.34441777,  0.9915282 ],
                                  [ 0.2984083 ,  0.9167951 ,  0.73820304,  0.7655299 ]])
        self.imarr = ImageArray(np.copy(self.arr))
        self.imarrfile = ImageArray(os.path.join(thisdir, 'coretestdata/im1_annotated.png'))
    
    #####test loading with different datatypes  ####
        
    def test_load_from_array(self):
        #from array
        self.assertTrue(np.array_equal(self.imarr,self.arr))
        #int type
        imarr = ImageArray(arange(12).reshape(3,4))
        self.assertTrue(imarr.dtype==np.dtype('int32'))        
    
    def test_load_from_ImageArray(self):
        #from ImageArray
        t = ImageArray(self.imarr)
        self.assertFalse(share_memory(self.imarr, t), 'no overlap on creating ImageArray from ImageArray')
    
    def test_load_from_png(self):
        subpath = 'coretestdata/im1_annotated.png'
        fpath = path.join(thisdir, subpath)
        anim=ImageArray(fpath)
        self.assertTrue(anim.metadata['Loaded from'] == fpath)
        cwd = os.getcwd()
        os.chdir(thisdir)
        anim = ImageArray(subpath)
        #check full path is in loaded from metadata
        self.assertTrue(anim.metadata['Loaded from'] == fpath, 'Full path not in metadata')
        os.chdir(cwd)
   
    def test_load_from_ImageFile(self):
        #uses the ImageFile.im attribute to set up ImageArray. Memory overlaps
        imfi = ImageFile(self.arr)
        imarr = ImageArray(imfi)
        self.assertTrue(np.array_equal(imarr, imfi.image), 'Initialising from ImageFile failed')
        self.assertTrue(share_memory(imarr, imfi.image))
    
    def test_load_from_list(self):
        t=ImageArray([[1,3],[3,2],[4,3]])
        self.assertTrue(np.array_equal(t, np.array([[1,3],[3,2],[4,3]])), 'Initialising from list failed')
    
    def test_load_1d_data(self):
        t=ImageArray(np.arange(10)/10.0)
        self.assertTrue(len(t.shape)==2) #converts to 2d
    
    def test_load_no_args(self):
        #Should be a 2d empty array
        t=ImageArray()
        self.assertTrue(len(t.shape)==2)
        self.assertTrue(t.size==0)
    
    def test_load_bad_data(self):
        #dictionary
        self.assertRaises(ValueError, ImageArray.__init__, {'a':1})
        #3d numpy array
        self.assertRaises(ValueError, ImageArray.__init__, np.arange(27).reshape(3,3,3))
        #bad filename
        self.assertRaises(ValueError, ImageArray.__init__, 'sillyfile.xyz')
        
    def test_load_kwargs(self):
        #metadata keyword arg
        t=ImageArray(self.arr, metadata={'a':5, 'b':7})
        self.assertTrue('a' in t.metdata.keys() and 'b' in t.metadata.keys())
        self.assertTrue(t.metadata['a']==5)
        #asfloat
        t=ImageArray(np.arange(12).reshape(3,4), asfloat=True)
        self.assertTrue(t.dtype == np.float64, 'Initialising asfloat failed')
        self.assertTrue('Loaded from' in t.metadata.keys(), 'Loaded from should always be in metadata')
    
    #####test attributes ##
    
    def test_filename(self):
        self.assertTrue(self.imarr.filename == '')
        self.assertTrue(self.imarrfile.filename == os.path.join(thisdir, 'coretestdata/im1_annotated.png'))

    def test_clone(self):
        self.imarr['abc'] = 123 #add some metadata
        self.imarr['nested'] = [1,2,3] #add some nested metadata to check deepcopy
        c = self.imarr.clone
        self.assertTrue(isinstance(c,ImageArray), 'Clone not ImageArray')
        self.assertTrue(np.array_equal(c,self.imarr),'Clone not replicating elements')
        self.assertTrue(all([k in c.metadata.keys() for k in \
                    self.imarr.metadata.keys()]), 'Clone not replicating metadata')
        self.assertFalse(share_memory(c, self.imarr), 'memory overlap on clone') #formal check
        self.imarr['bcd'] = 234
        self.assertTrue('bcd' not in c.metadata.keys(), 'memory overlap for metadata on clone')
        self.imarr['nested'][0] = 2
        self.assertTrue(self.imarr['nested'][0] != c['nested'][0])
    
    def test_metadata(self):
        self.assertTrue(isinstance(self.imarr.metadata,typeHintedDict))
        self.imarr['testmeta']='abc'
        self.assertTrue(td1['testmeta']=='abc', 'Couldn\'t change metadata')
        del(td1['testmeta'])
        self.assertTrue('testmeta' not in td1.keys(),'Couldn\'t delete metadata')
        #bad data
        self.imarr.metadata=(1,2,3) #check it won't let you do this        
        
    ### test numpy like creation behaviour #
    def test_user_attributes(self):
        self.imarr.abc = 'new att'
        self.assertTrue(hasattr(self.imarr, 'abc'))
        t = ImageArray(self.imarr)
        self.assertTrue(hasattr(t, 'abc'), 'problem copying new attributes')
        t = self.imarr.vew(ImageArray) #check array_finalize copies attribute over
        self.assertTrue(hasattr(t, 'abc'))
        t = self.imarr * np.random.rand(self.imarr.shape)
        self.assertTrue(isinstance(t, ImageArray), 'problem with ufuncs')
        self.assertTrue(hasattr(t, 'abc'))

    #####  test functionality  ##
    def test_save(self):
        testfile=path.join(thisdir,'coretestdata','testsave.png')
        keys=self.anim.keys()
        self.anim.save(filename=testfile)
        load=ImageArray(testfile)
        del load["Loaded from"]
        self.assertTrue(all([k in keys for k in load.keys()]), 'problem saving metadata')
        os.remove(testfile) #tidy up

    def test_max_box(self):
        s=self.anim.shape
        self.assertTrue(self.anim.max_box==(0,s[1],0,s[0]))

    def test_data(self):
        self.assertTrue(np.array_equal(self.td1.data,self.td1[:]), 'self.data doesn\'t look like the data')
        t=self.td1.clone
        t.data[0,0]+=1
        self.assertTrue(np.array_equal(t.data,t), 'self.data did not change')

    def test_box(self):
        t=self.td1.clone
        b=t.box(1,2,1,3)
        self.assertTrue(b.shape==(2,1))
        b[0,0]+=1
        self.assertTrue(t[1,1]==(self.td1[1,1]+1),
                        'box does not have same memory space as original array')

    

    def test_crop_image(self):
        td1=self.td1.clone
        c=td1.crop_image(box=(1,3,1,4),copy=False)
        self.assertTrue(np.array_equal(c,self.td1[1:4,1:3]),'crop didn\'t work')
        self.assertTrue(np.may_share_memory(c,td1), 'crop copied image when it shouldn\'t')
        c=td1.crop_image(box=(1,3,1,4),copy=True)
        self.assertFalse(c.base is td1, 'crop didn\'t copy image')

    def test_other_funcs(self):
        td1=td1.level_image() #test kfuncs
        td1=td1.img_as_float() #test skimage
        td1=td1.gaussian(sigma=2)

#class KerrArrayTest(unittest.TestCase):
#    
#    def test_tesseract_ocr(self):
#        #this incidently tests get_metadata too
#        if not self.anim.tesseractable:
#            print("#"*80)
#            print("Skipping test that uses tesseract.")
#            return None
#        m=self.anim.metadata
#        self.assertTrue(all((m['ocr_average']=='on,8x',
#                            m['ocr_date']=='09/03/16',
#                            m['ocr_field'] == 148.63)), 'Misread metadata')
#        keys=('ocr_scalebar_length_pixels', 'ocr_field_of_view_microns',
#                          'filename', 'ocr_microns_per_pixel', 'ocr_pixels_per_micron')
#        self.assertTrue(all([k in m.keys() for k in keys]), 'some part of the metadata didn\'t load')
#        m_un=self.unanim.metadata
#        self.assertTrue('ocr_field' not in m_un.keys(), 'Unannotated image has wrong metadata')
        
        
        
if __name__=="__main__": # Run some tests manually to allow debugging
    #test=ImageArrayTest()
    #test.setUp()
    with warnings.catch_warnings():
#        warnings.simplefilter("ignore")
        test=ImageArrayTest("test_load")
        test.setUp()
        for attr in dir(test):
            if attr.startswith("test_"):
                print(attr)
                getattr(test,attr)()



