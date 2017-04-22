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

def shares_memory(arr1, arr2):
    """Check if two numpy arrays share memory"""
    #ret = arr1.base is arr2 or arr2.base is arr1
    ret = np.may_share_memory(arr1, arr2)
    return ret

class ImageArrayTest(unittest.TestCase):

    def setUp(self):
        #random array shape 3,4
        self.arr = np.array([[ 0.41674764,  0.66100043,  0.91755303,  0.33796703],
                                  [ 0.06017535,  0.1440342 ,  0.34441777,  0.9915282 ],
                                  [ 0.2984083 ,  0.9167951 ,  0.73820304,  0.7655299 ]])
        self.imarr = ImageArray(np.copy(self.arr)) #ImageArray object
        self.imarrfile = ImageArray(os.path.join(thisdir, 'coretestdata/im1_annotated.png'))
                        #ImageArray object from file
    
    #####test loading with different datatypes  ####
        
    def test_load_from_array(self):
        #from array
        self.assertTrue(np.array_equal(self.imarr,self.arr))
        #int type
        imarr = ImageArray(np.arange(12,dtype="int32").reshape(3,4))
        self.assertTrue(imarr.dtype==np.dtype('int32'),"Failed to set correct dtype - actual dtype={}".format(imarr.dtype))        
    
    def test_load_from_ImageArray(self):
        #from ImageArray
        t = ImageArray(self.imarr)
        self.assertTrue(shares_memory(self.imarr, t), 'no overlap on creating ImageArray from ImageArray')
    
    def test_load_from_png(self):
        subpath = os.path.join("coretestdata","im1_annotated.png")
        fpath = os.path.join(thisdir, subpath)
        anim=ImageArray(fpath)
        self.assertTrue(os.path.normpath(anim.metadata['Loaded from']) == os.path.normpath(fpath))
        cwd = os.getcwd()
        os.chdir(thisdir)
        anim = ImageArray(subpath)
        #check full path is in loaded from metadata
        self.assertTrue(os.path.normpath(anim.metadata['Loaded from']) == os.path.normpath(fpath), 'Full path not in metadata: {}'.format(anim["Loaded from"]))
        os.chdir(cwd)
   
    def test_load_from_ImageFile(self):
        #uses the ImageFile.im attribute to set up ImageArray. Memory overlaps
        pass
#        imfi = ImageFile(self.arr)
#        imarr = ImageArray(imfi)
#        self.assertTrue(np.array_equal(imarr, imfi.image), 'Initialising from ImageFile failed')
#        self.assertTrue(shares_memory(imarr, imfi.image))
    
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
        def testload(arg):
            ImageArray(arg)
        #dictionary
        self.assertRaises(ValueError, testload, {'a':1})
        #3d numpy array
        self.assertRaises(ValueError, testload, np.arange(27).reshape(3,3,3))
        #bad filename
        self.assertRaises(ValueError, testload, 'sillyfile.xyz')
            
    def test_load_kwargs(self):
        #metadata keyword arg
        t=ImageArray(self.arr, metadata={'a':5, 'b':7})
        self.assertTrue('a' in t.metadata.keys() and 'b' in t.metadata.keys())
        self.assertTrue(t.metadata['a']==5)
        #asfloat
        t=ImageArray(np.arange(12).reshape(3,4), asfloat=True)
        self.assertTrue(t.dtype == np.float64, 'Initialising asfloat failed')
        self.assertTrue('Loaded from' in t.metadata.keys(), 'Loaded from should always be in metadata')
    
    #####test attributes ##
    
    def test_filename(self):
        im = ImageArray(np.linspace(0,1,12).reshape(3,4))
        fpath = os.path.join(thisdir, 'coretestdata/im1_annotated.png')
        self.assertTrue(os.path.normpath(self.imarrfile.filename)\
                        == os.path.normpath(fpath))
        im = ImageArray(np.linspace(0,1,12).reshape(3,4))
        im['Loaded from']
        im.filename
        self.assertTrue(im.filename == '', '{}, {}'.format(self.imarr.shape, im.filename))

    def test_clone(self):
        self.imarr['abc'] = 123 #add some metadata
        self.imarr['nested'] = [1,2,3] #add some nested metadata to check deepcopy
        self.imarr.userxyz = 123 #add a user attribute
        c = self.imarr.clone
        self.assertTrue(isinstance(c,ImageArray), 'Clone not ImageArray')
        self.assertTrue(np.array_equal(c,self.imarr),'Clone not replicating elements')
        self.assertTrue(all([k in c.metadata.keys() for k in \
                    self.imarr.metadata.keys()]), 'Clone not replicating metadata')
        self.assertFalse(shares_memory(c, self.imarr), 'memory overlap on clone') #formal check
        self.imarr['bcd'] = 234
        self.assertTrue('bcd' not in c.metadata.keys(), 'memory overlap for metadata on clone')
        self.imarr['nested'][0] = 2
        self.assertTrue(self.imarr['nested'][0] != c['nested'][0], 'deepcopy not working on metadata')
        self.assertTrue(c.userxyz == 123)
        c.userxyz = 234
        self.assertTrue(c.userxyz != self.imarr.userxyz)
    
    def test_metadata(self):
        self.assertTrue(isinstance(self.imarr.metadata,typeHintedDict))
        self.imarr['testmeta']='abc'
        self.assertTrue(self.imarr['testmeta']=='abc', 'Couldn\'t change metadata')
        del(self.imarr['testmeta'])
        self.assertTrue('testmeta' not in self.imarr.keys(),'Couldn\'t delete metadata')
        #bad data
        def test(imarr):
            imarr.metadata=(1,2,3)
        self.assertRaises(TypeError, test, self.imarr) #check it won't let you do this        
        
    ### test numpy like creation behaviour #
    def test_user_attributes(self):
        self.imarr.abc = 'new att'
        self.assertTrue(hasattr(self.imarr, 'abc'))
        t = ImageArray(self.imarr)
        self.assertTrue(hasattr(t, 'abc'), 'problem copying new attributes')
        t = self.imarr.view(ImageArray) #check array_finalize copies attribute over
        self.assertTrue(hasattr(t, 'abc'))
        t = self.imarr * np.random.random(self.imarr.shape)
        self.assertTrue(isinstance(t, ImageArray), 'problem with ufuncs')
        self.assertTrue(hasattr(t, 'abc'))

    #####  test functionality  ##
    def test_save(self):
        testfile=path.join(thisdir,'coretestdata','testsave.png')
        keys=self.imarr.keys()
        self.imarr.save(filename=testfile)
        load=ImageArray(testfile, asfloat=True)
        #del load["Loaded from"]
        self.assertTrue(all([k in keys for k in load.keys()]), 'problem saving metadata')
        self.assertTrue(np.allclose(self.imarr, load, atol=0.001))
        os.remove(testfile) #tidy up

    def test_max_box(self):
        s=self.imarr.shape
        self.assertTrue(self.imarr.max_box==(0,s[1],0,s[0]))

    def test_crop(self):
        c=self.imarr.crop((1,3,1,4),copy=True)
        self.assertTrue(np.array_equal(c,self.imarr[1:4,1:3]),'crop didn\'t work')
        self.assertFalse(shares_memory(c,self.imarr), 'crop copy failed')
        c2=self.imarr.crop(1,3,1,4,copy=True)
        self.assertTrue(np.array_equal(c2,c),'crop with seperate arguments didn\'t work')
        c3 = self.imarr.crop(box=(1,3,1,4), copy=False)
        self.assertTrue(np.array_equal(c3,c), 'crop with no arguments failed')
        self.assertTrue(shares_memory(self.imarr, c3), 'crop with no copy failed')
    
    def test_asint(self):
        ui = self.imarr.asint()
        self.assertTrue(ui.dtype==np.uint16)
        intarr = np.array([[27312, 43319, 60132, 22149],
                           [ 3944,  9439, 22571, 64980],
                           [19556, 60082, 48378, 50169]], dtype=np.uint16)
        self.assertTrue(np.array_equal(ui,intarr))
        
    def test_other_funcs(self):
        """test imagefuncs add ons. the functions themselves are not checked
        and should include a few examples in the doc strings for testing"""
        self.assertTrue('do_nothing' in dir(self.imarr), 'imagefuncs not being added to dir')
        self.assertTrue('img_as_float' in dir(self.imarr), 'skimage funcs not being added to dir')
        im = self.imarr.do_nothing() #see if it can run
        self.assertTrue(np.allclose(im, self.imarr), 'imagefuncs not working')
        self.assertFalse(shares_memory(im, self.imarr), 'imagefunc failed to clone')
        im0 = ImageArray(np.linspace(0,1,12).reshape(3,4))
        im1 = im0.clone * 5        
        im2 = im1.rescale_intensity() #test skimage
        self.assertTrue(np.allclose(im2, im0), 'skimage func failed')
        self.assertFalse(shares_memory(im2, im1), 'skimage failed to clone')
        im3 = im1.exposure_rescale_intensity() #test call with module name
        self.assertTrue(np.allclose(im3, im0), 'skimage call with module name failed')

class ImageFileTest(unittest.TestCase):
    def setUp(self):
        self.a = np.linspace(0,5,12).reshape(3,4)
        self.ifi = ImageFile(self.a)
    
    def test_properties(self):
        self.assertTrue(np.allclose(self.ifi.image, self.a))
        self.ifi[0,1]=10.1
        self.assertTrue(self.ifi.image[0,1]==10.1)
    
    def test_attrs(self):
        self.assertTrue(self.ifi['Loaded from'] == '')
        self.ifi.abc = 123
        self.assertTrue(self.ifi.image.abc == 123)
        self.assertTrue(self.ifi.abc == 123)
        
    def test_methods(self):
        b=np.arange(12).reshape(3,4)
        ifi = ImageFile(b)
        ifi.asfloat() #convert in place
        self.assertTrue(ifi.image.dtype.kind == 'f')
        ifi.image == ifi.image*5
        ifi.rescale_intensity()
        self.assertTrue(np.allclose(ifi.image, np.linspace(0,1,12).reshape(3,4)))

        
if __name__=="__main__": # Run some tests manually to allow debugging
    test=ImageArrayTest("test_filename")
    test2=ImageFileTest("test_methods")
    unittest.main()

