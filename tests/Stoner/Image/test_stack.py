#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 21:22:18 2018

@author: phygbu
"""
from Stoner import Data
from Stoner.Image import ImageFile,ImageFolder, ImageStack, KerrStack
from Stoner.Image.stack import ImageStack2
import numpy as np
import unittest
from os import path
import os

testdir=os.path.join(os.path.dirname(__file__),"coretestdata","testims")

class ImageStackTest(unittest.TestCase):

    def setUp(self):
        self.td = ImageFolder(testdir, pattern='*.png')
        self.ks = ImageStack(testdir)
        self.ks = ImageStack(self.td) #load in two ways
        self.assertTrue(len(self.ks)==len(os.listdir(testdir)))
        self.istack2=ImageStack2()
        for theta in np.linspace(0,360,91):
            i=ImageFile(np.zeros((100,100)))
            x,y=10*np.cos(np.pi*theta/180)+50,10*np.sin(np.pi*theta/180)+50
            i.draw.circle(x,y,25)
            self.istack2.insert(0,i)

    def test_imagestack(self):
        ks=self.ks.clone
        ks.append(self.ks[0].clone)
        self.assertTrue(len(ks)==len(self.ks)+1, 'ImageStack append failed')
        self.assertTrue(len(ks.allmeta)==len(self.ks.allmeta)+1, 'ImageStack append failed')
        ks.insert(1, self.ks[0].clone)
        del(ks[1:3])
        self.assertTrue(len(ks)==len(self.ks), 'ImageStack insert or del failed')

    def test_kerrstack(self):
        ks=KerrStack(self.ks)
        self.assertTrue(np.min(ks.imarray)==0.0 and np.max(ks.imarray)==1.0, 'KerrStack subtract failed min,max: {},{}'.format(np.min(ks.imarray),np.max(ks.imarray)))
        d=ks.hysteresis()
        self.assertTrue(isinstance(d, Data), 'hysteresis didnt return Data')
        self.assertTrue(d.data.shape==(len(ks),2), 'hysteresis didnt return correct shape')

class ImageStack2Test(unittest.TestCase):
    
    def setUp(self):
        self.td = ImageFolder(testdir, pattern='*.png')
        self.ks = ImageStack2(testdir)
        self.ks = ImageStack2(self.td) #load in two ways
        self.assertTrue(len(self.ks)==len(os.listdir(testdir)))
        self.istack2=ImageStack2()
        for theta in np.linspace(0,360,91):
            i=ImageFile(np.zeros((100,100)))
            x,y=10*np.cos(np.pi*theta/180)+50,10*np.sin(np.pi*theta/180)+50
            i.draw.circle(x,y,25)
            self.istack2.insert(0,i)
    
    def test_ImageStack2(self):
        self.assertTrue(self.istack2.shape==(91,100,100),"ImageStack2.shape wrong at {}".format(self.istack2.shape))
        i=ImageFile(np.zeros((100,100))).draw.circle(50,50,25)
        self.m1=self.istack2.mean()
        self.istack2.align(i,method="imreg_dft")
        data=self.istack2.slice_metadata(["tvec","angle","scale"],output="Data")
        self.assertTrue(data.shape==(91,4),"Slice metadata went a bit funny")
        self.assertTrue(sorted(data.column_headers)==['angle','scale','tvec_0', 'tvec_1'],"slice metadata column headers wrong at {}".format(data.column_headers))
        self.m2=self.istack2.mean()
        self.assertTrue(np.abs(self.m1.mean()-self.m2.mean())/self.m1.mean()<1E-2,"Problem calculating means of stacks.")
        s1=self.istack2[:,45:55,45:55]
        s2=self.istack2[:,50,:]
        s3=self.istack2[:,:,50]
        s4=self.istack2[50,:,:]
        self.assertEqual(s1.shape,(91,10,10),"3D slicing to produce 3D stack didn't work.")
        self.assertEqual(s2.shape,(91,100),"3D slicing to 2D section z-y plane failed.")
        self.assertEqual(s3.shape,(91,100),"3D slicing to 2D section z-x plane failed.")
        self.assertEqual(s4.shape,(100,100),"3D slicing to 2D section x-y plane failed.")
        self.assertEqual(len(self.istack2.images),91,"len(ImageFolder.images) failed.")
        sa=[]
        for im in self.istack2.images:
            sa.append(im.shape)
        sa=np.array(sa)
        self.assertTrue(np.all(sa==np.ones((91,2))*100),"Result from iterating over images failed.")
        self.istack2.adjust_contrast()
        self.assertEqual((np.array(self.istack2.min()).mean(),np.array(self.istack2.max()).mean()),(-1.0,1.0),"Adjust contrast failure")
        self.im1=self.istack2[0]
        self.im1.normalise()
        self.im1.convert(np.int32)
        self.im2=self.im1.convert(np.float32)
        conv_err=(self.istack2[0].image-self.im2.image).max()
        self.assertTrue(conv_err<1E-7,"Problems double converting images:{}.".format(conv_err))
        self.im1=self.istack2[0].convert(np.int64)
        self.im1=self.im1.convert(np.int8)
        self.im2=self.istack2[0].convert(np.int8)
        self.assertTrue(abs((self.im2-self.im1).max())<=2.0,"Failed up/down conversion to integer images.")
   
    def test_init(self):
        #try to init with a few different call sequences
        listinit = []
        for i in range(10):
            listinit.append(np.arange(12).reshape(3,4))
        npinit = np.arange(1000).reshape(5,10,20)
        listinit = ImageStack2(listinit)
        self.assertTrue(listinit.shape == (10,3,4), "problem with initialising ImageStack2 with list of data")
        npinitist = ImageStack2(npinit)
        self.assertTrue(np.allclose(npinitist.imarray,npinit), "problem initiating with 3d numpy array")
        ist2init = ImageStack2(self.istack2)
        self.assertTrue(np.allclose(ist2init.imarray,self.istack2.imarray), "problem initiating with other ImageStack")
        self.assertTrue(all([k in ist2init[0].metadata.keys() for k in self.istack2[0].metadata.keys()]), 
                        "problem with metadata when initiating with other ImageStack")
        imfinit = ImageStack2(self.td) #init with another ImageFolder
        self.assertTrue(len(imfinit)==8, "Couldn't load from another ImageFolder object")
    
    def test_accessing(self):
        #ensure we can write and read to the stack in different ways 
        ist2 = ImageStack2(np.arange(60).reshape(4,3,5))
        im = np.zeros((3,5), dtype=int)
        ist2[0].image = im
        ist2[1] = im
        ist2[2].image[:] = im
        ist2[3][0,0] = 100
        ist2[3].image[0,1] = 101
        #ist2[3,0,2] = 102 may want to support this type of index accessing in the future?
        for i in range(3):
            self.assertTrue(np.allclose(ist2[i],im))
        self.assertTrue(np.allclose(ist2[3][0,0],100))
        self.assertTrue(np.allclose(ist2[3][0,1],101))
        #check imarray behaviour
        ist2.imarray = np.arange(60).reshape(4,3,5)*3
        self.assertTrue(np.allclose(ist2.imarray, np.arange(60).reshape(4,3,5)*3))
        ist2.imarray[1,2,3] = 500
        self.assertTrue(ist2.imarray[1,2,3] == 500)
        #check slice access #not implemented yet
        #im2 = np.zeros((2,3,5))
        #ist2[3:5] = im2 #try setting two images at once
        #self.assertTrue(np.allclose(ist2[4],im))
        
        
    def test_methods(self):
        #check function generator machinery works
        self.istack2.crop(0,30,0,50)
        self.assertTrue(self.istack2.shape==(91,50,30),"Unexpected size of imagestack2 got {} for 91x50x30".format(self.istack2.shape))
        ist2 = ImageStack2(np.arange(60).reshape(4,3,5))
        self.assertTrue(issubclass(ist2.imarray.dtype.type, np.integer),"Unexpected dtype in image stack2 got {} not int32".format(ist2.imarray.dtype))
        t1 = ImageStack2(np.arange(60).reshape(4,3,5))
        t1.asfloat(normalise=False, clip_negative=False)
        self.assertTrue(t1.imarray.dtype==np.float64)
        self.assertTrue(np.max(t1.imarray)==59.0)
        t2 = ImageStack2(np.arange(60).reshape(4,3,5))
        t2.asfloat(normalise=True, clip_negative=True)
        #self.assertTrue( np.max(t2.imarray) == (2*59+1)/(2**31-(-2**31)) )
        self.assertTrue(np.min(t2.imarray)>=0)
        ist3 = ist2.clone
        self.assertFalse(np.may_share_memory(ist2.imarray, ist3.imarray))
        del ist3[-1]
        self.assertTrue(len(ist3)==len(ist2)-1)
        self.assertTrue(np.allclose(ist3[0],ist2[0]))
        ist3.insert(1, np.arange(18).reshape(3,6))
        self.assertTrue(ist3[1].shape==(3,6), 'inserting an image of different size to stack')
        

if __name__=="__main__":
    test=ImageStack2Test()
    test.setUp()
    #test.test_methods()
    unittest.main()
   
