# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 18:21:52 2017

@author: phyrct
"""

from Stoner import Data
from Stoner.Image import ImageArray, ImageFolder, ImageStack, KerrStack
import numpy as np
import unittest
from os import path
import os

knownkeys = ['Averaging', 'Comment:', 'Contrast Shift', 'HorizontalFieldOfView', 
             'Lens', 'Loaded from', 'Magnification', 'MicronsPerPixel', 'field: units', 
             'field: units', 'filename', 'subtraction']
knownfieldvals = [-233.432852, -238.486666, -243.342465, -248.446173, 
                  -253.297813, -258.332918, -263.340476, -268.20511]

testdir=os.path.join(os.path.dirname(__file__),"coretestdata","testims")

class ImageFolderTest(unittest.TestCase):

    def setUp(self):
        self.td = ImageFolder(testdir, pattern='*.png')
        self.td=self.td.sort(key=lambda x:x.filename.lower())
        self.ks = ImageStack(testdir)
        self.ks = ImageStack(self.td) #load in two ways
        self.assertTrue(len(self.ks)==len(os.listdir(testdir)))

    def test_load(self):
        self.assertTrue(len(self.td)==len(os.listdir(testdir)), "Didn't find all the images")
        self.assertTrue(isinstance(self.td[0],ImageArray), 'Getting an image array from the ImageFolder failed') #'{}, '.format(isinstance(self.td[0], ImageArray))#
        #self.assertTrue(self.td.slice_metadata(key='field',values_only=True)==knownfieldvals, 'slice metadata failed')
    
    def test_clone(self):
        c=self.ks.clone
        c.imarray[0,0,0] = 15.534
        self.assertFalse(np.array_equal(self.ks, c), 'clone failed to create new array')
        
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

     
if __name__=="__main__":
    #t=ImageFolder(testdir)
    #ti=KerrStack(t)
    test=ImageFolderTest("test_load")
    #test.setUp()
    #test.test_kerrstack()
    #test.test_load()
    unittest.main()
