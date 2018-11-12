# -*- coding: utf-8 -*-
"""
Created on Fri Mar 03 18:21:52 2017

@author: phyrct
"""

from Stoner import Data
from Stoner.Image import ImageArray, ImageFolder, ImageStack, KerrStack,ImageStack2
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
        self.ks_dir = ImageStack(testdir)
        self.ks = ImageStack(self.td) #load in two ways
        self.ks2_dir = ImageStack2(testdir, pattern='*.png')
        self.ks2 = ImageStack2(self.td) #load in two ways

    def test_load(self):
        self.assertTrue(len(self.td)==len(os.listdir(testdir)), "Didn't find all the images")
        self.assertTrue(len(self.ks)==len(os.listdir(testdir)),"ImageStack conversion from ImageFolder failed")
        self.assertTrue(len(self.ks_dir)==len(os.listdir(testdir)),"IamgeStack read from directory failed")
        self.assertTrue(len(self.ks2)==len(os.listdir(testdir)),"ImageStack2 conversion from ImagerFolder failed")
        self.assertTrue(len(self.ks2_dir)==len(os.listdir(testdir)),"ImageStack2 read from directory failed.")

        self.assertTrue(isinstance(self.td[0],ImageArray), 'Getting an image array from the ImageFolder failed type is {}'.format(type(self.td[0]))) #'{}, '.format(isinstance(self.td[0], ImageArray))#
        #self.assertTrue(self.td.slice_metadata(key='field',values_only=True)==knownfieldvals, 'slice metadata failed')

    def test_clone(self):
        c=self.ks.clone
        c.imarray[0,0,0] = 15.534
        self.assertFalse(np.array_equal(self.ks, c), 'clone failed to create new array')



if __name__=="__main__":
    #t=ImageFolder(testdir)
    #ti=KerrStack(t)
    #test=ImageFolderTest()
    #test.setUp()
    #test.test_kerrstack()
    #test.test_load()
    unittest.main()
