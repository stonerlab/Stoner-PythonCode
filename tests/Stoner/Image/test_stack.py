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

    def test_ImageStack2(self):
        self.istack2=ImageStack2()
        for theta in np.linspace(0,360,91):
            i=ImageFile(np.zeros((100,100)))
            x,y=10*np.cos(np.pi*theta/180)+50,10*np.sin(np.pi*theta/180)+50
            i.draw.circle(x,y,25)
            self.istack2.insert(0,i)
        self.assertTrue(self.istack2.shape==(100,100,91),"ImageStack2.shape wrong at {}".format(self.istack2.shape))
        i=ImageFile(np.zeros((100,100))).draw.circle(50,50,25)
        self.istack2.align(i,method="imreg_dft")
        data=self.istack2.slice_metadata(["tvec","angle","scale"],output="Data")
        self.assertTrue(data.shape==(91,4),"Slice metadata went a bit funny")
        self.assertTrue(sorted(data.column_headers)==['angle','scale','tvec_0', 'tvec_1'],"slice metadata column headers wrong at {}".format(data.column_headers))


if __name__=="__main__":
    #t=ImageFolder(testdir)
    #ti=KerrStack(t)
    test=ImageStackTest("test_imagestack")
    #test.setUp()
    #test.test_kerrstack()
    #test.test_load()
    unittest.main()
    test.test_ImageStack2()
    st=test.istack2
