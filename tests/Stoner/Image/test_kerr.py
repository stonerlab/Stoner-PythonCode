# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:09:04 2016

@author: phyrct
"""

from Stoner.Image import ImageArray, KerrArray, ImageFile,KerrStack
from Stoner.Core import typeHintedDict
from Stoner import Data
import numpy as np
import unittest
import sys
from os import path
import os

import warnings

#data arrays for testing - some useful small images for tests

testdir=os.path.join(os.path.dirname(__file__),"kerr_testdata")
testdir2=os.path.join(os.path.dirname(__file__),"coretestdata","testims")


def shares_memory(arr1, arr2):
    """Check if two numpy arrays share memory"""
    #ret = arr1.base is arr2 or arr2.base is arr1
    ret = np.may_share_memory(arr1, arr2)
    return ret

class KerrArrayTest(unittest.TestCase):

    def setUp(self):
        self.image=KerrArray(os.path.join(testdir,"kermit3.png"),ocr_metadata=True)
        self.image2=KerrArray(os.path.join(testdir,"kermit3.png"))
        self.ks = KerrStack(testdir2)


    def test_tesseract_ocr(self):
        #this incidently tests get_metadata too
        if not self.image.tesseractable:
            print("#"*80)
            print("Skipping test that uses tesseract.")
            return None
        m=self.image.metadata

        # self.assertTrue(all((m['ocr_scalebar_length_microns']==50.0,
        #                     m['ocr_date']=='11/30/15',
        #                     m['ocr_field'] == -0.13)), 'Misread metadata {}'.format(m))
        keys=('ocr_scalebar_length_pixels', 'ocr_field_of_view_microns',
                          'Loaded from', 'ocr_microns_per_pixel', 'ocr_pixels_per_micron')
        # self.assertTrue(all([k in m.keys() for k in keys]), 'some part of the metadata didn\'t load {}'.format(m))
        m_un=self.image2.metadata
        self.assertTrue('ocr_field' not in m_un.keys(), 'Unannotated image has wrong metadata')

    def test_kerrstack(self):
        print("X"*80+"\n"+"Test Kerrstack")
        ks=self.ks.clone
        ks.normalise(scale=(0,1.0))
        self.assertTrue(np.min(ks.imarray)==0.0 and np.max(ks.imarray)==1.0, 'KerrStack subtract failed min,max: {},{}'.format(np.min(ks.imarray),np.max(ks.imarray)))
        d=ks.hysteresis()
        self.assertTrue(isinstance(d, Data), 'hysteresis didnt return Data')
        self.assertTrue(d.data.shape==(len(ks),2), 'hysteresis didnt return correct shape')

if __name__=="__main__": # Run some tests manually to allow debugging
    #test=KerrArrayTest()
    #test.setUp()
    #test.test_tesseract_ocr()
    unittest.main()