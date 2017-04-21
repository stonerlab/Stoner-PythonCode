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
    test=ImageArrayTest()
    #test.setUp()
    #test.test_filename()
    unittest.main()