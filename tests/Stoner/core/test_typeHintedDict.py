#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 15:20:15 2018

@author: phygbu
"""

import unittest
import sys
import  os.path as path
from Stoner.core.base import typeHintedDict

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../"))
#sys.path.insert(0,pth)

class typeHintedDictTest(unittest.TestCase):
    """Test typeHintedDict class"""

    def test_ops(self):
        d = typeHintedDict([('el1',1),('el2',2),('el3',3),('other',4)])
        d.filter('el')
        self.assertTrue(len(d)==3)
        d.filter(lambda x: x.endswith('3'))
        self.assertTrue(len(d)==1)
        self.assertTrue(d['el3']==3)
        d["munge"]=None
        self.assertTrue(d.types["munge"]=="Void","Setting type for None value failed.")
        d["munge"]=1
        self.assertTrue(d["munge{String}"]=="1","Munging return type in getitem failed.")
        self.assertEqual(repr(d),
        """'el3':I32:3
'munge':I32:1""","Repr failed \n{}".format(d))


if __name__=="__main__": # Run some tests manually to allow debugging
    test=typeHintedDictTest("test_ops")
    test.setUp()
    #test.test_properties()
    #test.test_methods()
    #test.test_filter()
#    test.test_deltions()
    #test.test_dir()
    unittest.main()
