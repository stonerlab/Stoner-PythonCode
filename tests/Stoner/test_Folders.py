# -*- coding: utf-8 -*-
"""
test_Folders.py

Created on Mon Jul 18 14:13:39 2016

@author: phygbu
"""


import unittest
import sys
import os.path as path
import os
import numpy as np
import re
import fnmatch
from numpy import ceil
from Stoner.compat import *
import Stoner.Folders as SF

from Stoner import Data,set_option
import Stoner.HDF5, Stoner.Zip
from Stoner.Util import hysteresis_correct

import matplotlib.pyplot as plt

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../"))
sys.path.insert(0,pth)

class Folders_test(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        pass

    def test_Folders(self):
        self.fldr=SF.DataFolder(self.datadir,debug=False)
        fldr=self.fldr
        fl=len(fldr)
        datfiles=fnmatch.filter(os.listdir(self.datadir),"*.dat")
        length = len([i for i in os.listdir(self.datadir) if path.isfile(os.path.join(self.datadir,i))])-1 # don't coiunt TDMS index
        self.assertEqual(length,fl,"Failed to initialise DataFolder from sample data")
        self.assertEqual(fldr.index(path.basename(fldr[-1].filename)),fl-1,"Failed to index back on filename")
        self.assertEqual(fldr.count(path.basename(fldr[-1].filename)),1,"Failed to count filename with string")
        self.assertEqual(fldr.count("*.dat"),len(datfiles),"Count with a glob pattern failed")
        self.assertEqual(len(fldr[::2]),ceil(len(fldr)/2.0),"Failed to get the correct number of elements in a folder slice")

    def test_discard_earlier(self):
        fldr2=SF.DataFolder(path.join(pth,"tests/Stoner/folder_data"),pattern="*.dat",discard_earlier=True)
        fldr3=SF.DataFolder(path.join(pth,"tests/Stoner/folder_data"),pattern="*.dat")
        self.assertEqual(len(fldr2),1,"Folder created with disacrd_earlier has wrong length ({})".format(len(fldr2)))
        self.assertEqual(len(fldr3),5,"Folder created without disacrd_earlier has wrong length ({})".format(len(fldr3)))
        fldr3.keep_latest()
        self.assertEqual(list(fldr2.ls),list(fldr3.ls),"Folder.keep_latest didn't do the same as discard_earliest in constructor.")

    def test_Operators(self):
        fldr=SF.DataFolder(self.datadir,debug=False)
        fl=len(fldr)
        d=Data(np.ones((100,5)))
        fldr+=d
        self.assertEqual(fl+1,len(fldr),"Failed += operator on DataFolder")
        fldr2=fldr+fldr
        self.assertEqual((fl+1)*2,len(fldr2),"Failed + operator with DataFolder on DataFolder")
        fldr-="Untitled"
        self.assertEqual(len(fldr),fl,"Failed to remove Untitled-0 from DataFolder by name.")
        fldr-="New-XRay-Data.dql"
        self.assertEqual(fl-1,len(fldr),"Failed to remove NEw Xray data by name.")
        fldr+="New-XRay-Data.dql"
        self.assertEqual(len(fldr),fl,"Failed += oeprator with string on DataFolder")
        fldr/="Loaded as"
        self.assertEqual(len(fldr["QDFile"]),4,"Failoed to group folder by Loaded As metadata with /= opeator.")

    def test_Properties(self):
        fldr=SF.DataFolder(self.datadir,debug=False)
        fldr/="Loaded as"
        fldr["QDFile"].group("Byapp")
        self.assertEqual(fldr.mindepth,1,"mindepth attribute of folder failed.")
        self.assertEqual(fldr.depth,2,"depth attribute failed.")
        fldr=SF.DataFolder(self.datadir,debug=False)
        fldr+=Data()
        self.assertEqual(len(list(fldr.loaded)),1,"loaded attribute failed {}".format(len(list(fldr.loaded))))
        self.assertEqual(len(list(fldr.not_empty)),len(fldr)-1,"not_empty attribute failed.")
        fldr-="Untitled"

    def test_methods(self):
        sliced=np.array(['DataFile', 'MDAASCIIFile', 'BNLFile', 'DataFile', 'DataFile',
       'DataFile', 'DataFile', 'MokeFile', 'EasyPlotFile', 'DataFile',
       'DataFile', 'DataFile'],
          dtype='<U12')
        fldr=SF.DataFolder(self.datadir, pattern='*.txt').sort()
        test_sliced=fldr.slice_metadata("Loaded as")
        self.assertEqual(len(sliced),len(test_sliced),"Test slice not equal length - sample-data changed? {}".format(test_sliced))
        self.assertTrue(np.all(test_sliced==sliced),"Slicing metadata failed to work.")

    def test_metadata(self):
        os.chdir(self.datadir)
        fldr6=SF.DataFolder(".",pattern="QD*.dat",pruned=True)
        self.assertEqual(repr(fldr6.metadata),"The DataFolder . has 9 common keys of metadata in 4 Data objects",
                         "Representation method of metadata wrong.")
        self.assertEqual(len(fldr6.metadata),9,"Length of common metadata not right.")
        self.assertEqual(list(fldr6.metadata.keys()),['Byapp',
                                                       'Datatype,Comment',
                                                       'Datatype,Time',
                                                       'Fileopentime',
                                                       'Loaded as',
                                                       'Loaded from',
                                                       'Startupaxis-X',
                                                       'Startupaxis-Y1',
                                                       'Stoner.class'],"metadata.keys() not right.")
        self.assertEqual(len(list(fldr6.metadata.all_keys())),49,"metadata.all_keys() the wrong length.")
        self.assertTrue(isinstance(fldr6.metadata.slice("Loaded from")[0],dict),"metadata.slice not returtning a dictionary.")
        self.assertTrue(isinstance(fldr6.metadata.slice("Loaded from",values_only=True),list),"metadata.slice not returtning a list with values_only=True.")
        self.assertTrue(isinstance(fldr6.metadata.slice("Loaded from",output="Data"),Data),"metadata.slice not returtning Data with outpt='data'.")

    def test_each(self):
        os.chdir(self.datadir)
        fldr6=SF.DataFolder(".",pattern="QD*.dat",pruned=True)
        fldr4=SF.DataFolder(self.datadir,pattern="QD-SQUID-VSM.dat")
        fldr5=fldr4.clone
        shaper=lambda f:f.shape
        fldr6.sort()
        res=fldr6.each(shaper)
        self.assertEqual(res,[(6048, 88), (3025, 41), (1409, 57), (411, 72)],"__call__ on each fauiled.")
        fldr6.each.del_column(0)
        res=fldr6.each(shaper)
        self.assertEqual(res,[(6048, 87), (3025, 40), (1409, 56), (411, 71)],"Proxy method call via each failed")
        paths=['QD-MH.dat', 'QD-PPMS.dat', 'QD-PPMS2.dat','QD-SQUID-VSM.dat']
        filenames=[path.relpath(x,start=fldr6.directory) for x in fldr6.each.filename.tolist()]
        self.assertEqual(filenames,paths,"Reading attributes from each failed.")
        if python_v3:
            eval('(hysteresis_correct@fldr4)(setas="3.xy",saturated_fraction=0.25)')
            self.assertTrue("Hc" in fldr4[0],"Matrix multiplication of callable by DataFolder failed test.")
        fldr5.each(hysteresis_correct,setas="3.xy",saturated_fraction=0.25)
        self.assertTrue("Hc" in fldr5[0],"Call on DataFolder.each() failed to apply function to folder")

    def test_clone(self):
         fldr=SF.DataFolder(self.datadir, pattern='*.txt')
         fldr.abc = 123 #add an attribute
         self.t = fldr.__clone__()
         self.assertTrue(self.t.pattern==fldr.pattern, 'pattern didnt copy over')
         self.assertTrue(hasattr(self.t, "abc") and self.t.abc==123, 'user attribute didnt copy over')
         self.assertTrue(isinstance(self.t['recursivefoldertest'],SF.DataFolder), 'groups didnt copy over')

    def test_grouping(self):
        fldr4=SF.DataFolder()
        x=np.linspace(-np.pi,np.pi,181)
        for phase in np.linspace(0,1.0,5):
            for amplitude in np.linspace(1,2,6):
                for frequency in np.linspace(1,2,5):
                    y=amplitude*np.sin(frequency*x+phase*np.pi)
                    d=Data(x,y,setas="xy",column_headers=["X","Y"])
                    d["frequency"]=frequency
                    d["amplitude"]=amplitude
                    d["phase"]=phase
                    d["params"]=[phase,frequency,amplitude]
                    d.filename="test/{amplitude}/{phase}/{frequency}.dat".format(**d)
                    fldr4+=d
        fldr4.unflatten()
        self.assertEqual(fldr4.mindepth,3,"Unflattened DataFolder had wrong mindepth.")
        self.assertEqual(fldr4.shape, (~~fldr4).shape,"Datafodler changed shape on flatten/unflatten")
        fldr5=fldr4.select(amplitude=1.4,recurse=True)
        fldr5.prune()
        pruned=(0,
                {'test': (0,
                   {'1.4': (0,
                     {'0.0': (5, {}),
                      '0.25': (5, {}),
                      '0.5': (5, {}),
                      '0.75': (5, {}),
                      '1.0': (5, {})})})})
        selected=(0,
                {'test': (0,
                   {'1.4': (0,
                     {'0.25': (1, {}), '0.5': (1, {}), '0.75': (1, {}), '1.0': (1, {})})})})
        self.assertEqual(fldr5.shape,pruned,"Folder pruning gave an unxpected shape.")
        self.assertEqual(fldr5[("test","1.4","0.5",0,"phase")],0.5,"Multilevel indexing of tree failed.")
        shape=(~(~fldr4).select(amplitude=1.4).select(frequency=1).select(phase__gt=0.2)).shape
        self.fldr4=fldr4
        self.assertEqual(shape, selected,"Multi selects and inverts failed.")
        g=(~fldr4)/10
        self.assertEqual(g.shape,(0,{'Group 0': (15, {}),'Group 1': (15, {}),'Group 2': (15, {}),'Group 3': (15, {}),'Group 4': (15, {}),
                                     'Group 5': (15, {}),'Group 6': (15, {}),'Group 7': (15, {}),'Group 8': (15, {}),'Group 9': (15, {})}),"Dive by int failed.")
        g["Group 6"]-=5
        self.assertEqual(g.shape,(0,{'Group 0': (15, {}),'Group 1': (15, {}),'Group 2': (15, {}),'Group 3': (15, {}),'Group 4': (15, {}),
                                     'Group 5': (15, {}),'Group 6': (14, {}),'Group 7': (15, {}),'Group 8': (15, {}),'Group 9': (15, {})}),"Sub by int failed.")
        remove=g["Group 3"][4]
        g["Group 3"]-=remove
        self.assertEqual(g.shape,(0,{'Group 0': (15, {}),'Group 1': (15, {}),'Group 2': (15, {}),'Group 3': (14, {}),'Group 4': (15, {}),
                                     'Group 5': (15, {}),'Group 6': (14, {}),'Group 7': (15, {}),'Group 8': (15, {}),'Group 9': (15, {})}),"Sub by object failed.")
        d=fldr4["test",1.0,1.0].gather(0,1)
        self.assertEqual(d.shape,(181,6),"Gather seems have failed.")
        self.assertTrue(np.all(fldr4["test",1.0,1.0].slice_metadata("phase")==
                               np.ones(5)),"Slice metadata failure.")
        d=(~fldr4).extract("phase","frequency","amplitude","params")
        self.assertEqual(d.shape,(150,6),"Extract failed to produce data of correct shape.")
        self.assertEqual(d.column_headers,['phase', 'frequency', 'amplitude', 'params', 'params', 'params'],"Exctract failed to get correct column headers.")
        p=fldr4["test",1.0,1.0]
        p=SF.PlotFolder(p)
        p.plot()
        self.assertEqual(len(plt.get_fignums()),1,"Failed to generate a single plot for PlotFolder.")
        plt.close("all")

if __name__=="__main__": # Run some tests manually to allow debugging
    test=Folders_test("test_Folders")
    test.setUp()
    test.test_each()
#    test.test_Operators()
#    print("Clone")
#    test.test_clone()
#    print("Folders")
#    test.test_Folders()
#    unittest.main()
#    print("Group")
#    test.test_grouping()
#    print("Each")
#    test.fldr.each.title
    pass
