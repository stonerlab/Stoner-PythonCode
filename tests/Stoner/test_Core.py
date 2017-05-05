# -*- coding: utf-8 -*-
"""
test_Core.py
Created on Tue Jan 07 22:05:55 2014

@author: phygbu
"""

import unittest
import sys
import os.path as path
import numpy as np
import re
from numpy import any,all,sqrt

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../"))
sys.path.insert(0,pth)
from Stoner import Data,__home__

class Datatest(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        self.d=Data(path.join(path.dirname(__file__),"CoreTest.dat"),setas="xy")
        self.d2=Data(path.join(__home__,"..","sample-data","TDI_Format_RT.txt"))
        self.d3=Data(path.join(__home__,"..","sample-data","New-XRay-Data.dql"))
        
        
    def test_constructor(self):
        """Constructor Tests"""
        d=Data()
        self.assertTrue(d.shape==(1,0),"Bare constructor failed")
        d=Data(self.d)
        self.assertTrue(np.all(d.data==self.d.data),"Constructor from DataFile failed")
        d=Data([np.ones(100),np.zeros(100)])
        self.assertTrue(d.shape==(100,2),"Constructor from iterable list of nd array failed")
        d=Data([np.ones(100),np.zeros(100)],["X","Y"])
        self.assertTrue(d.column_headers==["X","Y"],"Failed to set column headers in constructor")

    def test_column(self):
        for i,c in enumerate(self.d.column_headers):
            # Column function checks
            self.assertTrue(all(self.d.data[:,i]==self.d.column(i)),"Failed to Access column {} by index".format(i))
            self.assertTrue(all(self.d.data[:,i]==self.d.column(c)),"Failed to Access column {} by name".format(i))
            self.assertTrue(all(self.d.data[:,i]==self.d.column(c[0:3])),"Failed to Access column {} by partial name".format(i))

        # Check that access by list of strings returns multpiple columns
        self.assertTrue(all(self.d.column(self.d.column_headers)==self.d.data),"Failed to access all columns by list of string indices")
        self.assertTrue(all(self.d.column([0,self.d.column_headers[1]])==self.d.data),"Failed to access all columns by mixed list of indices")

        # Check regular expression column access
        self.assertTrue(all(self.d.column(re.compile(r"^X\-"))[:,0]==self.d.column(0)),"Failed to access column by regular expression")
        # Check attribute column access
        self.assertTrue(all(self.d.X==self.d.column(0)),"Failed to access column by attribute name")

    def test_indexing(self):
        #Check all the indexing possibilities
        data=np.array(self.d.data)
        colname=self.d.column_headers[0]
        self.assertTrue(all(self.d.column(colname)==self.d[:,0]),"Failed direct indexing versus column method")
        self.assertTrue(all(self.d[:,0]==data[:,0]),"Failed direct idnexing versusus direct array index")
        self.assertTrue(all(self.d[:,[0,1]]==data),"Failed direct list indexing")
        self.assertTrue(all(self.d[::2,:]==data[::2]),"Failed slice indexing rows")
        self.assertTrue(all(self.d[colname]==data[:,0]),"Failed direct indexing by column name")
        self.assertTrue(all(self.d[:,colname]==data[:,0]),"Failed fallback indexing by column name")
        self.assertEqual(self.d[25,1],645.0,"Failed direct single cell index")
        self.assertEqual(self.d[25,"Y-Data"],645.0,"Failoed single cell index direct")
        self.assertEqual(self.d["Y-Data",25],645.0,"Failoed single cell fallback index order")
        self.d["X-Dat"]=[11,12,13,14,15]
        self.assertEqual(self.d["X-Dat",2],13,"Failed indexing of metadata lists with tuple")
        self.assertEqual(self.d["X-Dat"][2],13,"Failed indexing of metadata lists with double indices")


    def test_len(self):
        # Check that length of the column is the same as length of the data
        self.assertEqual(len(Data()),0,"Empty DataFile not length zero")
        self.assertEqual(len(self.d.column(0)),len(self.d),"Column 0 length not equal to DataFile length")
        self.assertEqual(len(self.d),self.d.data.shape[0],"DataFile length not equal to data.shape[0]")
        # Check that self.column_headers returns the right length
        self.assertEqual(len(self.d.column_headers),self.d.data.shape[1],"Length of column_headers not equal to data.shape[1]")

    def test_attributes(self):
        """Test various atribute accesses,"""
        self.assertEqual(self.d.shape,(100,2),"shape attribute not correct.")
        self.assertTrue(all(self.d.T==self.d.data.T),"Transpose attribute not right")
        self.assertTrue(all(self.d.x==self.d.column(0)),"x attribute quick access not right.")
        self.assertTrue(all(self.d.y==self.d.column(1)),"y attribute not right.")
        self.assertTrue(all(self.d.q==np.arctan2(self.d.data[:,0],self.d.data[:,1])),"Calculated theta attribute not right.")
        self.assertTrue(all(sqrt(self.d[:,0]**2+self.d[:,1]**2)==self.d.r),"Calculated r attribute not right.")
        self.assertTrue(self.d2.records._[5]["Column 2"]==self.d2.data[5,2],"Records and as array attributes problem")


    def test_setas(self):
        #Check readback of setas
        self.assertEqual(self.d.setas[:],["x","y"],"setas attribute not set in constructor")
        self.assertEqual(str(self.d.setas),"xy","setas attribute not not converted to string")
        self.assertTrue(all(self.d.x==self.d.column(0)),"Attribute setas column axis fails")
        self.d.setas(x="Y-Data",reset=False)
        self.assertEqual(self.d.setas[:],["x","x"],"Failed to set setas by type=column keyword assignment")
        self.d.setas(Y="y",reset=False)
        self.assertEqual(self.d.setas[:],["x","y"],"Failed to set setas by column=type keyword assignment")
        self.assertEqual(self.d.setas["x"],"X-Data","Failed to return column name from setas dict reading")
        self.assertEqual(self.d.setas["#x"],0,"Failed to return column index from setas dict reading")
        e=~self.d
        self.assertEqual(str(e.setas),"yx","Failed the invert xy columns operator ~")



    def test_iterators(self):
        for i,c in enumerate(self.d.columns()):
            self.assertTrue(all(self.d.column(i)==c),"Iterating over DataFile.columns not the same as direct indexing column")
        for j,r in enumerate(self.d.rows()):
            self.assertTrue(all(self.d[j]==r),"Iteratinf over DataFile.rows not the same as indexed access")
        for k,r in enumerate(self.d):
            pass
        self.assertEqual(j,k,"Iterating over DataFile not the same as DataFile.rows")
        self.assertEqual(self.d.data.shape,(j+1,i+1),"Iterating over rows and columns not the same as data.shape")

    def test_metadata(self):
        self.d["Test"]="This is a test"
        self.d["Int"]=1
        self.d["Float"]=1.0
        self.assertEqual(self.d["Int"],1)
        self.assertEqual(self.d["Float"],1.0)
        self.assertEqual(self.d["Test"],self.d.metadata["Test"])
        self.assertEqual(self.d.metadata._typehints["Int"],"I32")
        self.assertEqual(len(self.d.dir()),6,"Failed meta data directory listing ({})".format(len(self.d.dir())))
        self.assertEqual(len(self.d3["Temperature"]),7,"Regular experssion metadata search failed")


    def test_dir(self):
        self.assertTrue(self.d.dir("S")==["Stoner.class"],"Dir method failed: dir was {}".format(self.d.dir()))

    def test_filter(self):
        self.d._push_mask()
        ix=np.argmax(self.d.x)
        self.d.filter(lambda r:r.x<=50)
        self.assertTrue(np.max(self.d.x)==50,"Failure of filter method to set mask")
        self.assertTrue(np.isnan(self.d.x[ix]),"Failed to mask maximum value")
        self.d._pop_mask()
        self.assertEqual(self.d2.select(Temp__not__gt=150).shape,(839,3),"Seect method failure.")

    def test_operators(self):
        #Test Column Indexer
        self.assertTrue(all(self.d//0==self.d.column(0)),"Failed the // operator with integer")
        self.assertTrue(all(self.d//"X-Data"==self.d.column(0)),"Failed the // operator with string")
        self.assertTrue(all((self.d//re.compile(r"^X\-"))[:,0]==self.d.column(0)),"Failed the // operator with regexp")
        t=[self.d%1,self.d%"Y-Data",(self.d%re.compile(r"Y\-"))]
        for ix,tst in enumerate(["integer","string","regexp"]):
            self.assertTrue(all(t[ix].data==np.atleast_2d(self.d.column(0)).T),"Failed % operator with {} index".format(tst))
        d=self.d&self.d.x
        self.assertTrue(d.shape[1]==3,"& operator failed.")
        self.assertTrue(len(d.setas)==3,"& messed up setas")
        self.assertTrue(len(d.column_headers)==3,"& messed up setas")
        d&=d.x
        self.assertTrue(d.shape[1]==4,"Inplace & operator failed")
        empty=Data()
        empty=empty+np.zeros(10)
        self.assertEqual(empty.shape,(1,10),"Adding to an empty array failed")
        d=self.d+np.array([0,0])
        self.assertTrue(len(d)==len(self.d)+1,"+ operator failed.")
        d+=np.array([0,0])
        self.assertTrue(len(d)==len(self.d)+2,"Inplace + operator failed.")
        d=d-(-1)
        self.assertTrue(len(d)==len(self.d)+1,"Delete with integer index failed. {} vs {}".format(len(d),len(self.d)+1))
        d-=-1
        self.assertTrue(len(d)==len(self.d),"Inplace delete with integer index failed. {} vs {}".format(len(d),len(self.d)))
        d-=slice(0,-1,2)
        self.assertTrue(len(d)==len(self.d)/2,"Inplace delete with slice index failed. {} vs {}".format(len(d),len(self.d)/2))
        d=d%1
        self.assertTrue(d.shape[1]==1,"Column division failed")
        self.assertTrue(len(d.setas)==d.shape[1],"Column removal messed up setas")
        self.assertTrue(len(d.column_headers)==d.shape[1],"Column removal messed up column headers")
        e=self.d.clone
        f=self.d2.clone
        g=e+f
        self.assertTrue(g.shape==(1776,2),"Add of 2 column and 3 column failed")
        g=f+e
        self.assertTrue(g.shape==(1776,3),"Add of 3 column and 2 column failed.")
        g=e&f
        h=f&e
        self.assertTrue(g.shape==h.shape,"Anding unequal column lengths faile!")
        e=~self.d
        self.assertTrue(e.setas[0]=="y","failed to invert setas columns")
        f.setas="xyz"

    def test_methods(self):
        d=self.d.clone
        d&=np.where(d.x<50,1.0,0.0)
        d.rename(2,"Z-Data")
        d.setas="xyz"
        self.assertTrue(all(d.unique(2)==np.array([0,1])),"Unique values failed: {}".format(d.unique(2)))
        d=self.d.clone
        d.insert_rows(10,np.zeros((2,2)))
        self.assertEqual(len(d),102,"Failed to inert extra rows")
        self.assertTrue(d[9,0]==10 and d[10,0]==0 and d[12,0]==11, "Failed to insert rows properly.")
        d=self.d.clone
        d.add_column(np.ones(len(d)),replace=False,header="added")
        self.assertTrue(d.shape[1]==self.d.shape[1]+1,"Adding a column with replace=False did add a column.")
        self.assertTrue(np.all(d.data[:,-1]==np.ones(len(d))),"Didn't add the new column to the end of the data.")
        self.assertTrue(len(d.column_headers)==len(self.d.column_headers)+1,"Column headers isn't bigger by one")
        self.assertTrue(d.column_headers==self.d.column_headers+["added",],"Column header not added correctly")
        e=d.clone
        d.swap_column([(0,1),(0,2)])
        self.assertTrue(d.column_headers==[e.column_headers[x] for x in [2,0,1]],"Swap column test failed: {}".format(d.column_headers))
        e=self.d(setas="yx")
        self.assertTrue(e.shape==self.d.shape and e.setas[0]=="y","Failed on a DataFile.__call__ test")
        self.assertEqual(len(repr(self.d).split("\n")),102,"Failed to do repr function")
        e=self.d.clone
        e=e.add_column(e.x,header=e.column_headers[0])
        e.del_column(duplicates=True)
        self.assertTrue(e.shape==(100,2),"Deleting duplicate columns failed")
        e=self.d2.clone
        e.reorder_columns([2,0,1])
        self.assertTrue(e.column_headers==[self.d2.column_headers[x] for x in [2,0,1]],"Failed to reorder columns: {}".format(e.column_headers))
        
    def test_setas_metadata(self):
        d=self.d.clone
        d.setas="xyz"
        self.assertEqual(repr(d.setas),"['x', 'y']","setas __repr__ failure {}".format(repr(d.setas)))
        self.assertEqual(d.find_col(slice(5)),[0,1],"findcol with a slice failed {}".format(d.find_col(slice(5))))
        d=self.d2.clone
        d.setas="xyz"
        self.assertTrue(d["Column 2",5]==d[5,"Column 2"],"Indexing with mixed integer and string failed.")
        self.assertEqual(d.metadata.type(["User","Timestamp"]),['String', 'String'],"Metadata.type with slice failed")        
        d.data["Column 2",:]=np.zeros(len(d)) #TODO make this work with d["Column 2",:] as well
        self.assertTrue(d.z.max()==0.0 and d.z.min()==0.0,"Failed to set Dataarray using string indexing")


if __name__=="__main__": # Run some tests manually to allow debugging
    test=Datatest("test_operators")
    test.setUp()
    test.test_constructor()
    test.test_attributes()
    test.test_operators()
    test.test_methods()
    test.test_setas_metadata()
    test.test_filter()