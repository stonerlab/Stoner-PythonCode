# -*- coding: utf-8 -*-
"""
test_Core.py
Created on Tue Jan 07 22:05:55 2014

@author: phygbu
"""

import unittest
import sys
import os, os.path as path
import numpy as np
import re
from numpy import any,all,sqrt,nan
from collections import MutableMapping,OrderedDict

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../"))
sys.path.insert(0,pth)
from Stoner import Data,__home__
from Stoner.Core import typeHintedDict,metadataObject
import Stoner.compat

def extra_get_filedialog(what="file", **opts):
    return path.join(path.dirname(__file__),"CoreTest.dat")

def mask_func(r):
    return np.abs(r.q)<0.25*np.pi


Stoner.Core.get_filedialog=extra_get_filedialog #Monkey patch to test the file dialog

class Datatest(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        self.d=Data(path.join(path.dirname(__file__),"CoreTest.dat"),setas="xy")
        self.d2=Data(path.join(__home__,"..","sample-data","TDI_Format_RT.txt"))
        self.d3=Data(path.join(__home__,"..","sample-data","New-XRay-Data.dql"))
        self.d4=Data(path.join(__home__,"..","sample-data","Cu_resistivity_vs_T.txt"))

    def test_repr(self):
        header="""==============  ======  ======
TDI Format 1.5  X-Data  Y-Data
    index       0 (x)   1 (y)
==============  ======  ======"""
        repr_header="""=============================  ========  ========
TDI Format 1.5                   X-Data    Y-Data
index                             0 (x)     1 (y)
=============================  ========  ========"""
        self.assertEqual(self.d.header,header,"Header output changed.")
        self.assertEqual("\n".join(repr(self.d).split("\n")[:4]),repr_header,"Representation gave unexpcted header.")

    def test_base_class(self):
        m=metadataObject()
        m.update(self.d2.metadata)
        self.assertTrue(isinstance(m,MutableMapping),"metadataObject is not a MutableMapping")
        self.assertEqual(len(m),len(self.d2.metadata),"metadataObject length failure.")
        self.assertEqual(len(m[".*"]),len(m),"Failure to index by regexp.")
        for k in self.d2.metadata:
            self.assertEqual(m[k],self.d2.metadata[k],"Failure to have equal keys for {}".format(k))
            m[k]=k
            self.assertEqual(m[k],k,"Failure to set keys for {}".format(k))
            del m[k]
        self.assertEqual(len(m),0,"Failed to delete from metadataObject.")

    def test_constructor(self):
        """Constructor Tests"""
        d=Data()
        self.assertTrue(d.shape==(1,0),"Bare constructor failed")
        d=Data(self.d)
        self.assertTrue(np.all(d.data==self.d.data),"Constructor from DataFile failed")
        d=Data([np.ones(100),np.zeros(100)])
        self.assertTrue(d.shape==(100,2),"Constructor from iterable list of nd array failed")
        d=Data([np.ones(100),np.zeros(100)],["X","Y"])
        self.assertTrue(d.column_headers==["X","Y"],"Failed to set column headers in constructor: {}".format(d.column_headers))
        c=np.zeros(100)
        d=Data({"X-Data":c,"Y-Data":c,"Z-Data":c})
        self.assertEqual(d.shape,(100,3),"Construction from dictionary of columns failed.")
        d=Data(False)
        self.assertEqual(self.d,d,"Faked file-dialog test")

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

    def test_deltions(self):
        ch=["{}-Data".format(chr(x)) for x in range(65,91)]
        data=np.zeros((100,26))
        metadata=OrderedDict([("Key 1",True),("Key 2",12),("Key 3","Hellow world")])
        self.dd=Data(metadata)
        self.dd.data=data
        self.dd.column_headers=ch
        self.dd.setas="3.x3.y3.z"
        self.repr_string="""===========================  ========  =======  ========  =======  ========  ========
TDI Format 1.5                 D-Data   ....      H-Data   ....      Y-Data    Z-Data
index                           3 (x)              7 (y)                 24        25
===========================  ========  =======  ========  =======  ========  ========
Key 1{Boolean}= True                0                  0                  0         0
Key 2{I32}= 12                      0  ...             0  ...             0         0
Key 3{String}= Hellow world         0  ...             0  ...             0         0
Stoner.class{String}= Data          0  ...             0  ...             0         0
...                                 0  ...             0  ...             0         0"""
        self.assertEqual("\n".join(repr(self.dd).split("\n")[:9]),self.repr_string,"Representation with interesting columns failed.")
        del self.dd["Key 1"]
        self.assertEqual(len(self.dd.metadata),3,"Deletion of metadata failed.")
        del self.dd[20:30]
        self.assertEqual(self.dd.shape,(90,26),"Deleting rows directly failed.")
        self.dd.del_column("Q")
        self.assertEqual(self.dd.shape,(90,25),"Deleting rows directly failed.")
        self.dd%=3
        self.assertEqual(self.dd.shape,(90,24),"Deleting rows directly failed.")
        self.dd.setas="x..y..z"
        self.dd.del_column(self.dd.setas.not_set)
        self.assertEqual(self.dd.shape,(90,3),"Deleting rows directly failed.")
        self.dd.mask[50,1]=True
        self.dd.del_column()
        self.assertEqual(self.dd.shape,(90,2),"Deletion of masked rows failed.")

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
        d=Data(np.ones((10,10)))
        d[0,0]=5 #Index by tuple into data
        d["Column_1",0]=6 # Index by column name, row into data
        d[0,"Column_2"]=7 #Index by row, column name into data
        d["Column_3"]=[1,2,3,4] # Create a metadata
        d["Column_3",2]=2 # Index existing metadata via tuple
        d.metadata[0,5]=10
        d[0,5]=12 # Even if tuple, index metadata if already existing.
        self.assertTrue(np.all(d[0]==np.array([5,6,7,1,1,1,1,1,1,1])),"setitem on Data to index into Data.data failed.\n{}".format(d[0]))
        self.assertEqual(d.metadata["Column_3"],[1,2,2,4],"Tuple indexing into metadata Failed.")
        self.assertEqual(d.metadata[0,5],12,"Indexing of pre-existing metadta keys rather than Data./data failed.")
        self.assertEqual(d.metadata[1],[1, 2, 2, 4],"Indexing metadata by integer failed.")

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
        keys,values=zip(*self.d.items())
        self.assertEqual(tuple(self.d.keys()),keys,"Keys from items() not equal to keys from keys()")
        self.assertEqual(tuple(self.d.values()),values,"Values from items() not equal to values from values()")
        self.assertEqual(self.d["Int"],1)
        self.assertEqual(self.d["Float"],1.0)
        self.assertEqual(self.d["Test"],self.d.metadata["Test"])
        self.assertEqual(self.d.metadata._typehints["Int"],"I32")
        self.assertEqual(len(self.d.dir()),6,"Failed meta data directory listing ({})".format(len(self.d.dir())))
        self.assertEqual(len(self.d3["Temperature"]),7,"Regular experssion metadata search failed")

    def test_dir(self):
        self.assertTrue(self.d.dir("S")==["Stoner.class"],"Dir method failed: dir was {}".format(self.d.dir()))
        bad_keys=set(['__metaclass__', 'iteritems', 'iterkeys', 'itervalues','__ge__', '__gt__', '__init_subclass__',
                      '__le__', '__lt__', '__reversed__', '__slots__',"_abc_negative_cache","_abc_registry",
                      "_abc_negative_cache_version","_abc_cache","_abc_impl"])
        self.attrs=set(dir(self.d))-bad_keys
        if len(self.attrs)!=226:
            expected={'_conv_string', '__str__', 'clear', 'scale', '__add__', 'popitem',  'priority', '_init_single', 'pop', 'reorder_columns', '_col_label', 'subclasses', '__sizeof__', 'rows', 'plot_matrix', '_PlotMixin__SurfPlotter', '_showfig', '__and__', '_repr_html_', '_pop_mask', 'filename', 'smooth', '__weakref__', 'dir', '_PlotMixin__mpl3DQuiver', 'spline', '__format__', 'plot_xy', 'labels', '_fix_kargs', 'span', '__getattr__', '_push_mask', 'normalise', 'mean', 'sort', '_set_mask', 'shape', 'x2', 'clone', 'save', 'setdefault', 'update', 'plot', 'colormap_xyz', 'plot_xyzuvw', '_patterns', 'section', 'count', 'extrapolate', 'fig', 'mime_type', 'find_col', 'SG_Filter', '__isub__', 'clip', '_fix_fig', '_AnalysisMixin__get_math_val', 'get_filename', 'griddata', 'axes', 'setas', '__sub__', '__floordiv__', '_baseclass', '_subplots', '__add_core__', 'max', 'rename', '__setstate__', 'rolling_window', '_labels', '__dict__', 'adjust_setas', 'column', '__mod__', 'xlim', '_repr_table_', '__abstractmethods__', '__call__', '_public_attrs_real', '__eq__', '_pyplot_proxy', 'stitch', 'lmfit', '_filename', '__lshift__', '_fix_titles', 'multiple', '__dir__', 'column_headers', 'header', '_public_attrs', 'swap_column', 'annotate_fit', '_init_many', 'min', '_load', '__sub_core__', 'dtype', '__doc__', '_col_args', 'filter', '__new__', '__len__', 'format', 'ylim', 'ax', '__hash__', '_PlotMixin__figure', 'polyfit', '__repr__', 'subtract', '__iand__', 'debug', 'diffsum', 'split', 'ylabel', '__iter__', '_vector_color', '__invert__', '__repr_core__', 'del_nan', 'y2', '__contains__', '__reduce__', 'plot_xyuv', 'plot_xyuvw', '__delattr__', 'curve_fit', '__module__', '_conv_float', '__getattribute__', 'keys', 'legend', 'quiver_plot', 'metadata', 'plot_xyz', '__regexp_meta__', 'data', 'figure', 'records', '_DataFile__parse_metadata', 'fignum', '__setattr__', 'insert_rows', 'add', 'make_bins', '_DataFile__setattr_col', '_repr_short_', '__getitem__', '_repr_limits', '_data', '_template', 'outlier_detection', 'template', '__imod__', 'image_plot', '__setitem__', '_get_curve_fit_data', '_DataFile__search_index', 'no_fmt', '__class__', '_AnalysisMixin__threshold', 'dims', 'threshold', 'basename', '_AnalysisMixin__lmfit_one', 'del_rows', 'patterns', 'del_column', '_metadata', '__deepcopy__', 'search', '_record_curve_fit_result', 'positional_fmt', '__getstate__', 'interpolate', 'dict_records', '_span_slice', 'columns', 'title', '__delitem__', '_repr_html_private', '_DataFile__file_dialog', '_getattr_col', 'mask', 'add_column', 'subplot', 'subplots', 'peaks', '_MutableMapping__marker', 'subplot2grid', 'get', '_DataFile__read_iterable', 'contour_xyz', 'inset', '__meta__', '_VectorFieldPlot', '__init__', '_masks', 'select', 'unique', 'xlabel', '_Plot', '__iadd__', 'values', 'multiply', '_raise_type_error', 'divide', 'odr', '__reduce_ex__', 'cmap', 'showfig', '__subclasshook__', 'items', '_interesting_cols', '_init_double', '__ne__', '_fix_cols', 'integrate', 'decompose', 'bin', 'closest', '__and_core__', 'T', 'load', 'apply'}
            print("="*120,"\n","Warning=====>",self.attrs-expected,expected-self.attrs)
        self.assertEqual(len(self.attrs),226,"DataFile.__dir__ failed.")

    def test_filter(self):
        self.d._push_mask()
        ix=np.argmax(self.d.x)
        self.d.filter(lambda r:r.x<=50)
        self.assertTrue(np.max(self.d.x)==50,"Failure of filter method to set mask")
        self.assertTrue(np.isnan(self.d.x[ix]),"Failed to mask maximum value")
        self.d._pop_mask()
        self.assertEqual(self.d2.select(Temp__not__gt=150).shape,(839,3),"Seect method failure.")
        self.assertEqual(self.d.select(lambda r:r.x<30),self.d.select(X__lt=30),"Select  method as callable failed.")
        self.assertEqual(self.d.select(__=lambda r:r.x<30),self.d.select(X__lt=30),"Select  method as callable failed.")

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

    def test_properties(self):
        self.little=Data()
        p=np.linspace(0,np.pi,91)
        q=np.linspace(0,2*np.pi,91)
        r=np.cos(p)
        x=r*np.sin(q)
        y=r*np.cos(q)
        self.little.data=np.column_stack((x,y,r))
        self.little.setas="xyz"
        q_ang=np.round(self.little.q/np.pi,decimals=2)
        p_ang=np.round(self.little.p/np.pi,decimals=2)
        self.assertTrue(np.max(q_ang)==1.0,"Data.q failure")
        self.assertTrue(np.max(p_ang)==0.5,"Data.p failure")
        self.assertTrue(np.min(p_ang)==-0.5,"Data.p failure")
        self.little.mask=mask_func
        self.little.mask[:,2]=True
        self.assertEqual(len(repr(self.little).split("\n")),len(self.little)+5,"Non shorterned representation failed.")

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
        self.assertTrue(d.shape[1]==self.d.shape[1]+1,"Adding a column with replace=False did not add a column.")
        self.assertTrue(np.all(d.data[:,-1]==np.ones(len(d))),"Didn't add the new column to the end of the data.")
        self.assertTrue(len(d.column_headers)==len(self.d.column_headers)+1,"Column headers isn't bigger by one")
        self.assertTrue(d.column_headers==self.d.column_headers+["added",],"Column header not added correctly")
        d=self.d.clone
        d.add_column(self.d.x)
        self.assertTrue(np.all(d.x==d[:,2]),"Adding a column as a DataArray with column headers didn't work")
        self.assertTrue(d.x.column_headers[0]==d.column_headers[2],"Adding a column as a DataArray with column headers didn't work")
        e=d.clone
        d.swap_column([(0,1),(0,2)])
        self.assertTrue(d.column_headers==[e.column_headers[x] for x in [2,0,1]],"Swap column test failed: {}".format(d.column_headers))
        e=self.d(setas="yx")
        self.assertTrue(e.shape==self.d.shape and e.setas[0]=="y","Failed on a DataFile.__call__ test")
        spl=len(repr(self.d).split("\n"))
        self.assertEqual(spl,105,"Failed to do repr function got {} lines".format(spl))
        e=self.d.clone
        e=e.add_column(e.x,header=e.column_headers[0])
        e.del_column(duplicates=True)
        self.assertTrue(e.shape==(100,2),"Deleting duplicate columns failed")
        e=self.d2.clone
        e.reorder_columns([2,0,1])
        self.assertTrue(e.column_headers==[self.d2.column_headers[x] for x in [2,0,1]],"Failed to reorder columns: {}".format(e.column_headers))
        d=self.d.clone
        d.del_rows(0,10.0)
        self.assertEqual(d.shape,(99,2),"Del Rows with value and column failed - actual shape {}".format(d.shape))
        d=self.d.clone
        d.del_rows(0,(10.0,20.0))
        self.assertEqual(d.shape,(89,2),"Del Rows with tuple and column failed - actual shape {}".format(d.shape))
        d=self.d.clone
        d.mask[::2,0]=True
        d.del_rows()
        self.assertEqual(d.shape,(50,2),"Del Rows with mask set - actual shape {}".format(d.shape))
        d=self.d.clone
        d[::2,1]=nan
        d.del_nan(1)
        self.assertEqual(d.shape,(50,2),"del_nan with explicit column set failed shape was {}".format(d.shape))
        d=self.d.clone
        d[::2,1]=nan
        d.del_nan(0)
        self.assertEqual(d.shape,(100,2),"del_nan with explicit column set and not nans failed shape was {}".format(d.shape))
        d=self.d.clone
        d[::2,1]=nan
        d.setas=".y"
        d.del_nan()
        self.assertEqual(d.shape,(50,2),"del_nan with columns from setas failed shape was {}".format(d.shape))
        d=self.d.clone
        d2=self.d.clone
        d2.data=d2.data[::-1,:]
        self.assertEqual(d.sort(reverse=True),d2,"Sorting revserse not the same as manually reversed data.")

    def test_metadata_save(self):
        local = path.dirname(__file__)
        t = np.arange(12).reshape(3,4) #set up a test data file with mixed metadata
        t = Data(t)
        t.column_headers = ["1","2","3","4"]
        metitems = [True,1,0.2,{"a":1, "b":"abc"},(1,2),np.arange(3),[1,2,3], "abc", #all types accepted
                    r"\\abc\cde", 1e-20, #extra tests
                    [1,(1,2),"abc"], #list with different types
                    [[[1]]], #nested list
                    None, #None value
                    ]
        metnames = ["t"+str(i) for i in range(len(metitems))]
        for k,v in zip(metnames,metitems):
            t[k] = v
        t.save(path.join(local, "mixedmetatest.dat"))
        tl = Data(path.join(local, "mixedmetatest.txt")) #will change extension to txt if not txt or tdi, is this what we want?
        t2 = self.d4.clone  #check that python tdi save is the same as labview tdi save
        t2.save(path.join(local, "mixedmetatest2.txt"))
        t2l = Data(path.join(local, "mixedmetatest2.txt"))
        for orig, load in [(t,tl), (t2, t2l)]:
            for k in ['Loaded as', 'TDI Format']:
                orig[k]=load[k]
            self.assertTrue(np.allclose(orig.data, load.data))
            self.assertTrue(orig.column_headers==load.column_headers)
            self.res=load.metadata^orig.metadata
            self.assertTrue(load.metadata==orig.metadata,"Metadata not the same on round tripping to disc")
        os.remove(path.join(local, "mixedmetatest.txt")) #clear up
        os.remove(path.join(local, "mixedmetatest2.txt"))

if __name__=="__main__": # Run some tests manually to allow debugging
    test=Datatest("test_operators")
    test.setUp()
    #test.test_properties()
    #test.test_methods()
    #test.test_filter()
#    test.test_deltions()
    #test.test_dir()
    test.test_metadata_save()
    #unittest.main()
