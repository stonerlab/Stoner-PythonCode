"""test_Core.py
Created on Tue Jan 07 22:05:55 2014

@author: phygbu
"""

import sys
import os, os.path as path
import numpy as np
import re
import pandas as pd
import pytest
from numpy import all,sqrt,nan
from collections import MutableMapping,OrderedDict

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../"))
sys.path.insert(0,pth)
from Stoner import Data,__home__
from Stoner.Core import metadataObject
import Stoner.compat

def mask_func(r):
    return np.abs(r.q)<0.25*np.pi


datadir=path.join(pth,"sample-data")

def setup_function(function):
    global selfd, selfd1, selfd2, selfd3, selfd4

    selfd=Data(path.join(path.dirname(__file__),"CoreTest.dat"),setas="xy")
    selfd2=Data(path.join(__home__,"..","sample-data","TDI_Format_RT.txt"))
    selfd3=Data(path.join(__home__,"..","sample-data","New-XRay-Data.dql"))
    selfd4=Data(path.join(__home__,"..","sample-data","Cu_resistivity_vs_T.txt"))

def test_repr():
    global selfd, selfd1, selfd2, selfd3, selfd4
    header="""==============  ======  ======
TDI Format 1.5  X-Data  Y-Data
    index       0 (x)   1 (y)
==============  ======  ======"""
    repr_header="""=============================  ========  ========
TDI Format 1.5                   X-Data    Y-Data
index                             0 (x)     1 (y)
=============================  ========  ========"""
    assert selfd.header,header=="Header output changed."
    assert "\n".join(repr(selfd).split("\n")[:4])==repr_header,"Representation gave unexpcted header."
    assert len(repr(selfd2).split("\n"))==261,"Long file representation failed"
    assert len(selfd2._repr_html_().split("\n"))==263,"Long file representation failed"

def test_base_class():
    global selfd, selfd1, selfd2, selfd3, selfd4
    m=metadataObject()
    m.update(selfd2.metadata)
    assert isinstance(m,MutableMapping),"metadataObject is not a MutableMapping"
    assert len(m)==len(selfd2.metadata), "metadataObject length failure."
    assert len(m[".*"])==len(m),"Failure to index by regexp."
    for k in selfd2.metadata:
        assert m[k]==selfd2.metadata[k],f"Failure to have equal keys for {k}"
        m[k]=k
        assert m[k]==k,f"Failure to set keys for {k}"
        del m[k]
    assert len(m)==0,"Failed to delete from metadataObject."

def test_constructor():
    """Constructor Tests"""
    global selfd, selfd1, selfd2, selfd3, selfd4
    d=Data()
    assert d.shape==(1,0),"Bare constructor failed"
    d=Data(selfd)
    assert np.all(d.data==selfd.data),"Constructor from DataFile failed"
    d=Data([np.ones(100),np.zeros(100)])
    assert d.shape==(100,2),"Constructor from iterable list of nd array failed"
    d=Data([np.ones(100),np.zeros(100)],["X","Y"])
    assert d.column_headers==["X","Y"],f"Failed to set column headers in constructor: {d.column_headers}"
    c=np.zeros(100)
    d=Data({"X-Data":c,"Y-Data":c,"Z-Data":c})
    assert d.shape==(100,3),"Construction from dictionary of columns failed."
    d=selfd.clone
    df=d.to_pandas()
    e=Data(df)
    assert d==e,"Roundtripping through Pandas DataFrame failed."
    d=selfd.clone
    e=Data(d.dict_records)
    e.metadata=d.metadata
    assert d==e,"Copy from dict records failed."
    d=Data(pd.DataFrame(np.zeros((100,3))))
    assert d.column_headers==['Column 0:0', 'Column 1:1', 'Column 2:2']
    d=Data(np.ones(100),np.zeros(100),np.zeros(100),selfd.filename)
    assert d==selfd,"Multiple argument constructor dind't find it's filename"
    with pytest.raises(TypeError):
        e=d(setas=34)
    with pytest.raises(TypeError):
        e=d(column_headers=34)

    with pytest.raises(AttributeError):
        d=Data(bad_attr=False)
    with pytest.raises(TypeError):
        d=Data(column_headers=43)
    with pytest.raises(TypeError):
        d=Data(object)

def test_column():
    global selfd, selfd1, selfd2, selfd3, selfd4
    for i,c in enumerate(selfd.column_headers):
        # Column function checks
        assert all(selfd.data[:,i]==selfd.column(i)),f"Failed to Access column {i} by index"
        assert all(selfd.data[:,i]==selfd.column(c)),f"Failed to Access column {i} by name"
        assert all(selfd.data[:,i]==selfd.column(c[0:3])),f"Failed to Access column {i} by partial name"

    # Check that access by list of strings returns multpiple columns
    assert all(selfd.column(selfd.column_headers)==selfd.data),"Failed to access all columns by list of string indices"
    assert all(selfd.column([0,selfd.column_headers[1]])==selfd.data),"Failed to access all columns by mixed list of indices"

    # Check regular expression column access
    assert all(selfd.column(re.compile(r"^X\-"))[:,0]==selfd.column(0)),"Failed to access column by regular expression"
    # Check attribute column access
    assert all(selfd.X==selfd.column(0)),"Failed to access column by attribute name"

def test_deltions():
    global selfd, selfd1, selfd2, selfd3, selfd4
    ch=["{}-Data".format(chr(x)) for x in range(65,91)]
    data=np.zeros((100,26))
    metadata=OrderedDict([("Key 1",True),("Key 2",12),("Key 3","Hellow world")])
    selfdd=Data(metadata)
    selfdd.data=data
    selfdd.column_headers=ch
    selfdd.setas="3.x3.y3.z"
    selfrepr_string="""===========================  ========  =======  ========  =======  ========  ========
TDI Format 1.5                 D-Data   ....      H-Data   ....      Y-Data    Z-Data
index                           3 (x)              7 (y)                 24        25
===========================  ========  =======  ========  =======  ========  ========
Key 1{Boolean}= True                0                  0                  0         0
Key 2{I32}= 12                      0  ...             0  ...             0         0
Key 3{String}= Hellow world         0  ...             0  ...             0         0
Stoner.class{String}= Data          0  ...             0  ...             0         0
...                                 0  ...             0  ...             0         0"""
    assert "\n".join(repr(selfdd).split("\n")[:9])==selfrepr_string,"Representation with interesting columns failed."
    del selfdd["Key 1"]
    assert len(selfdd.metadata)==3,"Deletion of metadata failed."
    del selfdd[20:30]
    assert selfdd.shape==(90,26),"Deleting rows directly failed."
    selfdd.del_column("Q")
    assert selfdd.shape==(90,25),"Deleting rows directly failed."
    selfdd%=3
    assert selfdd.shape==(90,24),"Deleting rows directly failed."
    selfdd.setas="x..y..z"
    selfdd.del_column(selfdd.setas.not_set)
    assert selfdd.shape==(90,3),"Deleting rows directly failed."
    selfdd.mask[50,1]=True
    selfdd.del_column()
    assert selfdd.shape==(90,2),"Deletion of masked rows failed."
    selfd5=Data(np.ones((10,10)))
    selfd5.column_headers=["A"]*5+["B"]*5
    selfd5.del_column("A",duplicates=True)
    assert selfd5.shape==(10,6),"Failed to delete columns with duplicates True and col specified."
    selfd5=Data(np.ones((10,10)))
    selfd5.column_headers=list("ABCDEFGHIJ")
    selfd5.setas="..x..y"
    selfd5.del_column(True)
    assert selfd5.column_headers==["C","F"],"Failed to delete columns with col=True"

def test_indexing():
    global selfd, selfd1, selfd2, selfd3, selfd4
    #Check all the indexing possibilities
    data=np.array(selfd.data)
    colname=selfd.column_headers[0]
    assert all(selfd.column(colname)==selfd[:,0]),"Failed direct indexing versus column method"
    assert all(selfd[:,0]==data[:,0]),"Failed direct idnexing versusus direct array index"
    assert all(selfd[:,[0,1]]==data),"Failed direct list indexing"
    assert all(selfd[::2,:]==data[::2]),"Failed slice indexing rows"
    assert all(selfd[colname]==data[:,0]),"Failed direct indexing by column name"
    assert all(selfd[:,colname]==data[:,0]),"Failed fallback indexing by column name"
    assert selfd[25,1]==645.0,"Failed direct single cell index"
    assert selfd[25,"Y-Data"]==645.0,"Failoed single cell index direct"
    assert selfd["Y-Data",25]==645.0,"Failoed single cell fallback index order"
    selfd["X-Dat"]=[11,12,13,14,15]
    assert selfd["X-Dat",2]==13,"Failed indexing of metadata lists with tuple"
    assert selfd["X-Dat"][2]==13,"Failed indexing of metadata lists with double indices"
    d=Data(np.ones((10,10)))
    d[0,0]=5 #Index by tuple into data
    d["Column_1",0]=6 # Index by column name, row into data
    d[0,"Column_2"]=7 #Index by row, column name into data
    d["Column_3"]=[1,2,3,4] # Create a metadata
    d["Column_3",2]=2 # Index existing metadata via tuple
    d.metadata[0,5]=10
    d[0,5]=12 # Even if tuple, index metadata if already existing.
    assert np.all(d[0]==np.array([5,6,7,1,1,1,1,1,1,1])),f"setitem on Data to index into Data.data failed.\n{d[0]}"
    assert d.metadata["Column_3"]==[1,2,2,4],"Tuple indexing into metadata Failed."
    assert d.metadata[0,5]==12,"Indexing of pre-existing metadta keys rather than Data./data failed."
    assert d.metadata[1]==[1, 2, 2, 4],"Indexing metadata by integer failed."

def test_len():
    # Check that length of the column is the same as length of the data
    global selfd, selfd1, selfd2, selfd3, selfd4
    assert len(Data())==0,"Empty DataFile not length zero"
    assert len(selfd.column(0))==len(selfd),"Column 0 length not equal to DataFile length"
    assert len(selfd)==selfd.data.shape[0],"DataFile length not equal to data.shape[0]"
    # Check that self.column_headers returns the right length
    assert len(selfd.column_headers)==selfd.data.shape[1],"Length of column_headers not equal to data.shape[1]"

def test_attributes():
    """Test various atribute accesses,"""
    global selfd, selfd1, selfd2, selfd3, selfd4
    header='==============  ======  ======\nTDI Format 1.5  X-Data  Y-Data\n    index       0 (x)   1 (y)\n==============  ======  ======'
    assert selfd.shape==(100,2),"shape attribute not correct."
    assert selfd.dims==2,"dims attribute not correct."
    assert selfd.header==header,"Shorter header not as expected"
    assert all(selfd.T==selfd.data.T),"Transpose attribute not right"
    assert all(selfd.x==selfd.column(0)),"x attribute quick access not right."
    assert all(selfd.y==selfd.column(1)),"y attribute not right."
    assert all(selfd.q==np.arctan2(selfd.data[:,0],selfd.data[:,1])),"Calculated theta attribute not right."
    assert all(sqrt(selfd[:,0]**2+selfd[:,1]**2)==selfd.r),"Calculated r attribute not right."
    assert selfd2.records._[5]["Column 2"]==selfd2.data[5,2],"Records and as array attributes problem"
    d=Data(np.ones((10,2)),["A","A"])
    assert d.records.dtype==np.dtype([('A_1', '<f8'), ('A', '<f8')]),".records with identical column headers handling failed"
    e=selfd.clone
    e.data=np.array([1,1,1])
    e.T=selfd.T
    e.column_headers=selfd.column_headers
    assert e==selfd,"Writingto transposed data failed"
    assert selfd.xcol == 0
    assert selfd.ycol == [1]
    with pytest.raises(AttributeError):
        selfd.zcol == None
    selfd.setas="xz"
    assert selfd.zcol == [1]
    selfd.x=np.ones((100,1))
    assert all(selfd.x==np.ones(100))
    selfd.x=np.ones((1,100))
    assert all(selfd.x==np.ones(100))
    with pytest.raises(RuntimeError):
        selfd.x=np.ones((2,2))


def test_mask():
    global selfd, selfd1, selfd2, selfd3, selfd4
    selflittle=Data()
    p=np.linspace(0,np.pi,91)
    q=np.linspace(0,2*np.pi,91)
    r=np.cos(p)
    x=r*np.sin(q)
    y=r*np.cos(q)
    selflittle.data=np.column_stack((x,y,r))
    selflittle.setas="xyz"
    selflittle.mask=mask_func
    selflittle.mask[:,2]=True
    assert len(repr(selflittle).split("\n"))==len(selflittle)+5,"Non shorterned representation failed."
    selflittle.mask = lambda r:np.abs(r)>0.5
    selflittle.del_rows()
    assert selflittle.shape==(30,3)


def test_iterators():
    global selfd, selfd1, selfd2, selfd3, selfd4
    for i,c in enumerate(selfd.columns()):
        assert all(selfd.column(i)==c),"Iterating over DataFile.columns not the same as direct indexing column"
    for j,r in enumerate(selfd.rows()):
        assert all(selfd[j]==r),"Iteratinf over DataFile.rows not the same as indexed access"
    for k,r in enumerate(selfd):
        pass
    assert j==k,"Iterating over DataFile not the same as DataFile.rows"
    assert selfd.data.shape==(j+1,i+1),"Iterating over rows and columns not the same as data.shape"

def test_metadata():
    global selfd, selfd1, selfd2, selfd3, selfd4
    selfd["Test"]="This is a test"
    selfd["Int"]=1
    selfd["Float"]=1.0
    keys,values=zip(*selfd.items())
    assert tuple(selfd.keys())==keys,"Keys from items() not equal to keys from keys()"
    assert tuple(selfd.values())==values,"Values from items() not equal to values from values()"
    assert selfd["Int"]==1
    assert selfd["Float"]==1.0
    assert selfd["Test"]==selfd.metadata["Test"]
    assert selfd.metadata._typehints["Int"]=="I32"
    assert len(selfd.dir())==6,f"Failed meta data directory listing ({selfd.dir()})"
    assert len(selfd3["Temperature"])==7,"Regular experssion metadata search failed"

def test_dir():
    global selfd, selfd1, selfd2, selfd3, selfd4
    assert selfd.dir("S")==["Stoner.class"],f"Dir method failed: dir was {selfd.dir()}"
    bad_keys=set(['__metaclass__', 'iteritems', 'iterkeys', 'itervalues','__ge__', '__gt__', '__init_subclass__',
                  '__le__', '__lt__', '__reversed__', '__slots__',"_abc_negative_cache","_abc_registry",
                  "_abc_negative_cache_version","_abc_cache","_abc_impl"])
    attrs=set(dir(selfd))-bad_keys
    assert len(attrs)==238,"DataFile.__dir__ failed."
    selfd.setas.clear()
    attrs=set(dir(selfd))-bad_keys
    assert len(attrs)==236,"DataFile.__dir__ failed."

def test_filter():
    global selfd, selfd1, selfd2, selfd3, selfd4
    selfd._push_mask()
    ix=np.argmax(selfd.x)
    selfd.filter(lambda r:r.x<=50)
    assert np.max(selfd.x)==50,"Failure of filter method to set mask"
    assert np.isnan(selfd.x[ix]),"Failed to mask maximum value"
    selfd._pop_mask()
    assert selfd2.select(Temp__not__gt=150).shape==(839,3),"Seect method failure."
    assert selfd.select(lambda r:r.x<30)==selfd.select(X__lt=30),"Select  method as callable failed."
    assert selfd.select(__=lambda r:r.x<30)==selfd.select(X__lt=30),"Select  method as callable failed."

def test_operators():
    #Test Column Indexer
    global selfd, selfd1, selfd2, selfd3, selfd4
    assert all(selfd//0==selfd.column(0)),"Failed the // operator with integer"
    assert all(selfd//"X-Data"==selfd.column(0)),"Failed the // operator with string"
    assert all((selfd//re.compile(r"^X\-"))[:,0]==selfd.column(0)),"Failed the // operator with regexp"
    t=[selfd%1,selfd%"Y-Data",(selfd%re.compile(r"Y\-"))]
    for ix,tst in enumerate(["integer","string","regexp"]):
        assert all(t[ix].data==np.atleast_2d(selfd.column(0)).T),f"Failed % operator with {tst} index"
    d=selfd&selfd.x
    assert d.shape[1]==3,"& operator failed."
    assert len(d.setas)==3,"& messed up setas"
    assert len(d.column_headers)==3,"& messed up setas"
    d&=d.x
    assert d.shape[1]==4,"Inplace & operator failed"
    empty=Data()
    empty=empty+np.zeros(10)
    assert empty.shape==(1,10),"Adding to an empty array failed"
    d=selfd+np.array([0,0])
    assert len(d)==len(selfd)+1,"+ operator failed."
    d+=np.array([0,0])
    assert len(d)==len(selfd)+2,"Inplace + operator failed."
    d=d-(-1)
    assert len(d)==len(selfd)+1,f"Delete with integer index failed. {len(d)} vs {len(selfd)+1}"
    d-=-1
    assert len(d)==len(selfd),f"Inplace delete with integer index failed. {len(d)} vs {len(selfd)}"
    d-=slice(0,-1,2)
    assert len(d)==len(selfd)/2,f"Inplace delete with slice index failed. {len(d)} vs {len(selfd)/2}"
    d=d%1
    assert d.shape[1]==1,"Column division failed"
    assert len(d.setas)==d.shape[1],"Column removal messed up setas"
    assert len(d.column_headers)==d.shape[1],"Column removal messed up column headers"
    e=selfd.clone
    f=selfd2.clone
    g=e+f
    assert g.shape==(1776,2),"Add of 2 column and 3 column failed"
    g=f+e
    assert g.shape==(1776,3),"Add of 3 column and 2 column failed."
    g=e&f
    h=f&e
    assert g.shape==h.shape,"Anding unequal column lengths faile!"
    e=~selfd
    assert e.setas[0]=="y","failed to invert setas columns"
    f.setas="xyz"
    f=selfd2.clone
    f+={"Res":0.3,"Temp":1.2,"New_Column":25}
    f1,f2=f.shape
    assert f1==selfd2.shape[0]+1,"Failed to add a row by providing a dictionary"
    assert f2==selfd2.shape[1]+1,"Failed to add an extra columns by adding a dictionary with a new key"
    assert np.isnan(f[-1,2]) and np.isnan(f[0,-1]),"Unset values when adding a dictionary not NaN"
    d=Data()
    d+=np.ones(5)
    d+=np.zeros(5)
    assert np.all(d//1==np.array([1,0])),"Adding 1D arrays should add new rows"
    with pytest.raises(TypeError):
        d+=True
    with open(selfd.filename,"r") as data:
        lines=data.readlines()
    e=Data()<<lines
    e.setas=selfd.setas
    e["Loaded as"]=selfd["Loaded as"]
    assert e==selfd,"Failed to read and create from << operator"
    d=Data(np.array([[1,2,3,4]]),setas="xdyz")
    assert (~d).setas==["y","e","z","x"],"Inverting multi-axis array failed"

def test_properties():
    global selfd, selfd1, selfd2, selfd3, selfd4
    selflittle=Data()
    p=np.linspace(0,np.pi,91)
    q=np.linspace(0,2*np.pi,91)
    r=np.cos(p)
    x=r*np.sin(q)
    y=r*np.cos(q)
    selflittle.data=np.column_stack((x,y,r))
    selflittle.setas="xyz"
    q_ang=np.round(selflittle.q/np.pi,decimals=2)
    p_ang=np.round(selflittle.p/np.pi,decimals=2)
    assert np.max(q_ang)==1.0,"Data.q failure"
    assert np.max(p_ang)==0.5,"Data.p failure"
    assert np.min(p_ang)==-0.5,"Data.p failure"

def test_methods():
    global selfd, selfd1, selfd2, selfd3, selfd4
    d=selfd.clone
    d&=np.where(d.x<50,1.0,0.0)
    d.rename(2,"Z-Data")
    d.setas="xyz"
    assert all(d.unique(2)==np.array([0,1])),f"Unique values failed: {d.unique(2)}"
    d=selfd.clone
    d.insert_rows(10,np.zeros((2,2)))
    assert len(d)==102,"Failed to inert extra rows"
    assert d[9,0]==10 and d[10,0]==0 and d[12,0]==11, "Failed to insert rows properly."
    d=selfd.clone
    d.add_column(np.ones(len(d)),replace=False,header="added")
    assert d.shape[1]==selfd.shape[1]+1,"Adding a column with replace=False did not add a column."
    assert np.all(d.data[:,-1]==np.ones(len(d))),"Didn't add the new column to the end of the data."
    assert len(d.column_headers)==len(selfd.column_headers)+1,"Column headers isn't bigger by one"
    assert d.column_headers==selfd.column_headers+["added",],"Column header not added correctly"
    d=selfd.clone
    d.add_column(selfd.x)
    assert np.all(d.x==d[:,2]),"Adding a column as a DataArray with column headers didn't work"
    assert d.x.column_headers[0]==d.column_headers[2],"Adding a column as a DataArray with column headers didn't work"
    e=d.clone
    d.swap_column([(0,1),(0,2)])
    assert d.column_headers==[e.column_headers[x] for x in [2,0,1]],f"Swap column test failed: {d.column_headers}"
    e=selfd(setas="yx")
    assert e.shape==selfd.shape and e.setas[0]=="y","Failed on a DataFile.__call__ test"
    spl=len(repr(selfd).split("\n"))
    assert spl,105==f"Failed to do repr function got {spl} lines"
    e=selfd.clone
    e=e.add_column(e.x,header=e.column_headers[0])
    e.del_column(duplicates=True)
    assert e.shape==(100,2),"Deleting duplicate columns failed"
    e=selfd2.clone
    e.reorder_columns([2,0,1])
    assert e.column_headers==[selfd2.column_headers[x] for x in [2,0,1]],"Failed to reorder columns: {}".format(e.column_headers)
    d=selfd.clone
    d.del_rows(0,10.0)
    assert d.shape==(99,2),f"Del Rows with value and column failed - actual shape {d.shape}"
    d=selfd.clone
    d.del_rows(0,(10.0,20.0))
    assert d.shape==(89,2),f"Del Rows with tuple and column failed - actual shape {d.shape}"
    d=selfd.clone
    d.mask[::2,0]=True
    d.del_rows()
    assert d.shape==(50,2),f"Del Rows with mask set - actual shape {d.shape}"
    d=selfd.clone
    d[::2,1]=nan
    d.del_nan(1)
    assert d.shape==(50,2),f"del_nan with explicit column set failed shape was {d.shape}"
    d=selfd.clone
    d[::2,1]=nan
    d.del_nan(0)
    assert d.shape==(100,2),f"del_nan with explicit column set and not nans failed shape was {d.shape}"
    d=selfd.clone
    d[::2,1]=nan
    d.setas=".y"
    d.del_nan()
    assert d.shape==(50,2),f"del_nan with columns from setas failed shape was {d.shape}"
    d=selfd.clone
    d2=selfd.clone
    d2.data=d2.data[::-1,:]
    assert d.sort(reverse=True)==d2,"Sorting revserse not the same as manually reversed data."

def test_metadata_save():
    global selfd, selfd1, selfd2, selfd3, selfd4
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
    t2 = selfd4.clone  #check that python tdi save is the same as labview tdi save
    t2.save(path.join(local, "mixedmetatest2.txt"))
    t2l = Data(path.join(local, "mixedmetatest2.txt"))
    for orig, load in [(t,tl), (t2, t2l)]:
        for k in ['Loaded as', 'TDI Format']:
            orig[k]=load[k]
        assert np.allclose(orig.data, load.data)
        assert orig.column_headers==load.column_headers
        selfres=load.metadata^orig.metadata
        assert load.metadata==orig.metadata,"Metadata not the same on round tripping to disc"
    os.remove(path.join(local, "mixedmetatest.txt")) #clear up
    os.remove(path.join(local, "mixedmetatest2.txt"))

if __name__=="__main__": # Run some tests manually to allow debugging
    pytest.main(["--pdb",__file__])