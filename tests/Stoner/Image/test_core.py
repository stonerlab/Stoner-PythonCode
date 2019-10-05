# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:09:04 2016

@author: phyrct
"""

from Stoner.Image import ImageArray, KerrArray, ImageFile
from Stoner.Core import typeHintedDict
from Stoner import Data,__home__
import numpy as np
import unittest
import sys
from sys import version_info
from os import path
import tempfile
import os
import shutil
from Stoner.compat import python_v3
from PIL import Image

import warnings

#data arrays for testing - some useful small images for tests

thisdir=path.dirname(__file__)

def shares_memory(arr1, arr2):
    """Check if two numpy arrays share memory"""
    #ret = arr1.base is arr2 or arr2.base is arr1
    ret = np.may_share_memory(arr1, arr2)
    return ret

class ImageArrayTest(unittest.TestCase):

    def setUp(self):
        #random array shape 3,4
        self.arr = np.array([[ 0.41674764,  0.66100043,  0.91755303,  0.33796703],
                                  [ 0.06017535,  0.1440342 ,  0.34441777,  0.9915282 ],
                                  [ 0.2984083 ,  0.9167951 ,  0.73820304,  0.7655299 ]])
        self.imarr = ImageArray(np.copy(self.arr)) #ImageArray object
        self.imarrfile = ImageArray(os.path.join(thisdir, 'coretestdata/im1_annotated.png'))
                        #ImageArray object from file

    #####test loading with different datatypes  ####

    def test_load_from_array(self):
        #from array
        self.assertTrue(np.array_equal(self.imarr,self.arr))
        #int type
        imarr = ImageArray(np.arange(12,dtype="int32").reshape(3,4))
        self.assertTrue(imarr.dtype==np.dtype('int32'),"Failed to set correct dtype - actual dtype={}".format(imarr.dtype))

    def test_load_from_ImageArray(self):
        #from ImageArray
        t = ImageArray(self.imarr)
        self.assertTrue(shares_memory(self.imarr, t), 'no overlap on creating ImageArray from ImageArray')

    def test_load_from_png(self):
        subpath = os.path.join("coretestdata","im1_annotated.png")
        fpath = os.path.join(thisdir, subpath)
        anim=ImageArray(fpath)
        self.assertTrue(os.path.normpath(anim.metadata['Loaded from']) == os.path.normpath(fpath))
        cwd = os.getcwd()
        os.chdir(thisdir)
        anim = ImageArray(subpath)
        #check full path is in loaded from metadata
        self.assertTrue(os.path.normpath(anim.metadata['Loaded from']) == os.path.normpath(fpath), 'Full path not in metadata: {}'.format(anim["Loaded from"]))
        os.chdir(cwd)

    def test_load_save_all(self):
        tmpdir=tempfile.mkdtemp()
        pth=path.join(__home__,"..")
        datadir=path.join(pth,"sample-data")
        image=ImageFile(path.join(datadir,"kermit.png"))
        ims={}
        fmts=["uint8","uint16","uint32","float32"]
        modes={"uint8":"L","uint32":"RGBX","float32":"F"}
        for fmt in fmts:
            ims[fmt]=image.clone.convert(fmt)
            ims[fmt].save_tiff(path.join(tmpdir,"kermit-{}.tiff".format(fmt)))
            ims[fmt].save_tiff(path.join(tmpdir,"kermit-forcetype{}.tiff".format(fmt)),forcetype=True)
            ims[fmt].save_npy(path.join(tmpdir,"kermit-{}.npy".format(fmt)))
            if fmt!="uint16":
                im=Image.fromarray((ims[fmt].view(np.ndarray)),modes[fmt])
                im.save(path.join(tmpdir,"kermit-nometadata-{}.tiff".format(fmt)))
            del ims[fmt]["Loaded from"]
        for fmt in fmts:
            iml=ImageFile(path.join(tmpdir,"kermit-{}.tiff".format(fmt)))
            del iml["Loaded from"]
            self.i=ims[fmt]
            self.i2=iml
            self.assertEqual(ims[fmt],iml,"Round tripping tiff with format {} failed".format(fmt))
            iml=ImageFile(path.join(tmpdir,"kermit-{}.npy".format(fmt)))
            del iml["Loaded from"]
            self.assertTrue(np.all(ims[fmt].data==iml.data),"Round tripping npy with format {} failed".format(fmt))
            if fmt!="uint16":
                im=ImageFile(path.join(tmpdir,"kermit-nometadata-{}.tiff".format(fmt)))
                self.assertTrue(np.all(im.data==ims[fmt].data),"Loading from tif without metadata failed for {}".format(fmt))
        shutil.rmtree(tmpdir)
        i8=image.convert("uint8")

    def test_load_from_ImageFile(self):
        #uses the ImageFile.im attribute to set up ImageArray. Memory overlaps
        imfi = ImageFile(self.arr)
        imarr = ImageArray(imfi)
        self.assertTrue(np.array_equal(imarr, imfi.image), 'Initialising from ImageFile failed')
        self.assertTrue(shares_memory(imarr, imfi.image))

    def test_load_from_list(self):
        t=ImageArray([[1,3],[3,2],[4,3]])
        self.assertTrue(np.array_equal(t, np.array([[1,3],[3,2],[4,3]])), 'Initialising from list failed')

    def test_load_1d_data(self):
        t=ImageArray(np.arange(10)/10.0)
        self.assertTrue(len(t.shape)==2) #converts to 2d

    def test_load_no_args(self):
        #Should be a 2d empty array
        t=ImageArray()
        self.assertTrue(len(t.shape)==2)
        self.assertTrue(t.size==0)

    def test_load_bad_data(self):
        def testload(arg):
            ImageArray(arg)
        #dictionary
        self.assertRaises(ValueError, testload, {'a':1})
        #3d numpy array
        self.assertRaises(ValueError, testload, np.arange(27).reshape(3,3,3))
        #bad filename
        self.assertRaises(ValueError, testload, 'sillyfile.xyz')

    def test_load_kwargs(self):
        #metadata keyword arg
        t=ImageArray(self.arr, metadata={'a':5, 'b':7})
        self.assertTrue('a' in t.metadata.keys() and 'b' in t.metadata.keys())
        self.assertTrue(t.metadata['a']==5)
        #asfloat
        t=ImageArray(np.arange(12).reshape(3,4), asfloat=True)
        self.assertTrue(t.dtype == np.float64, 'Initialising asfloat failed')
        self.assertTrue('Loaded from' in t.metadata.keys(), 'Loaded from should always be in metadata')

    #####test attributes ##

    def test_filename(self):
        im = ImageArray(np.linspace(0,1,12).reshape(3,4))
        fpath = os.path.join(thisdir, 'coretestdata/im1_annotated.png')
        self.assertTrue(os.path.normpath(self.imarrfile.filename)\
                        == os.path.normpath(fpath))
        im = ImageArray(np.linspace(0,1,12).reshape(3,4))
        im['Loaded from']
        im.filename
        self.assertTrue(im.filename == '', '{}, {}'.format(self.imarr.shape, im.filename))

    def test_clone(self):
        self.imarr['abc'] = 123 #add some metadata
        self.imarr['nested'] = [1,2,3] #add some nested metadata to check deepcopy
        self.imarr.userxyz = 123 #add a user attribute
        c = self.imarr.clone
        self.assertTrue(isinstance(c,ImageArray), 'Clone not ImageArray')
        self.assertTrue(np.array_equal(c,self.imarr),'Clone not replicating elements')
        self.assertTrue(all([k in c.metadata.keys() for k in \
                    self.imarr.metadata.keys()]), 'Clone not replicating metadata')
        self.assertFalse(shares_memory(c, self.imarr), 'memory overlap on clone') #formal check
        self.imarr['bcd'] = 234
        self.assertTrue('bcd' not in c.metadata.keys(), 'memory overlap for metadata on clone')
        self.imarr['nested'][0] = 2
        self.assertTrue(self.imarr['nested'][0] != c['nested'][0], 'deepcopy not working on metadata')
        self.assertTrue(c.userxyz == 123)
        c.userxyz = 234
        self.assertTrue(c.userxyz != self.imarr.userxyz)

    def test_metadata(self):
        self.assertTrue(isinstance(self.imarr.metadata,typeHintedDict))
        self.imarr['testmeta']='abc'
        self.assertTrue(self.imarr['testmeta']=='abc', 'Couldn\'t change metadata')
        del(self.imarr['testmeta'])
        self.assertTrue('testmeta' not in self.imarr.keys(),'Couldn\'t delete metadata')
        #bad data
        def test(imarr):
            imarr.metadata=(1,2,3)
        self.assertRaises(TypeError, test, self.imarr) #check it won't let you do this

    ### test numpy like creation behaviour #
    def test_user_attributes(self):
        self.imarr.abc = 'new att'
        self.assertTrue(hasattr(self.imarr, 'abc'))
        t = ImageArray(self.imarr)
        self.assertTrue(hasattr(t, 'abc'), 'problem copying new attributes')
        t = self.imarr.view(ImageArray) #check array_finalize copies attribute over
        self.assertTrue(hasattr(t, 'abc'))
        t = self.imarr * np.random.random(self.imarr.shape)
        self.assertTrue(isinstance(t, ImageArray), 'problem with ufuncs')
        self.assertTrue(hasattr(t, 'abc'),"Ufunc lost attribute!")

    #####  test functionality  ##
    def test_save(self):
        testfile=path.join(thisdir,'coretestdata','testsave')
        ext = ['.png', '.npy']
        keys=self.imarr.keys()
        for e in ext:
            self.imarr.save(filename=testfile+e)
            load=ImageArray(testfile+e)
            self.assertTrue(all([k in load.keys() for k in keys]), 'problem saving metadata {} {}'.format(list(load.keys()),e))
            if e=='.npy':
                #tolerance is really poor for png whcih savees in 8bit format
                self.assertTrue(np.allclose(self.imarr, load),
                                'data not the same for extension {}'.format(e))
            os.remove(testfile+e) #tidy up

    def test_savetiff(self):
        testfile=path.join(thisdir,'coretestdata','testsave.tiff')
        #create a few different data types
        testb = ImageArray(np.zeros((4,5), dtype=bool))  #bool
        testb[0,:]=True
        testui = ImageArray(np.arange(20).reshape(4,5)) #int32
        testi = ImageArray(np.copy(testui)-10)
        testf = ImageArray(np.linspace(-1,1,20).reshape(4,5)) #float64
        for im in [testb, testui, testi, testf]:
            im['a']=[1,2,3]
            im['b']='abc' #add some test metadata
            im.filename=testfile
            im.save()
            n = ImageArray(testfile)
            self.n=n
            self.assertTrue(all([n['a'][i]==im['a'][i] for i in range(len(n['a']))]))
            self.assertTrue(n['b']==im['b'])
            self.assertTrue('ImageArray.dtype' in n.metadata.keys()) #check the dtype metdata got added
            self.assertTrue(im.dtype==n.dtype) #check the datatype
            self.im=im
            self.n=n
            self.assertTrue(np.allclose(im, n))  #check the data





    def test_max_box(self):
        s=self.imarr.shape
        self.assertTrue(self.imarr.max_box==(0,s[1],0,s[0]))

    def test_crop(self):
        c=self.imarr.crop((1,3,1,4),copy=True)
        self.assertTrue(np.array_equal(c,self.imarr[1:4,1:3]),'crop didn\'t work')
        self.assertFalse(shares_memory(c,self.imarr), 'crop copy failed')
        c2=self.imarr.crop(1,3,1,4,copy=True)
        self.assertTrue(np.array_equal(c2,c),'crop with seperate arguments didn\'t work')
        c3 = self.imarr.crop(box=(1,3,1,4), copy=False)
        self.assertTrue(np.array_equal(c3,c), 'crop with no arguments failed')
        self.assertTrue(shares_memory(self.imarr, c3), 'crop with no copy failed')

    def test_asint(self):
        ui = self.imarr.asint()
        self.assertTrue(ui.dtype==np.uint16)
        intarr = np.array([[27312, 43319, 60132, 22149],
                           [ 3944,  9439, 22571, 64980],
                           [19556, 60082, 48378, 50169]], dtype=np.uint16)
        self.assertTrue(np.array_equal(ui,intarr))

    def test_other_funcs(self):
        """test imagefuncs add ons. the functions themselves are not checked
        and should include a few examples in the doc strings for testing"""
        self.assertTrue(hasattr(self.imarr,'do_nothing'), 'imagefuncs not being added to dir')
        self.assertTrue(hasattr(self.imarr,'Stoner__Image__imagefuncs__do_nothing'),"Stoner image functions not added with full namke")
        self.assertTrue(hasattr(self.imarr,'img_as_float'), 'skimage funcs not being added to dir')
        im = self.imarr.do_nothing() #see if it can run
        self.assertTrue(np.allclose(im, self.imarr), 'imagefuncs not working')
        self.assertFalse(shares_memory(im, self.imarr), 'imagefunc failed to clone')
        im0 = ImageArray(np.linspace(0,1,12).reshape(3,4))
        im1 = im0.clone * 5
        im2 = im1.rescale_intensity() #test skimage
        self.assertTrue(np.allclose(im2, im0), 'skimage func failed')
        self.assertFalse(shares_memory(im2, im1), 'skimage failed to clone')
        im3 = im1.exposure__rescale_intensity() #test call with module name
        self.assertTrue(np.allclose(im3, im0), 'skimage call with module name failed')


    def test_attrs(self):
        attrs=[x for x in dir(self.imarr) if not x.startswith("_")]
        counts={(2,7):803,(3,5):846}
        expected=counts.get(version_info[0:2],871)
        self.assertEqual(len(attrs),expected,"Length of ImageArray dir failed. {}".format(len(attrs)))


class ImageFileTest(unittest.TestCase):
    def setUp(self):
        self.a = np.linspace(0,5,12).reshape(3,4)
        self.ifi = ImageFile(self.a)
        self.imgFile = ImageFile(os.path.join(thisdir, 'coretestdata/im1_annotated.png'))

    def test_constructors(self):
        self.imgFile = ImageFile(os.path.join(thisdir, 'coretestdata/im1_annotated.png'))
        self.d=Data(self.imgFile)
        self.imgFile2=ImageFile(self.d)
        del self.imgFile2["Stoner.class"]
        del self.imgFile2["x_vector"]
        del self.imgFile2["y_vector"]
        self.assertTrue(np.all(self.imgFile.image==self.imgFile2.image),"Roundtripping constructor through Data failed to dupliocate data.")
        self.assertEqual(self.imgFile.metadata,self.imgFile2.metadata,"Roundtripping constructor through Data failed to duplicate metadata: {} {}".format(
                self.imgFile2.metadata^self.imgFile.metadata, self.imgFile.metadata^self.imgFile2.metadata))

    def test_properties(self):
        self.assertTrue(np.allclose(self.ifi.image, self.a))
        self.ifi[0,1]=10.1
        self.assertTrue(self.ifi.image[0,1]==10.1)

    def test_attrs(self):
        self.assertTrue(self.ifi['Loaded from'] == '')
        self.ifi.abc = 123
        self.assertTrue(self.ifi.image.abc == 123)
        self.assertTrue(self.ifi.abc == 123)

    def test_methods(self):
        b=np.arange(12).reshape(3,4)
        ifi = ImageFile(b)
        ifi.asfloat(normalise=False, clip_negative=False) #convert in place
        self.assertTrue(ifi.image.dtype.kind == 'f')
        self.assertTrue(np.max(ifi) == 11.0)
        ifi.image == ifi.image*5
        ifi.rescale_intensity()
        self.assertTrue(np.allclose(ifi.image, np.linspace(0,1,12).reshape(3,4)))
        ifi.crop(0,3,0,None)
        self.assertTrue(ifi.shape==(3,3)) #check crop is forced to overwrite ifi despite shape change
        datadir=path.join(__home__,"..","sample-data")
        image=ImageFile(path.join(datadir,"kermit.png"))
        i2=image.clone.box(5,_=True)
        self.assertEqual(i2.shape,(469, 349),"Failed to trim box by integer")
        i2=image.clone.box(0.25,_=True)
        self.assertEqual(i2.shape,(269, 269),"Failed to trim box by float")
        i2=image.clone.box([0.1,0.2,0.05,0.1],_=True)
        self.assertEqual(i2.shape,(24, 36),"Failed to trim box by sequence of floats")
        self.assertAlmostEqual(image.aspect,0.7494780793,places=6,msg="Aspect ratio failed" )
        self.assertEqual(image.centre,(239.5, 179.5),"Failed to read image.centre")
        i2=image.CW
        self.assertEqual(i2.shape,(359,479),"Failed to rotate clockwise")
        i3=i2.CCW
        self.assertEqual(i3.shape,(479,359),"Failed to rotate counter-clockwise")
        i2=image.clone
        self.assertAlmostEqual((i2-127).mean(), 39086.4687283,places=2,msg="Subtract integer failed.")
        try:
            i2-"Gobble"
        except TypeError:
            pass
        else:
            self.AssertTrue(False,"Subtraction of string didn't raise not implemented")
        attrs=[x for x in dir(image) if not x.startswith("_")]
        counts={(2,7):803,(3,5):846}
        expected=counts.get(version_info[0:2],871)
        self.assertEqual(len(attrs),expected,"Length of ImageFile dir failed. {}:{}".format(expected,len(attrs)))
        self.assertTrue(image._repr_png_().startswith(b'\x89PNG\r\n'),"Failed to do ImageFile png representation")


    def test_mask(self):
        i=np.ones((200,200),dtype="uint8")*np.linspace(1,200,200).astype("uint8")
        i=ImageFile(i)
        self.i=i
        i.mask.draw.rectangle(100,50,100,100)
        self.assertAlmostEqual(i.mean(), 117.1666666666666,msg="Mean after masked rectangle failed")
        i.mask.invert()
        self.assertAlmostEqual(i.mean(), 50.5,msg="Mean after inverted masked rectangle failed")
        i.mask.clear()
        self.assertAlmostEqual(i.mean(), 100.5,msg="Mean after clearing mask failed")
        i.mask.draw.square(100,50,100)
        self.assertAlmostEqual(i.mean(), 117.1666666666666,msg="Mean after masked rectangle faile")
        i.mask.clear()
        i.mask.draw.annulus(100,50,35,25)
        self.assertAlmostEqual(i.mean(), 102.96850393700,msg="Mean after annular block mask failed")
        i.mask=False
        i.mask.draw.annulus(100,50,25,35)
        self.assertAlmostEqual(i.mean(), 51.0,msg="Mean after annular pass mask failed")
        i.mask[:,:]=False
        self.assertFalse(np.any(i.mask),"Setting Mask by index failed")
        i.mask=-i.mask
        self.assertTrue(np.all(i.mask),"Setting Mask by index failed")
        i.mask=~i.mask
        self.assertFalse(np.any(i.mask),"Setting Mask by index failed")
        i.mask.draw.circle(100,100,20)
        st=repr(i.mask)
        self.assertEqual(st.count("X"),i.mask.sum(),"String representation of mask failed")
        self.assertTrue(np.all(i.mask.image==i.mask._mask),"Failed to access mak data by image attr")
        i.mask=False
        i2=i.clone
        i.mask.draw.rectangle(100,100,100,50,angle=np.pi/2)
        i2.mask.draw.rectangle(100,100,50,100)
        self.i2=i2
        self.assertTrue(np.all(i.mask.image==i2.mask.image),"Drawing rectange with angle failed")
        self.assertTrue(i.mask._repr_png_().startswith(b'\x89PNG\r\n'),"Failed to do mask png representation")
        i=ImageFile(np.zeros((100,100)))
        i2=i.clone
        i2.draw.circle(50,50,25)
        i.mask=i2
        self.assertEqual(i.mask.sum(),i2.sum(),"Setting mask from ImageFile Failed")
        i2.mask=i.mask
        self.assertTrue(np.all(i.mask.image==i2.mask.image),"Failed to set mask by mask proxy")
        i=ImageFile(np.ones((100,100)))
        i.mask.draw.square(50,50,10)
        i.mask.rotate(angle=np.pi/4)
        i.mask.invert()
        i2=ImageFile(np.zeros((100,100)))
        i2.draw.square(50,50,10,angle=np.pi/4)
        self.assertAlmostEqual(i.sum(),i2.sum(),delta=1.5,msg="Check on rotated mask failed !")


    def test_draw(self):
        i=ImageFile(np.zeros((200,200)))
        attrs=[x for x in dir(i.draw) if not x.startswith("_")]
        counts={(2,7):19,(3,5):19}
        expected=counts.get(version_info[0:2],20)
        self.assertEqual(len(attrs),expected,"Directory of DrawProxy failed")

    def test_operators(self):
        i=ImageFile(np.zeros((10,10)))
        i=i+1
        i+=4
        i+=i.clone
        i+=i.clone.data
        self.assertEqual(i.sum(),2000,"Addition operators failed")
        i/=4
        self.assertEqual(i.sum(),500,"Division operators failed")
        i=i/5
        self.assertEqual(i.sum(),100,"Division operators failed")
        i2=i.clone
        i-=0.75
        i2-=0.25
        i3=i2//i
        self.assertEqual(i3.sum(),50,"Division operators failed")
        i=ImageFile(np.zeros((10,5)))
        i=~i
        self.assertEqual(i.shape,(5,10),"Invert to rotate failed")
        i.image=i.image.astype("uint8")
        i=-i
        self.assertEqual(i.sum(),50*255,"Negate operators failed")










if __name__=="__main__": # Run some tests manually to allow debugging
    test=ImageArrayTest("test_filename")
    test.setUp()
    #test.test_load_from_ImageFile()
    #test.test_load_save_all()
#    test.test_save()
    #test.test_savetiff()
#
    test2=ImageFileTest("test_constructors")
    test2.setUp()
    #test2.test_constructors()
    #test2.test_mask()
    unittest.main()



