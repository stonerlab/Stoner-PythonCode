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
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

pth=path.dirname(__file__)
pth=path.realpath(path.join(pth,"../../../"))
sys.path.insert(0,pth)
from Stoner import Data,__home__,Options
from Stoner.compat import python_v3
from Stoner.Core import typeHintedDict
from Stoner.plot.core import hsl2rgb
from Stoner.Image import ImageFile

from Stoner.plot.formats import DefaultPlotStyle

class Plottest(unittest.TestCase):

    """Path to sample Data File"""
    datadir=path.join(pth,"sample-data")

    def setUp(self):
        self.d=Data(path.join(__home__,"..","sample-data","New-XRay-Data.dql"))

    def test_set_no_figs(self):
        self.assertTrue(Options.no_figs,"Default setting for no_figs option is incorrect.")
        Options.no_figs=True
        e=self.d.clone
        ret=e.plot()
        self.assertTrue(ret is None,"Output of Data.plot() was not None when no_figs is True  and showfig is not set({})".format(type(ret)))
        Options.no_figs=False
        e.showfig=False
        ret=e.plot()
        self.assertTrue(isinstance(ret,Data),"Return value of Data.plot() was not self when Data.showfig=False ({})".format(type(ret)))
        e.showfig=True
        ret=e.plot()
        self.assertTrue(isinstance(ret,Figure),"Return value of Data.plot() was not Figure when Data.showfig=False({})".format(type(ret)))
        e.showfig=None
        ret=e.plot()
        self.assertTrue(ret is None,"Output of Data.plot() was not None when Data.showfig is None ({})".format(type(ret)))
        Options.no_figs=True
        self.assertTrue(Options.no_figs,"set_option no_figs failed.")
        self.d=Data(path.join(__home__,"..","sample-data","New-XRay-Data.dql"))
        self.d.showfig=False
        ret=self.d.plot()
        self.assertTrue(ret is None,"Output of Data.plot() was not None when no_figs is True ({})".format(type(ret)))
        Options.no_figs=True
        plt.close("all")

    def test_template_settings(self):
        template=DefaultPlotStyle(font__weight="bold")
        self.assertEqual(template["font.weight"],"bold","Setting ytemplate parameter in init failed.")
        template(font__weight="normal")
        self.assertEqual(template["font.weight"],"normal","Setting ytemplate parameter in call failed.")
        template["font.weight"]="bold"
        self.assertEqual(template["font.weight"],"bold","Setting ytemplate parameter in setitem failed.")
        del template["font.weight"]
        self.assertEqual(template["font.weight"],"normal","Resettting template parameter failed.")
        keys=sorted([x for x in template])
        self.assertEqual(sorted(template.keys()),keys,"template.keys() and template.iter() disagree.")
        attrs=[x for x in dir(template) if template._allowed_attr(x)]
        length=len(dict(plt.rcParams))+len(attrs)
        self.assertEqual(len(template),length,"templa length wrong.")

    def test_plot_magic(self):
        self.d.figure()
        dpi=self.d.fig_dpi
        self.d.fig_dpi=dpi*2
        self.assertEqual(self.d.fig.dpi,dpi*2,"Failed to get/set attributes on current figure")
        vis=self.d.fig_visible
        self.d.fig_visible=not vis
        self.assertFalse(self.d.fig_visible,"Setting/Getting figure.visible failed")
        plt.close("all")
        plt.figure()
        fn=plt.get_fignums()[0]
        self.d.fig=fn
        self.d.plot()
        self.assertEqual(len(plt.get_fignums()),1,"Setting Data.fig by integer failed.")
        plt.close("all")
        self.d.plot(plotter=plt.semilogy)
        self.assertEqual(self.d.ax_lines[0].get_c(),"k","Auto formatting of plot failed")
        self.d.plot(figure=False)
        self.d.plot(figure=1)
        self.assertEqual(len(plt.get_fignums()),2,"Plotting setting figure failed")
        self.assertEqual(len(self.d.ax_lines),2,"Plotting setting figure failed")
        self.d.figure(2)
        self.d.plot()
        self.d.plot(figure=True)
        self.assertEqual(len(plt.get_fignums()),2,"Plotting setting figure failed")
        self.assertEqual(len(self.d.ax_lines),3,"Plotting setting figure failed")
        plt.close("all")
        d=Data()

    def test_extra_plots(self):
        if not python_v3:
            return
        x=np.random.uniform(-np.pi,np.pi,size=5001)
        y=np.random.uniform(-np.pi,np.pi,size=5001)
        z=(np.cos(4*np.sqrt(x**2+y**2))*np.exp(-np.sqrt(x**2+y**2)/3.0))**2
        self.d2=Data(x,y,z,column_headers=["X","Y","Z"],setas="xyz")
        self.d2.contour_xyz(projection="2d")#
        self.assertEqual(len(plt.get_fignums()),1,"Setting Data.fig by integer failed.")
        plt.close("all")
        X,Y,Z = self.d2.griddata(xlim=(-np.pi,np.pi), ylim=(-np.pi,np.pi))
        plt.imshow(Z)
        self.assertEqual(len(plt.get_fignums()),1,"Setting Data.fig by integer failed.")
        plt.imshow(Z)
        plt.close("all")
        x,y=np.meshgrid(np.linspace(-np.pi,np.pi,10),np.linspace(-np.pi,np.pi,10))
        z=np.zeros_like(x)
        w=np.cos(np.sqrt(x**2+y**2))
        q=np.arctan2(x,y)
        u=np.abs(w)*np.cos(q)
        v=np.abs(w)*np.sin(q)
        self.d3=Data(x.ravel(),y.ravel(),z.ravel(),u.ravel(),v.ravel(),w.ravel(),setas="xyzuvw")
        self.d3.plot()
        self.assertEqual(len(plt.get_fignums()),1,"Setting Data.fig by integer failed.")
        plt.close("all")
        # i=ImageFile(path.join(__home__,"..","sample-data","Kermit.png"))
        # self.d3=Data(i)
        # self.d3.data=i.data
        # self.d3.plot_matrix()



    def test_misc_funcs(self):
        self.assertTrue(np.all(hsl2rgb(0.5,0.5,0.5)==np.array([[ 63, 191, 191]])))
        self.d.load(self.d.filename,Q=True)
        self.d.plot()
        self.d.x2()
        self.d.setas=".yx"
        self.d.plot()
        self.d.tight_layout()
        self.assertEqual(len(self.d.fig_axes),2,"Creating a second X axis failed")
        plt.close("all")
        for i in range(4):
            self.d.subplot(2,2,i+1)
            self.d.plot()
        self.assertEqual(len(self.d.fig_axes),4,"Creating subplots failed")
        self.d.close()







if __name__=="__main__": # Run some tests manually to allow debugging
    test=Plottest("test_extra_plots")
    test.setUp()
    test.test_extra_plots()
    #unittest.main()
