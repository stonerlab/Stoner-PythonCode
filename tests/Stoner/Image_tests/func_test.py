# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:21:48 2016

@author: phyrct
"""
from kermit import KerrArray
import numpy as np

a=KerrArray('coretestdata/im2_noannotations.png')
a1=KerrArray('coretestdata/im1_annotated.png')



b=a.translate((2.5,3))
c=b.correct_drift(ref=a)
print a.metadata
print a1.metadata
print all([k in a.metadata.keys() for k in a1.metadata.keys()])

