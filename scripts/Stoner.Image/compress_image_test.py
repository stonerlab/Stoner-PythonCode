# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:21:36 2016

@author: phyrct

Proof that the compression is lossless
"""

from PIL import Image
import os
import numpy as np

os.chdir(r"C:\Users\phyrct\KermitData")
i = Image.open("test.png")
i.save("test_PILcompress.png", optimize=True)

j = Image.open("test_PILcompress.png")

p = np.asarray(i)
q = np.asarray(j)

print(np.allclose(p, q))
