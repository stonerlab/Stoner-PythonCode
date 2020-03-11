# -*- coding: utf-8 -*-
"""Proof that the compression is lossless."""
# pylint: disable=invalid-name
import os
import numpy as np
from PIL import Image

os.chdir(r"C:\Users\phyrct\KermitData")
i = Image.open("test.png")
i.save("test_PILcompress.png", optimize=True)

j = Image.open("test_PILcompress.png")

p = np.asarray(i)
q = np.asarray(j)

print(np.allclose(p, q))
