# -*- coding: utf-8 -*-
"""
Stoner class Python2/3 compatibility module

Created on Tue Jan 14 19:53:11 2014

@author: phygbu
"""
from __future__ import print_function,absolute_import,division,unicode_literals
from sys import version_info as __vi__

# Nasty hacks to sort out some naming conventions
if __vi__[0]==2:
    range=xrange
    string_types=(str,unicode)
    python_v3=False
    def str2bytes(s):
        return str(s)
elif __vi__[0]==3:
    string_types=(str)
    python_v3=True
    def str2bytes(s):
        return bytes(str(s),"utf-8")

