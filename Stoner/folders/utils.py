# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 17:02:06 2018

@author: phygbu
"""
__all__=["pathsplit","pathjoin"]
import os.path as path

def pathsplit(pth):
    """Split pth into a sequence of individual parts with path.split."""
    dpart,fpart=path.split(pth)
    if dpart=="":
        return [fpart,]
    else:
        rest=pathsplit(dpart)
        rest.append(fpart)
        return rest

def pathjoin(*args):
    """Join a path like path.join, but then replace the path separator with a standard /."""
    tmp=path.join(*args)
    return tmp.replace(path.sep,"/")
