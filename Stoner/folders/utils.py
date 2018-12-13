# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 17:02:06 2018

@author: phygbu
"""
__all__=["pathsplit","pathjoin","scan_dir","discard_earlier","filter_files"]
import os.path as path
import os
import re
from collections import OrderedDict
from Stoner.compat import string_types,_pattern_type
import fnmatch
from numpy import array

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

def scan_dir(root):
    """Helper function to gather a list of files and directories."""
    dirs=[]
    files=[]
    for f in os.listdir(root):
        if path.isdir(path.join(root, f)):
            dirs.append(f)
        elif path.isfile(path.join(root, f)):
            files.append(f)
    return dirs,files

def discard_earlier(files):
    """Helper function to discard files where a similar named file with !#### exists."""
    search=re.compile(r"^(?P<basename>.*)\!(?P<rev>\d+)(?P<ext>\.[^\.]*)$")
    dups=OrderedDict()
    ret=[]
    for f in files:
        match=search.match(f)
        if match:
            fname="{basename}{ext}".format(**match.groupdict())
            rev=int(match.groupdict()["rev"])
            entry=dups.get(fname,[])
            entry.append((rev,f))
            dups[fname]=entry
        else:
            dups[f]=[(-1,f)]
    for f,revs in dups.items():
        rev=sorted(revs)[-1]
        ret.append(rev[1])
    return ret

def filter_files(files,patterns,keep=True):
    """Helper to filter a list of files against include/exclusion patterns.

    Args:
        files (list of str): Filename to filter
        pattens (list of (str,regular expressions): List of patterns to consider

    Keyword Arguments:
        keep (bool): True (default) to keep matching files.

    Returns:
        (list of str): Files that pass the filter
    """
    dels=[]
    for p in patterns: #Remove excluded files
        if isinstance(p,string_types):
            for f in list(fnmatch.filter(files,p)):
                dels.append(files.index(f))
        if isinstance(p,_pattern_type):
            # For reg expts we iterate over all files, but we can't delete matched
            # files as we go as we're iterating over them - so we store the
            # indices and delete them later.
            for f in files:
                if p.search(f):
                    dels.append(files.index(f))
        index=array([not keep^(i in dels) for i in range(len(files))],dtype=bool)
        files=(array(files)[index]).tolist()
    return files