# -*- coding: utf-8 -*-
"""
Stoner class Python2/3 compatibility module

Created on Tue Jan 14 19:53:11 2014

@author: phygbu
"""
from __future__ import print_function, absolute_import, division, unicode_literals
from sys import version_info as __vi__
from re import _pattern_type

# Nasty hacks to sort out some naming conventions
if __vi__[0] == 2:
    range = xrange
    string_types = (str, unicode)
    int_types=(int,long)
    python_v3 = False

    def str2bytes(s):
        return str(s)

    def bytes2str(b):
        return str(b)

    def get_filedialog(what="file", **opts):
        """Wrapper around Tk file dialog to mange creating file dialogs in a cross platform way.

        Args:
            what (str): What sort of a dialog to create - options are 'file','directory','save','files'
            **opts (dict): Arguments to pass through to the underlying dialog function.

        Returns:
            A file name or directory or list of files. """
        from Tkinter import Tk
        import tkFileDialog as filedialog
        r = Tk()
        r.withdraw()
        funcs = {
            "file": filedialog.askopenfilename,
            "directory": filedialog.askdirectory,
            "files": filedialog.askopenfilenames,
            "save": filedialog.asksaveasfilename
        }
        if what not in funcs:
            raise RuntimeError("Unable to recognise required file dialog type:{}".format(what))
        else:
            return funcs[what](**opts)

elif __vi__[0] == 3:
    string_types = (str, )
    int_types=(int,)
    python_v3 = True

    def str2bytes(s):
        return bytes(str(s), "utf-8")

    def bytes2str(b):
        if isinstance(b, bytes):
            return b.decode("utf-8", "ignore")
        else:
            return b

    def get_filedialog(what="file", **opts):
        """Wrapper around Tk file dialog to mange creating file dialogs in a cross platform way.

        Args:
            what (str): What sort of a dialog to create - options are 'file','directory','save','files'
            **opts (dict): Arguments to pass through to the underlying dialog function.

        Returns:
            A file name or directory or list of files. """
        from tkinter import Tk, filedialog
        r = Tk()
        r.withdraw()
        funcs = {
            "file": filedialog.askopenfilename,
            "directory": filedialog.askdirectory,
            "files": filedialog.askopenfilenames,
            "save": filedialog.asksaveasfilename
        }
        if what not in funcs:
            raise RuntimeError("Unable to recognise required file dialog type:{}".format(what))
        else:
            return funcs[what](**opts)

index_types = string_types + (int, _pattern_type)
