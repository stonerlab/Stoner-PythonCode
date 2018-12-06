#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mixin calsses for :py:class:`Stoner.folders.core.baseFoler`."""
__all__=["DiskBasedFolder","DataMethodsMixin","PlotMethodsMixin"]

import os
import os.path as path
import fnmatch
import string
import unicodedata
import re
from collections import OrderedDict
from functools import partial
from numpy import mean,std,array,append,any,floor,sqrt,ceil
from matplotlib.pyplot import figure,Figure,subplot,tight_layout

from Stoner.compat import string_types,get_filedialog,_pattern_type
from Stoner.tools import isiterable

from Stoner.core.base import metadataObject
from Stoner.Core import DataFile
from Stoner.folders.core import baseFolder

regexp_type=(_pattern_type,)

def _loader(loader,typ,name):
    """Lods and returns an object."""
    return typ(loader(name)),name


class DiskBasedFolder(object):
    """A Mixin class that implmenets reading metadataObjects from disc.

    Attributes:
        type (:py:class:`Stoner.Core.metadataObject`) the type ob object to sotre in the folder (defaults to :py:class:`Stoner.Core.Data`)

        extra_args (dict): Extra arguments to use when instantiatoing the contents of the folder from a file on disk.

        pattern (str or regexp): A filename globbing pattern that matches the contents of the folder. If a regular expression is provided then
            any named groups are used to construct additional metadata entryies from the filename. Default is *.* to match all files with an extension.

        exclude (str or regexp): A filename globbing pattern that matches files to exclude from the folder.  Default is *.tdms_index to exclude all
            tdms index files.

        read_means (bool): IF true, additional metatdata keys are added that return the mean value of each column of the data. This can hep in
            grouping files where one column of data contains a constant value for the experimental state. Default is False

        recursive (bool): Specifies whether to search recurisvely in a whole directory tree. Default is True.

        flatten (bool): Specify where to present subdirectories as spearate groups in the folder (False) or as a single group (True). Default is False.
            The :py:meth:`DiskBasedFolder.flatten` method has the equivalent effect and :py:meth:`DiskBasedFolder.unflatten` reverses it.

        discard_earlier (bool): IF there are several files with the same filename apart from !#### being appended just before the extension, then discard
            all except the one with the largest value of #### when collecting the list of files.

        directory (str): The root directory on disc for the folder - by default this is the current working directory.

        multifile (boo): Whether to select individual files manually that are not (necessarily) in  a common directory structure.

        readlist (bool): Whether to read the directory immediately on creation. Default is True
    """

    _defaults={"type":None,
              "extra_args":dict(),
              "pattern":["*.*"],
              "exclude":["*.tdms_index"],
              "read_means":False,
              "recursive":True,
              "flat":False,
              "prefetch":True,
              "directory":None,
              "multifile":False,
              "pruned":True,
              "readlist":True,
              "discard_earlier":False,
              }


    def __init__(self,*args,**kargs):
        """Additional constructor for DiskbasedFolders"""
        from Stoner import Data
        _=self.defaults #Force the default store to be populated.
        if "directory" in self._default_store and self._default_store["directory"] is None:
            self._default_store["directory"]=os.getcwd()
        if "type" in self._default_store and self._default_store["type"] is None and self._type==metadataObject:
            self._default_store["type"]=Data
        elif self._type!=metadataObject: # Looks like we've already set our type in a subbclass
            self._default_store.pop("type")
        flat=kargs.pop("flat",self._default_store.get("flat",False))
        prefetch=kargs.pop("prefetch",self._default_store.get("prefetch",False))
        super(DiskBasedFolder,self).__init__(*args,**kargs) #initialise before __clone__ is called in getlist
        if self.readlist and len(args)>0 and isinstance(args[0],string_types):
            self.getlist(directory=args[0])
        if flat:
            self.flatten()
        if prefetch:
            self.fetch()
        if self.pruned:
            self.prune()

    @baseFolder.key.getter
    def key(self):
        k=getattr(super(DiskBasedFolder,self),"key",None)
        if k is None:
            self.key=self.directory
            return self._key
        else:
            return k

    def _dialog(self, message="Select Folder",  new_directory=True):
        """Creates a directory dialog box for working with

        Keyword Arguments:
            message (string): Message to display in dialog
            new_directory (bool): True if allowed to create new directory

        Returns:
            A directory to be used for the file operation.
        """
        # Wildcard pattern to be used in file dialogs.
        if not self.multifile:
            mode="directory"
        else:
            mode="files"
        dlg = get_filedialog(what=mode,title=message,mustexist=not new_directory)
        if len(dlg)!=0:
            if not self.multifile:
                self.directory = dlg
                ret=self.directory
            else:
                ret=None
        else:
            self.pattern=[path.basename(name) for name in dlg]
            self.directory = path.commonprefix(dlg)
            ret = self.directory
        return ret

    def _removeDisallowedFilenameChars(self,filename):
        """Utility method to clean characters in filenames

        Args:
            filename (string): filename to cleanse

        Returns:
            A filename with non ASCII characters stripped out
        """
        validFilenameChars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        cleanedFilename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore')
        return ''.join(c for c in cleanedFilename if c in validFilenameChars)

    def _save(self,grp,trail,root=None):
        """Save a group of files to disc by calling the save() method on each file.

        This internal method is called by walk_groups in turn
        called from the public save() method. The trail of group keys is used to create a directory tree.

        Args:
            grp (:py:class:`objectFolder` or :py:calss:`Stoner.metadataObject`): A group or file to save
            trail (list of strings): the trail of paths used to get here
            root (string or None): a replacement root directory

        Returns:
            Saved Path
        """
        trail=[self._removeDisallowedFilenameChars(t) for t in trail]
        grp.filename=self._removeDisallowedFilenameChars(grp.filename)
        if root is None:
            root=self.directory

        pth=path.join(root,*trail)
        os.makesdirs(pth)
        if isinstance(grp,metadataObject) and not isinstance(grp,self.loader):
            grp=self.loader(grp)
        grp.save(path.join(pth,grp.filename))
        return grp.filename

    def __add_core__(self,result,other):
        """Additional logic for the add operator."""
        if isinstance(other,string_types):
            othername=path.join(self.directory,other)
            if path.exists(othername) and othername not in result:
                result.append(othername)
            else:
                raise RuntimeError("{} either does not exist of is already in the folder.".format(othername))
        else:
            return super(DiskBasedFolder,self).__add_core__(result,other)
        return result

    def __sub_core__(self,result,other):
        """Additional logic to check for match to basenames,"""
        if isinstance(other,string_types):
            if other in list(result.basenames) and path.join(result.directory,other) in list(result.ls):
                other=path.join(result.directory,other)
                result.__deleter__(other)
                return result
        return super(DiskBasedFolder,self).__sub_core__(result,other)


    def __lookup__(self,name):
        """Addional logic for the looking up names."""
        if isinstance(name,string_types):
            if list(self.basenames).count(name)==1:
                return self.__names__()[list(self.basenames).index(name)]

        return super(DiskBasedFolder,self).__lookup__(name)

    def __getter__(self,name,instantiate=True):
        """Loads the specified name from a file on disk.

        Parameters:
            name (key type): The canonical mapping key to get the dataObject. By default
                the baseFolder class uses a :py:class:`regexpDict` to store objects in.

        Keyword Arguments:
            instatiate (bool): IF True (default) then always return a :py:class:`Stoner.Core.Data` object. If False,
                the __getter__ method may return a key that can be used by it later to actually get the
                :py:class:`Stoner.Core.Data` object.

        Returns:
            (metadataObject): The metadataObject
        """
        try: #Try the parent methods first
            return super(DiskBasedFolder,self).__getter__(name,instantiate=instantiate)
        except (AttributeError,IndexError,KeyError):
            pass
        if name is not None and not path.exists(name): #Default we assume the name is relative to the root directory
            fname=path.join(self.directory,name)
        else: #Fall back to direct
            fname=name
        tmp= self.type(self.loader(fname,**self.extra_args))
        if not hasattr(tmp,"filename") or not isinstance(tmp.filename,string_types):
            tmp.filename=path.basename(fname)
        tmp=self.on_load_process(tmp)
        tmp=self._update_from_object_attrs(tmp)
        self.__setter__(name,tmp)
        return tmp

    def _scan_dir(self,root):
        """Helper function to gather a list of files and directories."""
        dirs=[]
        files=[]
        for f in os.listdir(root):
            if path.isdir(path.join(root, f)):
                dirs.append(f)
            elif path.isfile(path.join(root, f)):
                files.append(f)
        return dirs,files

    def _discard_earlier(self,files):
        """Helper function to discard files where a similar named file with !#### exists."""
        search=re.compile(r"^(?P<basename>.*)\!(?P<rev>\d+)(?P<ext>\.[^\.]*)$")
        dups=OrderedDict()
        ret=[]
        for f in files:
            match=search.match(f)
            if match:
                fname="{basename}{ext}".format(**match.groupdict())
                rev=int(match.groupdict()["rev"])
                if fname in dups:
                    try:
                        if dups[fname]>rev:
                            continue
                    except TypeError:
                        pass
                dups[fname]=rev
            else:
                if f not in dups:
                    dups[f]=None
        for f,rev in dups.items():
            if rev is None:
                ret.append(f)
            else:
                base,ext=os.path.splitext(f)
                ret.append("{}!{:04d}{}".format(base,rev,ext))
        return ret

    @property
    def basenames(self):
        """Returns a list of just the filename parts of the objectFolder."""
        for x in self.__names__():
            yield path.basename(x)

    @property
    def directory(self): #Just alias directory to root now
        return self.root

    @directory.setter
    def directory(self,value):
        self.root=value

    @property
    def not_loaded(self):
        """Return an array of True/False for whether we've loaded a metadataObject yet."""
        for ix,n in enumerate(self.__names__()):
            if isinstance(self.__getter__(n,instantiate=False),self._type):
                yield n
    @property
    def pattern(self):
        """Provide support for getting the pattern attribute."""
        return self._pattern

    @pattern.setter
    def pattern(self,value):
        """Sets the filename searching pattern[s] for the :py:class:`Stoner.Core.metadataObject`s."""
        if isinstance(value,string_types):
            self._pattern=(value,)
        elif isinstance(value,_pattern_type):
            self._pattern=(value,)
        elif isiterable(value):
            self._pattern=[x for x in value]
        else:
            raise ValueError("pattern should be a string, regular expression or iterable object not a {}".format(type(value)))

    def fetch(self):
        """Preload the contents of the DiskbasedFolder.

        With multiprocess enabled this will parallel load the contents of the folder into memory.
        """
        p,imap=self._get_pool()
        for ix,(f,name) in enumerate(imap(partial(_loader,loader=self.loader,typ=self._type),self.not_loaded)):
            self._folder.__setter__(name,f)

        if p is not None:
            p.close()
            p.join()
        return self

    def getlist(self, recursive=None, directory=None,flatten=None, discard_earlier=None):
        """Scans the current directory, optionally recursively to build a list of filenames

        Keyword Arguments:
            recursive (bool): Do a walk through all the directories for files
            directory (string or False): Either a string path to a new directory or False to open a dialog box or not set in which case existing
                directory is used.
            flatten (bool): After scanning the directory tree, flaten all the subgroupos to make a flat file list. (this is the previous behaviour of
            :py:meth:`objectFolder.getlist()`)

        Returns:
            A copy of the current DataFoder directory with the files stored in the files attribute

        getlist() scans a directory tree finding files that match the pattern. By default it will recurse through the entire
        directory tree finding sub directories and creating groups in the data folder for each sub directory.
        """
        self.__clear__()
        if recursive is None:
            recursive=self.recursive
        if discard_earlier is None:
            discard_earlier=self.discard_earlier
        if flatten is None:
            flatten=getattr(self,"flat",False) #ImageFolders don't have flat because it clashes with a numpy attribute
        if isinstance(directory,  bool) and not directory:
            self._dialog()
        elif isinstance(directory, string_types):
            self.directory=directory
            if self.multifile:
                self._dialog()
        if self.directory is None:
            self.directory=os.getcwd()
        root=self.directory
        dirs,files=self._scan_dir(root)
        if discard_earlier:
            files=self._discard_earlier(files)
        for p in self.exclude: #Remove excluded files
            if isinstance(p,string_types):
                for f in list(fnmatch.filter(files,p)):
                    del files[files.index(f)]
            if isinstance(p,_pattern_type):
                matched=[]
                # For reg expts we iterate over all files, but we can't delete matched
                # files as we go as we're iterating over them - so we store the
                # indices and delete them later.
                for f in files:
                    if p.search(f):
                        matched.append(files.index(f))
                matched.sort(reverse=True)
                for i in matched: # reverse sort the matching indices to safely delete
                    del(files[i])

        for p in self.pattern: # pattern is a list of strings and regeps
            if isinstance(p,string_types):
                for f in fnmatch.filter(files, p):
                    self.append(f)
                    # Now delete the matched file from the list of candidates
                    #This stops us double adding fles that match multiple patterns
                    del(files[files.index(f)])
            if isinstance(p,_pattern_type):
                matched=[]
                # For reg expts we iterate over all files, but we can't delete matched
                # files as we go as we're iterating over them - so we store the
                # indices and delete them later.
                for f in files:
                    if p.search(f):
                        self.__setter__(f,path.join(root,f))
                        matched.append(files.index(f))
                matched.sort(reverse=True)
                for i in matched: # reverse sort the matching indices to safely delete
                    del(files[i])
        if recursive:
            for d in dirs:
                if self.debug: print("Entering directory {}".format(d))
                self.add_group(d)
                self.groups[d].directory=path.join(root,d)
                self.groups[d].getlist(recursive=recursive,flatten=flatten)
        if flatten and not self.is_empty:
            self.flatten()
            #Now collapse out the common path in the names
            self.directory=path.commonprefix(self.__names__())
            if self.directory[-1]!=path.sep:
                self.directory=path.dirname(self.directory)
            relpaths=[path.relpath(f,self.directory) for f in self.__names__()]
            for n,o in zip(relpaths,self.__names__()):
                if n!=o:
                    self.__setter__(n,self.__getter__(o))
                    self.__deleter__(o)
        return self

    def keep_latest(self):
        """Filter out earlier revisions of files with the same name.

        The CM group LabVIEW software will avoid overwirting files when measuring by inserting !#### where #### is an integer revision number just before the
        filename extension. This method will look for instances of several files which differ in name only by the presence of the revision number and will
        kepp only the highest revision number. This is useful if several measurements of the same experiment have been carried out, but only the last file is
        the correct one.

        Returns:
            A copy of the DataFolder.
        """
        files=list(self.ls)
        keep=set(self._discard_earlier(files))
        for f in list(set(files)-keep):
            self.__deleter__(self.__lookup__(f))
        return self

    def on_load_process(self,tmp):
        """Carry out processing on a newly loaded file to set means and extra metadata."""
        for p in self.pattern:
            if isinstance(p,_pattern_type) and (p.search(tmp.filename) is not None):
                m=p.search(tmp.filename)
                for k in m.groupdict():
                    tmp.metadata[k]=tmp.metadata.string_to_type(m.group(k))
        if self.read_means: #Add mean and standard deviations to the metadata
            if len(tmp)==0:
                pass
            elif len(tmp)==1:
                for h in tmp.column_headers:
                    tmp[h]=tmp.column(h)[0]
                    tmp["{}_stdev".format(h)]=None
            else:
                for h in tmp.column_headers:
                    tmp[h]=mean(tmp.column(h))
                    tmp["{}_stdev".format(h)]=std(tmp.column(h))
        tmp['Loaded from']=tmp.filename
        return tmp

    def save(self,root=None):
        """Save the entire data folder out to disc using the groups as a directory tree,
        calling the save method for each file in turn.

        Args:
            root (string): The root directory to start creating files and subdirectories under. If set to None or not specified, the current folder's
                diretory attribute will be used.
        Returns:
            A list of the saved files
        """
        return self.walk_groups(self._save,walker_args={"root",root})

    def unload(self,name):
        """Removes the instance from memory without losing the name in the Folder.

        Args:
            name(string or int): Specifies the entry to unload from memeory.

        Returns:
            (DataFolder): returns a copy of itself.
        """
        name = self.__lookup__(name)
        self.__setter__(name,None)
        return self

class DataMethodsMixin(object):

    """A mixin class that provides a :py:class:`Stoner.folders.core.baseFolder` with methods for working with :py:class:`Stoner.Data` objects.
    """

    def __read__(self,f):
        """Reads a single filename in and creates an instance of metadataObject.

        Args:
            f(string or :py:class:`Stoner.Core.metadataObject`): A filename or metadataObject object

        Returns:
            A metadataObject object

        Note:
             If self.pattern is a regular expression then use any named groups in it to create matadata from the
            filename. If self.read_means is true then create metadata from the mean of the data columns.
        """
        if isinstance(f,DataFile):
            return f
        tmp= self.type(self.loader(f,**self.extra_args))
        if not isinstance(tmp.filename,string_types):
            tmp.filename=path.basename(f)
        for p in self.pattern:
            if isinstance(p,_pattern_type) and (p.search(tmp.filename) is not None):
                m=p.search(tmp.filename)
                for k in m.groupdict():
                    tmp.metadata[k]=tmp.metadata.string_to_type(m.group(k))
        if self.read_means:
            if len(tmp)==0:
                pass
            elif len(tmp)==1:
                for h in tmp.column_headers:
                    tmp[h]=tmp.column(h)[0]
            else:
                for h in tmp.column_headers:
                    tmp[h]=mean(tmp.column(h))
        tmp['Loaded from']=tmp.filename
        for k in self._file_attrs:
            tmp.__setattr__(k,self._file_attrs[k])
        return tmp

    def concatenate(self,sort=None,reverse=False):
        """Concatentates all the files in a objectFolder into a single metadataObject like object.

        Keyword Arguments:
            sort (column index, None or bool, or clallable function): Sort the resultant metadataObject by this column (if a column index),
                or by the *x* column if None or True, or not at all if False. *sort* is passed directly to the eponymous method as the
                *order* paramter.
            reverse (bool): Reverse the order of the sort (defaults to False)

        Returns:
            The current objectFolder with only one metadataObject item containing all the data.
        """
        for d in self[1:]:
            self[0]+=d
        del self[1:]

        if not isinstance(sort,bool) or sort:
            if isinstance(sort, bool) or sort is None:
                sort=self[0].setas["x"]
            self[0].sort(order=sort,reverse=True)

        return self

    def extract(self,*metadata,**kargs):
        """Walks through the terminal group and gets the listed metadata from each file and constructsa replacement metadataObject.

        Args:
            *metadata (str): One or more metadata indices that should be used to construct the new data file.

        Ketyword Arguments:
            copy (bool): Take a copy of the :py:class:`DataFolder` before starting the extract (default is True)

        Returns:
            An instance of a metadataObject like object.
        """
        copy=kargs.pop("copy",True)

        args=[]
        for m in metadata:
            if isinstance(m,string_types):
                args.append(m)
            elif isiterable(m):
                args.extend(m)
            else:
                raise TypeError("Metadata values should be strings, or lists of strings, not {}".format(type(m)))
        metadata=args

        def _extractor(group,trail,metadata):

            results=group.type()
            results.metadata=group[0].metadata
            headers=[]

            ok_data=list()
            for m in metadata: # Sanity check the metadata to include
                try:
                    test=results[m]
                    if not isiterable(test) or isinstance(test,string_types):
                        test=array([test])
                    else:
                        test=array(test)
                except Exception:
                    continue
                else:
                    ok_data.append(m)
                    headers.extend([m]*len(test))

            for d in group:
                row=array([])
                for m in ok_data:
                    row=append(row,array(d[m]))
                results+=row
            results.column_headers=headers

            return results

        if copy:
            ret=self.clone
        else:
            ret=self

        return ret.walk_groups(_extractor,group=True,replace_terminal=True,walker_args={"metadata":metadata})

    def gather(self,xcol=None,ycol=None):
        """Collects xy and y columns from the subfiles in the final group in the tree and builds iunto a :py:class:`Stoner.Core.metadataObject`

        Keyword Arguments:
            xcol (index or None): Column in each file that has x data. if None, then the setas settings are used
            ycol (index or None): Column(s) in each filwe that contain the y data. If none, then the setas settings are used.

        Notes:
            This is a wrapper around walk_groups that assembles the data into a single file for further analysis/plotting.

        """
        def _gatherer(group,trail,xcol=None,ycol=None,xerr=None,yerr=None,**kargs):
            yerr=None
            xerr=None
            cols=group[0]._col_args(xcol=xcol,ycol=ycol,xerr=xerr,yerr=yerr,scalar=False)
            lookup=xcol is None and ycol is None
            xcol=cols["xcol"]

            if  cols["has_xerr"]:
                xerr=cols["xerr"]
            else:
                xerr=None

            common_x=kargs.pop("common_x",True)

            results=group.type()
            results.metadata=group[0].metadata
            xbase=group[0].column(xcol)
            xtitle=group[0].column_headers[xcol]
            results.add_column(xbase,header=xtitle,setas="x")
            if cols["has_xerr"]:
                xerrdata=group[0].column(xerr)
                xerr_title="Error in {}".format(xtitle)
                results.add_column(xerrdata,header=xerr_title,setas="d")
            for f in group:
                if lookup:
                    cols=f._col_args(scalar=False)
                    xcol=cols["xcol"]
                xdata=f.column(xcol)
                if any(xdata!=xbase) and not common_x:
                    xtitle=group[0].column_headers[xcol]
                    results.add_column(xbase,header=xtitle,setas="x")
                    xbase=xdata
                    if cols["has_xerr"]:
                        xerr=cols["xerr"]
                        xerrdata=f.column(xerr)
                        xerr_title="Error in {}".format(xtitle)
                        results.add_column(xerrdata,header=xerr_title,setas="d")
                for col,has_err,ecol,setcol,setecol in zip(["ycol","zcol","ucol","vcol","wcol"],
                                                    ["has_yerr","has_zerr","","",""],
                                                    ["yerr","zerr","","",""], "yzuvw","ef..."):
                    if len(cols[col])==0:
                        continue
                    data=f.column(cols[col])
                    for i in range(len(cols[col])):
                        title="{}:{}".format(path.basename(f.filename),f.column_headers[cols[col][i]])
                        results.add_column(data[:,i],header=title,setas=setcol)
                    if has_err!="" and cols[has_err]:
                        err_data=f.column(cols[ecol])
                        for i in range(len(cols[ecol])):
                            title="{}:{}".format(path.basename(f.filename),f.column_headers[cols[ecol][i]])
                            results.add_column(err_data[:,i],header=title,setas=setecol)
            return results

        return self.walk_groups(_gatherer,group=True,replace_terminal=True,walker_args={"xcol":xcol,"ycol":ycol})

class PlotMethodsMixin(object):
    """A Mixin for :py:class:`Stoner.folders.core.baseFolder` with extra methods for plotting lots of files.

    Example:

        .. plot:: samples/plot-folder-test.py
            :include-source:
            :outname:  plotfolder
    """

    def figure(self,*args,**kargs):
        """Pass through for :py:func:`matplotlib.pyplot.figure` but alos takes a note of the arguments for later."""
        self._fig_args=args
        self._fig_kargs=kargs
        self.__figure=figure(*args,**kargs)
        return self.__fiogure

    def plot(self,*args,**kargs):
        """Call the plot method for each metadataObject, but switching to a subplot each time.

        Args:
            args: Positional arguments to pass through to the :py:meth:`Stoner.plot.PlotMixin.plot` call.
            kargs: Keyword arguments to pass through to the :py:meth:`Stoner.plot.PlotMixin.plot` call.

        Keyword Arguments:
            extra (callable(i,j,d)): A callable that can carry out additional processing per plot after the plot is done
            figsize(tuple(x,y)): Size of the figure to create
            dpi(float): dots per inch on the figure
            edgecolor,facecolor(matplotlib colour): figure edge and frame colours.
            frameon (bool): Turns figure frames on or off
            FigureClass(class): Passed to matplotlib figure call.
            plots_per_page(int): maximum number of plots per figure.
            tight_layout(dict or False): If not False, arguments to pass to a call of :py:func:`matplotlib.pyplot.tight_layout`. Defaults to {}

        Returns:
            A list of :py:class:`matplotlib.pyplot.Axes` instances.

        Notes:
            If the underlying type of the :py:class:`Stoner.Core.metadataObject` instances in the :py:class:`PlotFolder`
            lacks a **plot** method, then the instances are converted to :py:class:`Stoner.Core.Data`.

            Each plot is generated as sub-plot on a page. The number of rows and columns of subplots is computed
            from the aspect ratio of the figure and the number of files in the :py:class:`PlotFolder`.
        """
        plts=kargs.pop("plots_per_page",getattr(self,"plots_per_page",len(self)))
        plts=min(plts,len(self))

        if not hasattr(self.type,"plot"): # switch the objects to being Stoner.Data instances
            from Stoner import Data
            for i,d in enumerate(self):
                self[i]=Data(d)

        extra=kargs.pop("extra",lambda i,j,d:None)
        tight=kargs.pop("tight_layout",{})

        fig_num=kargs.pop("figure",getattr(self,"__figure",None))
        if isinstance(fig_num,Figure):
            fig_num=fig_num.number
        fig_args=getattr(self,"_fig_args",[])
        fig_kargs=getattr(self,"_fig_kargs",{})
        for arg in ("figsize", "dpi", "facecolor", "edgecolor", "frameon", "FigureClass"):
            if arg in kargs:
                fig_kargs[arg]=kargs.pop(arg)
        if fig_num is None:
            fig=figure(*fig_args,**fig_kargs)
        else:
            fig=figure(fig_num,**fig_args)
        w,h=fig.get_size_inches()
        plt_x=floor(sqrt(plts*w/h))
        plt_y=ceil(plts/plt_x)

        kargs["figure"]=fig
        ret=[]
        j=0
        for i,d in enumerate(self):
            if i%plts==0 and i!=0:
                if isinstance(tight,dict):
                    tight_layout(**tight)
                fig=figure(*fig_args,**fig_kargs)
                j=1
            else:
                j+=1
            ax=subplot(plt_y,plt_x,j)
            kargs["fig"]=fig
            kargs["ax"]=ax
            ret.append(d.plot(*args,**kargs))
            extra(i,j,d)
        tight_layout()
        return ret
