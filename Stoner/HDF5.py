"""Stoner.HDF5 - Defines classes that use the hdf5 file format to store data on disc.

Classes include

* HDF5File - A :py:class:`Stoner.Code.DataFile` subclass that can save and load data from hdf5 files
* HDF5Folder - A :py:class:`Stoner.Folders.DataFolder` subclass that can save and load data from a single hdf5 file

It is only necessary to import this module for the subclasses of :py:class:`Stoner.Core.DataFile` to become available
to :py:class:`Stoner.Core.Data`.

"""
from Stoner.compat import string_types,bytes2str,get_filedialog
import h5py
import numpy as _np_
from .Core import DataFile, StonerLoadError
from .Core import Data
from .Image.core import ImageArray
import os.path as path
import os


class HDF5File(DataFile):
    
    """A sub class of DataFile that sores itself in a HDF5File or group.

    Args:
        args (tuple): Supplied arguments, only recognises one though !
        kargs (dict): Dictionary of keyword arguments

    If the first non-keyword arguement is not an h5py File or Group then
    initialises with a blank parent constructor and then loads data, otherwise,
    calls parent constructor.

    Datalayout is dead simple, the numerical data is in a dataset called *data*,
    metadata are attribtutes of a group called *metadata* with the keynames being the
    full name + typehint of the stanard DataFile metadata dictionary
    *column_headers* are an attribute of the root file/group
    *filename* is set from either an attribute called filename, or from the
    group name or from the hdf5 filename.
    The root has an attribute *type* that must by 'HDF5File' otherwise the load routine
    will refuse to load it. This is to try to avoid loading rubbish from random hdf files.
    """

    priority = 16
    compression = 'gzip'
    compression_opts = 6
    patterns = ["*.hdf", "*.hf5"]
    mime_type=["application/x-hdf"]

    #    def __init__(self,*args,**kargs):
    #       """Constructor to catch initialising with an h5py.File or h5py.Group
    #       """
    #       if len(args)>0:
    #           other=args[0]
    #           if isinstance(other,h5py.File) or isinstance(other,h5py.Group):
    #               super(HDF5File,self).__init__(**kargs)
    #               self.load(other,**kargs)
    #           else:
    #               super(HDF5File,self).__init__(*args,**kargs)

    def _raise_error(self,f,message="Not a valid hdf5 file."):
        """Try to clsoe the filehandle f and raise a StonerLoadError."""
        try:
            f.file.close()
        except Exception:
            pass
        raise StonerLoadError(message)


    def _open_filename(self,filename):
        """Examine a file to see if it is an HDF5 file and open it if so.
        
        Args:
            filename (str): Name of the file to open
            
        Returns:
            (f5py.Group): Valid h5py.Group containg data/
            
        Raises:
            StonerLoadError if not a valid file.
        """
        parts=filename.split(os.pathsep)
        filename=parts.pop(0)
        group=""
        while len(parts)>0:
            if not path.exists(path.join(filename,parts[0])):
                group="/".join(parts)
            else:
                path.join(filename,parts.pop(0))
                    
                
        with open(filename,"rb") as sniff: # Some code to manaully look for the HDF5 format magic numbers
            sniff.seek(0,2)
            size=sniff.tell()
            sniff.seek(0)
            blk=sniff.read(8)
            if not blk==b'\x89HDF\r\n\x1a\n':
                c=0
                while sniff.tell()<size and len(blk)==8:
                    sniff.seek(512*2**c)
                    c+=1
                    blk=sniff.read(8)
                    if blk==b'\x89HDF\r\n\x1a\n':
                        break
                else:
                    raise StonerLoadError("Couldn't find the HD5 format singature block")
        try:
            f = h5py.File(filename, 'r')
            for grp in group.split("/"):
                if grp.strip()!="":
                    f=f[grp]
        except IOError:
            self._raise_error(f,message = "Failed to open {} as a n hdf5 file".format(filename))
        except KeyError:
            self._raise_error(f,message = "Could not find group {} in file {}".format(group,filename))
        return f

    def _load(self, filename=None, **kargs):
        """Loads data from a hdf5 file

        Args:
            h5file (string or h5py.Group): Either a string or an h5py Group object to load data from

        Returns:
            itself after having loaded the data
        """
        if filename is None or not filename:
            self.get_filename('r')
            filename = self.filename
        else:
            self.filename = filename
        if isinstance(filename, string_types):  #We got a string, so we'll treat it like a file...
            f = self._open_filename(filename)
        elif isinstance(filename, h5py.File) or isinstance(filename, h5py.Group):
            f = filename
        else:
            self._raise_error(f,message = "Couldn't interpret {} as a valid HDF5 file or group or filename".format(filename))
        if "type" not in f.attrs or bytes2str(
            f.attrs["type"]) != "HDF5File":  #Ensure that if we have a type attribute it tells us we're the right type !
            self._raise_error(f,message = "HDF5 group doesn't hold an HD5File")
        data = f["data"]
        if _np_.product(_np_.array(data.shape)) > 0:
            self.data = data[...]
        else:
            self.data = [[]]
        metadata = f.require_group('metadata')
        if "column_headers" in f.attrs:
            self.column_headers = [x.decode("utf8") for x in f.attrs["column_headers"]]
            if isinstance(self.column_headers, string_types):
                self.column_headers = self.metadata.string_to_type(self.column_headers)
            self.column_headers = [bytes2str(x) for x in self.column_headers]
        else:
            raise StonerLoadError("Couldn't work out where my column headers were !")
        for i in metadata.attrs:
            self[i] = metadata.attrs[i]
        if isinstance(f, h5py.Group):
            if f.name!="/":
                self.filename = os.path.join(f.file.filename,f.name)
            else:
                self.filename = os.path.realpath(f.file.filename)
        else:
            self.filename = os.path.realpath(f.filename)
        if isinstance(filename, string_types):
            f.file.close()
        return self

    def save(self, h5file=None):
        """Writes the current object into  an hdf5 file or group within a file in afashion that is compatible with being loaded in again.

        Args:
            h5file (string or h5py.Group): Either a string, of h5py.File or h5py.Group object into which
                to save the file. If this is a string, the corresponding file is opened for
                writing, written to and save again.

        Returns
            A copy of the object
        """
        if h5file is None:
            h5file = self.filename
        if h5file is None or (isinstance(h5file, bool) and not h5file):  # now go and ask for one
            h5file = self.__file_dialog('w')
            self.filename = h5file
        if isinstance(h5file, string_types):
            f = h5py.File(h5file, 'w')
        elif isinstance(h5file, h5py.File) or isinstance(h5file, h5py.Group):
            f = h5file
        try:
            f.require_dataset("data",
                             data=self.data,
                             shape=self.data.shape,
                             dtype=self.data.dtype,
                             compression=self.compression,
                             compression_opts=self.compression_opts)
            metadata = f.require_group("metadata")
            for k in self.metadata:
                try:
                    metadata.attrs[k] = self[k]
                except TypeError:  # We get this for trying to store a bad data type - fallback to metadata export to string
                    parts = self.metadata.export(k).split('=')
                    metadata[parts[0]] = "=".join(parts[1:])
            f.attrs["column_headers"] = [x.encode("utf8") for x in self.column_headers]
            f.attrs["filename"] = self.filename
            f.attrs["type"] = "HDF5File"
        except Exception as e:
            if isinstance(h5file, str):
                f.file.close()
            raise e
        if isinstance(f,h5py.File):
            self.filename=f.filename
        elif isinstance(f,h5py.Group):
            self.filename=f.file.filename
        else:
            self.filename=h5file
        if isinstance(h5file, string_types):
            f.file.close()

        return self


class HGXFile(DataFile):
 
    """A subclass of DataFile for reading GenX HDF Files.

    These files typically have an extension .hgx. This class has been based on a limited sample
    of hgx files and so may not be sufficiently general to handle all cases.
    """

    priority=16
    pattern=["*.hgx"]
    mime_type=["application/x-hdf"]

    def _load(self, filename=None, *args, **kargs):
        """GenX HDF file loader routine.

        Args:
            filename (string or bool): File to load. If None then the existing filename is used,
                if False, then a file dialog will be used.

        Returns:
            A copy of the itself after loading the data.
        """
        self.seen=[]
        if filename is None or not filename:
            self.get_filename('r')
        else:
            self.filename = filename
        with open(filename,"rb") as sniff: # Some code to manaully look for the HDF5 format magic numbers
            sniff.seek(0,2)
            sniff.seek(0)
            blk=sniff.read(8)
            if not blk==b'\x89HDF\r\n\x1a\n':
                c=0
                while len(blk)==8:
                    sniff.seek(512*2**c)
                    c+=1
                    blk=sniff.read(8)
                    if blk==b'\x89HDF\r\n\x1a\n':
                        break
                else:
                    raise StonerLoadError("Couldn't find the HD5 format singature block")

        try:
            with h5py.File(filename) as f:
                if "current" in f and "config" in f["current"]:
                    pass
                else:
                    raise StonerLoadError("Looks like an unexpected HDF layout!.")
                self.scan_group(f["current"],"")
                self.main_data(f["current"]["data"])
        except IOError:
            raise StonerLoadError("Looks like an unexpected HDF layout!.")
        return self

    def scan_group(self,grp,pth):
        """Recursively list HDF5 Groups."""
        if pth in self.seen:
            return None
        else:
            self.seen.append(pth)
        if not isinstance(grp,h5py.Group):
            return None
        if self.debug: 
            print("Scanning in {}".format(pth))
        for x in grp:
            if pth=="":
                new_pth=x
            else:
                new_pth=pth+"."+x
            if pth=="" and x=="data": # Special case for main data
                continue
            if isinstance(grp[x],type(grp)):
                self.scan_group(grp[x],new_pth)
            elif isinstance(grp[x],h5py.Dataset):
                y=grp[x].value
                self[new_pth]=y
        return None

    def main_data(self,data_grp):
        """Work through the main data group and build something that looks like a numpy 2D array."""
        if not isinstance(data_grp,h5py.Group) or data_grp.name!="/current/data":
            raise StonerLoadError("HDF5 file not in expected format")
        root=data_grp["datasets"]
        for ix in root: # Hack - iterate over all items in root, but actually data is in Groups not DataSets
            dataset=root[ix]
            if isinstance(dataset,h5py.Dataset):
                continue
            x=dataset["x"].value
            y=dataset["y"].value
            e=dataset["error"].value
            self&=x
            self&=y
            self&=e
            self.column_headers[-3]=dataset["x_command"].value
            self.column_headers[-2]=dataset["y_command"].value
            self.column_headers[-1]=dataset["error_command"].value
            self.column_headers=[str(ix) for ix in self.column_headers]



class HDF5Folder(object):
    
    """A mixin class for :py:class:`Stoner.Folders.DataFolder` that provides a method to load and save data from a single HDF5 file with groups.

    See :py:class:`Stoner.Folders.DataFolder` for documentation on constructor.

    Datalayout consistns of sub-groups that are either instances of HDF5Files (i.e. have a type attribute that contains 'HDF5File')
    or are themsleves HDF5Folder instances (with a type attribute that reads 'HDF5Folder').
    """

    def __init__(self, *args, **kargs):
        """Constructor for the HDF5Folder Class."""
        self.File = None
        super(HDF5Folder, self).__init__(*args, **kargs)

    def _dialog(self, message="Select Folder", new_directory=True, mode='r'):
        """Creates a file dialog box for working with

        Args:
            message (string): Message to display in dialog
            new_file (bool): True if allowed to create new directory

        Returns:
            A directory to be used for the file operation.
        """
        file_wildcard = "hdf file (*.hdf5)|*.hdf5|Data file (*.dat)|\
        *.dat|All files|*"

        if mode == "r":
            mode2 = "Open HDF file"
        elif mode == "w":
            mode2 = "Save HDF file as"

        dlg = get_filedialog("file",title=mode2,filetypes=file_wildcard,message=message,mustexist=not new_directory)
        if len(dlg) != 0:
            self.directory = dlg
            self.File = h5py.File(self.directory, mode)
            self.File.close()
            return self.directory
        else:
            return None

    def getlist(self, recursive=None, directory=None, flatten=False):
        """Reads the HDF5 File to construct a list of file HDF5File objects"""
        if recursive is None:
            recursive = self.recursive
        self.files = []
        self.groups = {}
        for d in [directory, self.directory, self.File, True]:
            if isinstance(d, bool) and d:
                d = self._dialog()
            directory = d
            if d is not None:
                break
        if directory is None:
            return None
        if isinstance(directory, string_types):
            self.directory = directory
            directory = h5py.File(directory, 'r')
            self.File = directory
        elif isinstance(directory, h5py.File) or isinstance(directory, h5py.Group):
            self.File = directory.file
            self.directory = self.File.filename
        #At this point directory contains an open h5py.File object, or possibly a group
        for obj in directory:
            obj = directory[obj]
            if isinstance(obj, h5py.Group) and "type" in obj.attrs:
                if obj.attrs["type"] == "HDF5File":
                    self.files.append(obj.name)
                elif obj.attrs["type"] == "HDF5Folder" and recursive:
                    self.groups[obj.name.split(path.sep)[-1]] = self.__class__(obj,
                                                                               pattern=self.pattern,
                                                                               type=self.type)
                else:
                    raise IOError("Found a group {} that isn't recognised".format(obj.name))
        if flatten:
            self.flatten()

        return self

    def save(self, root=None):
        """Saves a load of files to a single HDF5 file, creating groups as it goes.

        Keyword Arguments:
            root (string): The name of the HDF5 file to save to if set to None, will prompt for a filename.

        Return:
            A list of group paths in the HDF5 file
        """
        if root is None:
            root = self._dialog(mode='w')
        elif isinstance(root, bool) and not root and isinstance(self.File, h5py.File):
            root = self.File.filename
            self.File.close()
        self.File = h5py.File(root, 'w')
        tmp = self.walk_groups(self._save)
        self.File.file.close()
        return tmp

    def _save(self, f, trail):
        """Create a virtual path of groups in the HDF5 file and save data.

        Args:
            f(DataFile):  A DataFile instance to save
            trail (list): The trail of groups

        Returns:
            The new filename of the saved DataFile.

        Ensure we have created the trail as a series of sub groups, then create a sub-groupfor the filename
        finally cast the DataFile as a HDF5File and save it, passing the new group as the filename which
        will ensure we create a sub-group in the main HDF5 file

        This routine is used by a walk_groups call - hence the prototype matches that required for
        :py:meth:`Stoner.Folders.DataFolder.walk_groups`.
        """
        tmp = self.File
        if not isinstance(f, DataFile):
            f = Data(f)
        for g in trail:
            if g not in tmp:
                tmp.create_group(g)
            tmp = tmp[g]
        tmp.create_group(f.filename)
        tmp = tmp[f.filename]
        f = HDF5File(f)
        f.save(tmp)
        return f.filename
        
class SLS_STXMFile(DataFile):
    
    """Load images from the Swiss Light Source Pollux beamline"""
    
    priority = 16
    compression = 'gzip'
    compression_opts = 6
    patterns = ["*.hdf",]
    mime_type=["application/x-hdf"]

    def _load(self, filename=None, **kargs):
        """Loads data from a hdf5 file

        Args:
            h5file (string or h5py.Group): Either a string or an h5py Group object to load data from

        Returns:
            itself after having loaded the data
        """
        if filename is None or not filename:
            self.get_filename('r')
            filename = self.filename
        else:
            self.filename = filename
        if isinstance(filename, string_types):  #We got a string, so we'll treat it like a file...
            try:
                f = h5py.File(filename, 'r')
            except IOError:
                raise StonerLoadError("Failed to open {} as a n hdf5 file".format(filename))
        elif isinstance(filename, h5py.File) or isinstance(filename, h5py.Group):
            f = filename
        else:
            raise StonerLoadError("Couldn't interpret {} as a valid HDF5 file or group or filename".format(filename))
        items=[x for x in f.items()]
        if len(items)==1 and items[0][0]=="entry1":
            group1=[x for x in f["entry1"]]
            if  "definition" in group1 and  bytes2str(f["entry1"]["definition"].value[0])=="NXstxm": #Good HDF5
                pass
            else:
                raise StonerLoadError("HDF5 file lacks single top level group called entry1")
        else:
            raise StonerLoadError("HDF5 file lacks single top level group called entry1")
        root=f["entry1"]
        data=root["counter0"]["data"]
        if _np_.product(_np_.array(data.shape)) > 0:
            self.data = data[...]
        else:
            self.data = [[]]
        self.scan_meta(root)
        if "file_name" in f.attrs:
            self["original filename"] = f.attrs["file_name"]
        elif isinstance(f, h5py.Group):
            self["original filename"] = f.name
        else:
            self["original filename"] = f.file.filename

        if isinstance(filename, string_types):
            f.file.close()
        self["Loaded from"]=self.filename
        return self
        
    def scan_meta(self,group):
        """Scan the HDF5 Group for atributes and datasets and sub groups and recursively add them to the metadata."""
        root=".".join(group.name.split('/')[2:])
        for name,thing in group.items():
            parts=thing.name.split("/")
            name=".".join(parts[2:])
            if isinstance(thing,h5py.Group):
                self.scan_meta(thing)
            elif isinstance(thing,h5py.Dataset):
                if len(thing.shape)>1:
                    continue
                if _np_.product(thing.shape)==1:
                    self.metadata[name]=thing.value[0]
                else:
                    self.metadata[name]=thing.value
        for attr in group.attrs:
            self.metadata["{}.{}".format(root,attr)]=group.attrs[attr]
            
class STXMImage(ImageArray):
    
    """An instance of KerrArray that will load itself from a Swiss Light Source STXM image"""

    _reduce_metadata=False

    @classmethod
    def _load(self,filename):
        d=SLS_STXMFile(filename)
        return d.data,d.metadata