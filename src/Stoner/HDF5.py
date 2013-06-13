"""Stoner.hdf5 Defines classes that use the hdf5 file format to store data on disc.
Classes include

* \b HDF5File - A \b DataFile subclass that can save and load data from hdf5 files
* \b HDF5Folder - A \b DataFolder subclass that can save and load data from a single hdf5 file

"""

import h5py
import numpy
from .Core import DataFile
from .Folders import DataFolder
import os.path as path

class HDF5File(DataFile):
    """A sub class of DataFile that sores itself in a HDF5File or group.
    Overloads self.load and self.save
    """
    
    priority=32
    compression='gzip'
    compression_opts=6
    
    def load(self,h5file,**kargs):
        """Loads data from a hdf5 file
        @param h5file Either a string or an h5py Group object to load data from
        @return itself after having loaded the data
        
        Datalayout is dead simple, the numerical data is in a dataset called 'data',
        metadata are attribtutes of a group called metadata
        column_headers are an attribute of the root file/group
        filename is set from either an attribute called filename, or from the
        group name or from the hdf5 filename.
        
        """
        if isinstance(h5file,str): #We got a string, so we'll treat it like a file...
            try:
                f=h5py.File(h5file,'r')
            except IOError:
                raise IOError("Failed to open {} as a n hdf5 file".format(h5file))
        elif isinstance(h5file,h5py.File) or isinstance(h5file,h5py.Group):
            f=h5file
        else:
            raise IOError("Couldn't interpret {} as a valid HDF5 file or group or filename".format(h5file))
        if "type" in f.attrs: #Ensure that if we have a type attribute it tells us we're the right type !
            assert f.attrs["type"]=="HDF5File",TypeError("HDF5 group doesn't hold an HD5File") 
        data=f["data"]
        if numpy.product(numpy.array(data.shape))>0:
            self.data=data[...]
        else:
            self.data=[[]]
        metadata=f.require_group('metadata')
        for i in metadata.attrs:
            self.metadata[i]=metadata.attrs[i]
        self.column_headers=f.attrs["column_headers"]
        if "filename" in f.attrs:
            self.filename=f.attrs["filename"]
        elif isinstance(f,h5py.Group):
            self.filename=f.name
        else:
            self.filename=f.file.filename
            
        if isinstance(h5file,str):
            f.file.close()
        return self
        
    def save(self,h5file):
        """Writes the current object into  an hdf5 file or group within a file in a 
        fashion that is compatible with being loaded in again with the same class.
        @param h5file Either a string, of h5py.File or h5py.Group object into which
        to save the file. If this is a string, the corresponding file is opened for
        writing, written to and save again.
        @return A copy of the object
        """
        if isinstance(h5file,str):
            f=h5py.File(h5file,'w')
        elif isinstance(h5file,h5py.File) or isinstance(h5file,h5py.Group):
            f=h5file
        try:
            data=f.require_dataset("data",data=self.data,shape=self.data.shape,dtype=self.data.dtype,compression=self.compression,compression_opts=self.compression_opts)
            data=self.data
            metadata=f.require_group("metadata")
            for k in self.metadata:
                try:
                    metadata.attrs[k]=self[k]
                except TypeError: # We get this for trying to store a bad data type - fallback to metadata export to string
                    parts=self.metadata.export(k).split('=')
                    metadata[parts[0]]="=".join(parts[1:])
            f.attrs["column_headers"]=self.column_headers
            f.attrs["filename"]=self.filename
            f.attrs["type"]="HDF5File"
        except Exception as e:
            if isinstance(h5file,str):
                f.file.close()
            raise e
        if isinstance(h5file,str):
            f.file.close()
        
        return self
        
class HDF5Folder(DataFolder):
    """A sub class of DataFolder that provides a method to load and save data from a single HDF5 file with groups."""

    def _dialog(self, message="Select Folder",  new_directory=True,mode='r'):
        """Creates a file dialog box for working with

        @param message Message to display in dialog
        @param new_file True if allowed to create new directory
        @return A directory to be used for the file operation."""
        try:
            from enthought.pyface.api import FileDialog, OK
        except ImportError:
            from pyface.api import FileDialog, OK
        # Wildcard pattern to be used in file dialogs.
        file_wildcard = "Text file (*.txt)|*.txt|Data file (*.dat)|\
        *.dat|All files|*"

        if mode == "r":
            mode = "open"
        elif mode == "w":
            mode = "save as"

        if self.directory is not None:
            filename = path.basename(self.directory)
            dirname = path.dirname(self.directory)
        else:
            filename = ""
            dirname = ""
        dlg = FileDialog(action=mode, wildcard=file_wildcard)
        dlg.open()
        if dlg.return_code == OK:
            self.directory = dlg.path
            return self.directory
        else:
            return None

    def getlist(self, recursive=True, directory=None):
         """@TODO Write the HDF5Folder getlist function"""
         pass
    
         
            
    def save(self,root=None):
        """Saves a load of files to a single HDF5 file, creating groups as it goes.
        @param root The name of the HDF5 file to save to if set to None, will prompt for a filename.
        @return A list of group paths in the HDF5 file
        """
        
        if root is None:
            root=self._file_dialog(mode='w')
        elif isinstance(root,bool) and not root and isinstance(self.File,h5py.File):
            root=self.File.filename
        self.File=h5py.File(root,'w')
        tmp=self.walk_groups(self._save,walker_args={"root":root})
        self.File.file.close()
        return tmp
        
    def _save(self,f,trail,root):
        """Ensure we have created the trail as a series of sub groups, then create a sub-groupfor the filename
        finally cast the DataFile as a HDF5File and save it, passing the new group as the filename which
        will ensure we create a sub-group in the main HDF5 file
        @param f A DataFile instance
        @param trail The trail of groups
        @param root the starting HDF5 File.
        @return the new filename
        """
        tmp=root
        for g in trail:
            if g not in tmp:
                tmp.create_group(g)
            tmp=tmp[g]
        tmp.create_group(f.filename)
        tmp=tmp[f.filename]
        f=HDF5File(f)
        f.save(tmp)
        return f.filename
            
        
            
            
        
    