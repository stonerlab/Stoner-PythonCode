#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mixin classes for :py:class:`Stoner.folders.core.baseFoler`."""
from __future__ import division

__all__ = ["DiskBasedFolderMixin", "DataMethodsMixin", "PlotMethodsMixin"]

import os
import os.path as path
from functools import partial
from copy import deepcopy
from importlib import import_module

from numpy import mean, std, array, append, any as np_any, floor, sqrt, ceil
from numpy.ma import masked_invalid
from matplotlib.pyplot import figure, subplots

from Stoner.tools import isiterable, make_Data
from ..compat import string_types, get_filedialog, _pattern_type, makedirs, path_types
from ..core.base import metadataObject, string_to_type
from ..core.exceptions import StonerUnrecognisedFormat
from .core import baseFolder, _add_core_ as _base_add_core_, _sub_core_ as _base_sub_core_
from .utils import scan_dir, discard_earlier, filter_files, get_pool, removeDisallowedFilenameChars
from ..core.exceptions import assertion


regexp_type = (_pattern_type,)


def _add_core_(result, other):
    """Additional logic for the add operator."""
    if isinstance(other, path_types):
        othername = path.join(result.directory, other)
        if path.exists(othername) and othername not in result:
            result.append(othername)
        else:
            raise RuntimeError(f"{othername} either does not exist of is already in the folder.")
    else:
        return _base_add_core_(result, other)
    return result


def _sub_core_(result, other):
    """Additional logic to check for match to basenames."""
    if isinstance(other, path_types):
        if str(other) in list(result.basenames) and path.join(result.directory, other) in list(result.ls):
            other = path.join(result.directory, other)
            result.__deleter__(other)
            return result
    return _base_sub_core_(result, other)


def _loader(name, loader=None, typ=None, directory=None):
    """Lods and returns an object."""
    filename = name if path.exists(name) else path.join(directory, name)
    return typ(loader(filename)), name


class DiskBasedFolderMixin:
    """A Mixin class that implements reading metadataObjects from disc.

    Attributes:
        type (:py:class:`Stoner.Core.metadataObject`):
            the type ob object to store in the folder (defaults to :py:class:`Stoner.Core.Data`)
        extra_args (dict):
            Extra arguments to use when instantiatoing the contents of the folder from a file on disk.
        pattern (str or regexp):
            A filename globbing pattern that matches the contents of the folder. If a regular expression is provided
            then any named groups are used to construct additional metadata entryies from the filename. Default
            is *.* to match all files with an extension.
        exclude (str or regexp):
            A filename globbing pattern that matches files to exclude from the folder.  Default is *.tdms_index to
            exclude all tdms index files.
        read_means (bool):
            If true, additional metadata keys are added that return the mean value of each column of the data.
            This can hep in grouping files where one column of data contains a constant value for the experimental
            state. Default is False
        recursive (bool):
            Specifies whether to search recursively in a whole directory tree. Default is True.
        flatten (bool):
            Specify where to present subdirectories as separate groups in the folder (False) or as a single group
            (True). Default is False. The :py:meth:`DiskBasedFolderMixin.flatten` method has the equivalent effect
            and :py:meth:`DiskBasedFolderMixin.unflatten` reverses it.
        discard_earlier (bool):
            If there are several files with the same filename apart from !#### being appended just before the
            extension, then discard all except the one with the largest value of #### when collecting the list of
            files.
        directory (str):
            The root directory on disc for the folder - by default this is the current working directory.
        multifile (boo):
            Whether to select individual files manually that are not (necessarily) in  a common directory structure.
        readlist (bool):
            Whether to read the directory immediately on creation. Default is True
    """

    _defaults = {
        "type": None,
        "extra_args": dict(),
        "pattern": ["*.*"],
        "exclude": ["*.tdms_index"],
        "read_means": False,
        "recursive": True,
        "flat": False,
        "prefetch": False,
        "directory": None,
        "multifile": False,
        "pruned": True,
        "readlist": True,
        "discard_earlier": False,
    }

    def __init__(self, *args, **kargs):
        """Additional constructor for DiskBasedFolderMixins."""
        _ = self.defaults  # Force the default store to be populated.
        if "directory" in self._default_store and self._default_store["directory"] is None:
            self._default_store["directory"] = os.getcwd()
        if "type" in self._default_store and self._default_store["type"] is None and self._type == metadataObject:
            self._default_store["type"] = make_Data(None)
        elif self._type != metadataObject:  # Looks like we've already set our type in a subbclass
            self._default_store.pop("type")
        flat = kargs.pop("flat", self._default_store.get("flat", False))
        prefetch = kargs.pop("prefetch", self._default_store.get("prefetch", False))
        if "type" in kargs and isinstance(kargs["type"], str):
            if "." in kargs["type"]:
                mod = ".".join(kargs["type"].split(".")[:-1])
                cls = kargs["type"].split(".")[-1]
                mod = import_module(mod)
                kargs["type"] = getattr(mod, cls)
            else:
                kargs["type"] = globals()[kargs["type"]]

        # Adjust the default pattern depending on the specified type
        if "type" in kargs and "pattern" not in kargs:
            kargs["pattern"] = kargs["type"]._patterns
        super().__init__(*args, **kargs)  # initialise before __clone__ is called in getlist
        if self.readlist and len(args) > 0 and isinstance(args[0], path_types):
            self.getlist(directory=args[0])
        if len(args) > 0 and isinstance(args[0], bool) and not args[0]:
            self.getlist(directory=args[0])
        if flat:
            self.flatten()
        if prefetch:
            self.fetch()
        if self.pruned:
            self.prune()

    @baseFolder.key.getter  # pylint: disable=no-member
    def key(self):
        """Override the parent class *key* to use the *directory* attribute."""
        k = getattr(super(), "key", None)
        if k is None:
            self.key = self.directory
            return self._key
        return k

    def _dialog(self, message="Select Folder", new_directory=True):
        """Create a directory dialog box for working with.

        Keyword Arguments:
            message (string):
                Message to display in dialog
            new_directory (bool):
                True if allowed to create new directory

        Returns:
            A directory to be used for the file operation.
        """
        # Wildcard pattern to be used in file dialogs.
        if not self.multifile:
            mode = "directory"
        else:
            mode = "files"
        dlg = get_filedialog(what=mode, title=message)
        if not dlg or len(str(dlg)) == 0:
            raise RuntimeError("No directory or files selected!")
        if mode == "directory" and new_directory and not path.exists(str(dlg)):
            os.makedirs(dlg, exist_ok=True)
        if self.multifile:
            self.pattern = [path.basename(name) for name in dlg]
            self.directory = path.dirname(dlg[0]) if len(dlg) == 1 else path.commonprefix(dlg)
        else:
            self.directory = dlg
        return self.directory

    def _save(self, grp, trail, root=None):
        """Save a group of files to disc by calling the save() method on each file.

        This internal method is called by walk_groups in turn
        called from the public save() method. The trail of group keys is used to create a directory tree.

        Args:
            grp (:py:class:`objectFolder` or :py:calss:`Stoner.metadataObject`):
                A group or file to save trail (list of strings): the trail of paths used to get here
            root (string or None):
                a replacement root directory

        Returns:
            Saved Path
        """
        trail = [removeDisallowedFilenameChars(t) for t in trail]
        grp.filename = removeDisallowedFilenameChars(grp.filename)
        if root is None:
            root = self.directory

        pth = path.join(root, *trail)
        makedirs(pth, exist_ok=True)
        if isinstance(grp, metadataObject) and not isinstance(grp, self.loader):
            grp = self.loader(grp)
        grp.save(path.join(pth, grp.filename))
        return grp.filename

    def __lookup__(self, name):
        """Additional logic for the looking up names."""
        if isinstance(name, string_types):
            if list(self.basenames).count(name) == 1:
                return self.__names__()[list(self.basenames).index(name)]

        return super().__lookup__(name)

    def __getter__(self, name, instantiate=True):
        """Load the specified name from a file on disk.

        Parameters:
            name (key type):
                The canonical mapping key to get the dataObject. By default
                the baseFolder class uses a :py:class:`regexpDict` to store objects in.

        Keyword Arguments:
            instantiate (bool):
                If True (default) then always return a :py:class:`Stoner.Core.Data` object. If False,
                the __getter__ method may return a key that can be used by it later to actually get the
                :py:class:`Stoner.Core.Data` object.

        Returns:
            (metadataObject): The metadataObject
        """
        assertion(name is not None, "Cannot get an anonympus entry!")
        try:  # Try the parent methods first
            return super().__getter__(name, instantiate=instantiate)
        except (AttributeError, IndexError, KeyError):
            pass
        # name may still be a number when we have unloaded entries referred to by index:
        if isinstance(name, int):
            name = self.__names__()[name]
        # Find a filename and load
        fname = name if path.exists(name) else path.join(self.directory, name)
        try:
            tmp = self.type(self.loader(fname, **self.extra_args))
        except StonerUnrecognisedFormat:
            return None
        if not isinstance(getattr(tmp, "filename", None), path_types) or len(getattr(tmp, "filename", "")) == 0:
            tmp.filename = path.basename(fname)
        # Process file hooks
        tmp = self.on_load_process(tmp)
        tmp = self._update_from_object_attrs(tmp)
        # Store the result
        self.__setter__(name, tmp)
        return tmp

    def __add__(self, other):
        """Implement the addition operator for baseFolder and metadataObjects."""
        result = deepcopy(self)
        result = _add_core_(result, other)
        return result

    def __iadd__(self, other):
        """Implement the addition operator for baseFolder and metadataObjects."""
        result = self
        result = _add_core_(result, other)
        return result

    def __sub__(self, other):
        """Implement the addition operator for baseFolder and metadataObjects."""
        result = deepcopy(self)
        result = _sub_core_(result, other)
        return result

    def __isub__(self, other):
        """Implement the addition operator for baseFolder and metadataObjects."""
        result = self
        result = _sub_core_(result, other)
        return result

    @property
    def basenames(self):
        """Return a list of just the filename parts of the objectFolder."""
        for x in self.__names__():
            yield path.basename(x)

    @property
    def directory(self):
        """Just alias directory to root now."""
        return self.root

    @directory.setter
    def directory(self, value):
        """Just alias directory to root now."""
        self.root = value

    @property
    def not_loaded(self):
        """Return an array of True/False for whether we've loaded a metadataObject yet."""
        for n in self.__names__():
            if not isinstance(self.__getter__(n, instantiate=None), self._type):
                yield n

    @property
    def pattern(self):
        """Provide support for getting the pattern attribute."""
        return self._pattern

    @pattern.setter
    def pattern(self, value):
        """Set the filename searching pattern[s] for the :py:class:`Stoner.Core.metadataObject`s."""
        if isinstance(value, string_types):
            self._pattern = (value,)
        elif isinstance(value, _pattern_type):
            self._pattern = (value,)
        elif isiterable(value):
            self._pattern = [x for x in value]
        else:
            raise ValueError(f"pattern should be a string, regular expression or iterable object not a {value}")

    def fetch(self):
        """Preload the contents of the DiskBasedFolderMixin.

        With multiprocess enabled this will parallel load the contents of the folder into memory.
        """
        p, imap = get_pool()
        for f, name in imap(
            partial(_loader, loader=self.loader, typ=self._type, directory=self.directory), self.not_loaded
        ):
            self.__setter__(
                name, self.on_load_process(f)
            )  # This doesn't run on_load_process in parallel, but it's not expensive enough to make it worth it.
        if p is not None:
            p.close()
            p.join()
        return self

    def getlist(self, **kargs):
        """Scan the current directory, optionally recursively to build a list of filenames.

        Keyword Arguments:
            recursive (bool):
                Do a walk through all the directories for files
            directory (string or False):
                Either a string path to a new directory or False to open a dialog box or not set in which case existing
                directory is used.
            flatten (bool):
                After scanning the directory tree, flaten all the subgroupos to make a flat file list. (this is the
                previous behaviour of :py:meth:`objectFolder.getlist()`)

        Returns:
            A copy of the current DataFoder directory with the files stored in the files attribute

        getlist() scans a directory tree finding files that match the pattern. By default it will recurse through
        the entire directory tree finding sub directories and creating groups in the data folder for each sub
        directory.
        """
        self.__clear__()
        recursive = kargs.pop("recursive", self.recursive)
        discard = kargs.pop("discard_earlier", self.discard_earlier)
        flatten = kargs.pop("flatten", getattr(self, "flat", False))
        directory = kargs.pop("directory", getattr(self, "directory", None))
        self.directory = directory

        if self.multifile or isinstance(directory, bool) and not directory:
            self._dialog()
        elif directory is None:
            self.directory = os.getcwd()
        elif isinstance(directory, path_types):
            self.directory = directory
        root = self.directory
        dirs, files = scan_dir(root)
        if discard:
            files = discard_earlier(files)
        files = filter_files(files, self.exclude, keep=False)
        files = filter_files(files, self.pattern, keep=True)
        for f in files:
            self.__setter__(f, f)
        if recursive and not self.multifile:  # No recursion in multifile mode
            for d in dirs:
                if self.debug:
                    print(f"Entering directory {d}")
                self.add_group(d)
                self.groups[d].getlist(
                    directory=path.join(root, d), recursive=recursive, flatten=flatten, discard_earlier=discard_earlier
                )
        if flatten and not self.is_empty:
            self.flatten()
        return self

    def keep_latest(self):
        """Filter out earlier revisions of files with the same name.

        The CM group LabVIEW software will avoid overwriting files when measuring by inserting !#### where #### is an
        integer revision number just before the filename extension. This method will look for instances of several
        files which differ in name only by the presence of the revision number and will kepp only the highest revision
        number. This is useful if several measurements of the same experiment have been carried out, but only the
        last file is the correct one.

        Returns:
            A copy of the DataFolder.
        """
        files = [x for x in self.ls]
        keep = set(discard_earlier(files))
        for f in list(set(files) - keep):
            self.__deleter__(self.__lookup__(f))
        return self

    def on_load_process(self, tmp):
        """Carry out processing on a newly loaded file to set means and extra metadata."""
        for p in self.pattern:
            if isinstance(p, _pattern_type) and (p.search(tmp.filename) is not None):
                m = p.search(tmp.filename)
                for k in m.groupdict():
                    tmp.metadata[k] = string_to_type(m.group(k))
        if self.read_means:  # Add mean and standard deviations to the metadata
            if len(tmp) == 0:
                pass
            elif len(tmp) == 1:
                for h in tmp.column_headers:
                    tmp[h] = tmp.column(h)[0]
                    tmp[f"{h}_stdev"] = None
            else:
                for h in tmp.column_headers:
                    try:
                        tmp[h] = mean(masked_invalid(tmp.column(h)))
                        tmp[f"{h}_stdev"] = std(masked_invalid(tmp.column(h)))
                    except ValueError:
                        continue
        tmp["Loaded from"] = tmp.filename
        return tmp

    def save(self, root=None):
        """Save the entire data folder out to disc using the groups as a directory tree.

        Calls the save method for each file in turn.

        Args:
            root (string):
                The root directory to start creating files and subdirectories under. If set to None or not specified,
                the current folder's directory attribute will be used.

        Returns:
            A list of the saved files
        """
        return self.walk_groups(self._save, walker_args={"root": root})

    def unload(self, name=None):
        """Remove the instance from memory without losing the name in the Folder.

        Args:
            name(string,int or None):
                Specifies the entry to unload from memory. If set to None all loaded entries are unloaded.

        Returns:
            (DataFolder): returns a copy of itself.
        """
        if name is not None:
            name = [self.__lookup__(name)]
        else:
            name = self.__names__()
        for n in name:
            self.__setter__(n, n)
        return self


class DataMethodsMixin:
    """Methods for working with :py:class:`Stner.Data` in py:class:`Stoner.DataFolder`s."""

    def concatenate(self, sort=None, reverse=False):
        """Concatenates all the files in a objectFolder into a single metadataObject like object.

        Keyword Arguments:
            sort (column index, None or bool, or clallable function):
                Sort the resultant metadataObject by this column (if a column index), or by the *x* column if None
                or True, or not at all if False. *sort* is passed directly to the eponymous method as the
                *order* parameter.
            reverse (bool):
                Reverse the order of the sort (defaults to False)

        Returns:
            The current objectFolder with only one metadataObject item containing all the data.
        """
        for d in self[1:]:
            self[0] += d
        del self[1:]

        if not isinstance(sort, bool) or sort:
            if isinstance(sort, bool) or sort is None:
                sort = self[0].setas.cols["xcol"]
            self[0].sort(order=sort, reverse=reverse)

        return self

    def extract(self, *metadata, **kargs):
        """Extract metadata from each of the files in the terminal group.

        Walks through the terminal group and gets the listed metadata from each file and constructsa replacement
        metadataObject.

        Args:
            *metadata (str):
                One or more metadata indices that should be used to construct the new data file.

        Ketyword Arguments:
            copy (bool):
                Take a copy of the :py:class:`DataFolder` before starting the extract (default is True)

        Returns:
            An instance of a metadataObject like object.
        """
        copy = kargs.pop("copy", True)

        args = []
        for m in metadata:
            if isinstance(m, string_types):
                args.append(m)
            elif isiterable(m):
                args.extend(m)
            else:
                raise TypeError(f"Metadata values should be strings, or lists of strings, not {type(m)}")
        metadata = args

        def _extractor(group, _, metadata):
            results = group.type()
            results.metadata = group[0].metadata
            headers = []

            ok_data = list()
            for m in metadata:  # Sanity check the metadata to include
                try:
                    test = results[m]
                    if not isiterable(test) or isinstance(test, string_types):
                        test = array([test])
                    else:
                        test = array(test)
                except (IndexError, KeyError, TypeError, ValueError):
                    continue
                else:
                    ok_data.append(m)
                    headers.extend([m] * len(test))

            for d in group:
                row = array([])
                for m in ok_data:
                    row = append(row, array(d[m]))
                results += row
            results.column_headers = headers

            return results

        if copy:
            ret = self.clone
        else:
            ret = self

        return ret.walk_groups(_extractor, group=True, replace_terminal=True, walker_args={"metadata": metadata})

    def gather(self, xcol=None, ycol=None):
        """Collect xy and y columns from the subfiles in the final group in the tree.

        Builds the collected data into a :py:class:`Stoner.Core.metadataObject`

        Keyword Arguments:
            xcol (index or None):
                Column in each file that has x data. if None, then the setas settings are used
            ycol (index or None):
                Column(s) in each filwe that contain the y data. If none, then the setas settings are used.

        Notes:
            This is a wrapper around walk_groups that assembles the data into a single file for
            further analysis or plotting.

        """

        def _gatherer(group, _, xcol=None, ycol=None, xerr=None, yerr=None, **kargs):
            yerr = None
            xerr = None
            cols = group[0]._col_args(xcol=xcol, ycol=ycol, xerr=xerr, yerr=yerr, scalar=False)
            lookup = xcol is None and ycol is None
            xcol = cols["xcol"][0]

            if cols["has_xerr"]:
                xerr = cols["xerr"]
            else:
                xerr = None

            common_x = kargs.pop("common_x", True)

            results = group.type()
            results.metadata = group[0].metadata
            xbase = group[0].column(xcol)
            xtitle = group[0].column_headers[xcol]
            results.add_column(xbase, header=xtitle, setas="x")
            if cols["has_xerr"]:
                xerrdata = group[0].column(xerr)
                xerr_title = f"Error in {xtitle}"
                results.add_column(xerrdata, header=xerr_title, setas="d")
            for f in group:
                if lookup:
                    cols = f._col_args(scalar=False)
                    xcol = cols["xcol"]
                xdata = f.column(xcol)
                if np_any(xdata != xbase) and not common_x:
                    xtitle = group[0].column_headers[xcol]
                    results.add_column(xbase, header=xtitle, setas="x")
                    xbase = xdata
                    if cols["has_xerr"]:
                        xerr = cols["xerr"]
                        xerrdata = f.column(xerr)
                        xerr_title = f"Error in {xtitle}"
                        results.add_column(xerrdata, header=xerr_title, setas="d")
                for col, has_err, ecol, setcol, setecol in zip(
                    ["ycol", "zcol", "ucol", "vcol", "wcol"],
                    ["has_yerr", "has_zerr", "", "", ""],
                    ["yerr", "zerr", "", "", ""],
                    "yzuvw",
                    "ef...",
                ):
                    if len(cols[col]) == 0:
                        continue
                    data = f.column(cols[col])
                    for i in range(len(cols[col])):
                        title = f"{path.basename(f.filename)}:{f.column_headers[cols[col][i]]}"
                        results.add_column(data[:, i], header=title, setas=setcol)
                    if has_err != "" and cols[has_err]:
                        err_data = f.column(cols[ecol])
                        for i in range(len(cols[ecol])):
                            title = f"{path.basename(f.filename)}:{f.column_headers[cols[ecol][i]]}"
                            results.add_column(err_data[:, i], header=title, setas=setecol)
            return results

        return self.walk_groups(_gatherer, group=True, replace_terminal=True, walker_args={"xcol": xcol, "ycol": ycol})


class PlotMethodsMixin:
    """A Mixin for :py:class:`Stoner.folders.core.baseFolder` with extra methods for plotting lots of files.

    Example:
        .. plot:: samples/plot-folder-test.py
            :include-source:
            :outname:  plotfolder
    """

    _defaults = {"plots_per_page": 12, "fig_defaults": {"figsize": (8, 6)}}

    def figure(self, *args, **kargs):
        """Pass through for :py:func:`matplotlib.pyplot.figure` but also takes a note of the arguments for later."""
        self._fig_args = args

        self._fig_kargs = getattr(self, "fig_defaults", {})
        self._fig_kargs.update(kargs)
        fig = figure(*self._fig_args, **self._fig_kargs)
        self.each.fig = fig
        self._figure = fig
        return self._figure

    def plot(self, *args, **kargs):
        """Call the plot method for each metadataObject, but switching to a subplot each time.

        Args:
            args: Positional arguments to pass through to the :py:meth:`Stoner.plot.PlotMixin.plot` call.
            kargs: Keyword arguments to pass through to the :py:meth:`Stoner.plot.PlotMixin.plot` call.

        Keyword Arguments:
            plot_extra (callable(i,j,d)):
                A callable that can carry out additional processing per plot after the plot is done
            figsize(tuple(x,y)):
                Size of the figure to create
            dpi(float):
                dots per inch on the figure
            edgecolor,facecolor(matplotlib colour):
                figure edge and frame colours.
            frameon (bool):
                Turns figure frames on or off
            FigureClass(class):
                Passed to matplotlib figure call.
            plots_per_page(int):
                maximum number of plots per figure.

        Returns:
            A list of :py:class:`matplotlib.pyplot.Axes` instances.

        Notes:
            If the underlying type of the :py:class:`Stoner.Core.metadataObject` instances in the
            :py:class:`PlotFolder` lacks a **plot** method, then the instances are converted to
            :py:class:`Stoner.Core.Data`.

            Each plot is generated as sub-plot on a page. The number of rows and columns of subplots is computed
            from the aspect ratio of the figure and the number of files in the :py:class:`PlotFolder`.
        """
        plts = kargs.pop("plots_per_page", getattr(self, "plots_per_page", len(self)))
        plts = min(plts, len(self))

        if not hasattr(self.type, "plot"):  # switch the objects to being Stoner.Data instances
            for i, d in enumerate(self):
                self[i] = make_Data(d)

        plot_extra = kargs.pop("plot_extra", lambda i, j, d: None)

        fig_args = getattr(self, "_fig_args", [])
        fig_kargs = getattr(self, "_fig_kargs", {})
        for arg in ("figsize", "dpi", "facecolor", "edgecolor", "frameon", "FigureClass"):
            if arg in kargs:
                fig_kargs[arg] = kargs.pop(arg)
        w, h = fig_kargs.setdefault("figsize", (18, 12))
        plt_x = int(floor(sqrt(plts * w / h)))
        plt_y = int(ceil(plts / plt_x))
        ret = []
        j = -1
        fig, axs = subplots(plt_x, plt_y, *fig_args, **fig_kargs)
        fignum = fig.number
        for i, d in enumerate(self):
            if i % plts == 0 and i != 0:
                fig, axs = subplots(plt_x, plt_y, *fig_args, **fig_kargs)
                fignum = fig.number
                j = 0
            else:
                j += 1
            fig = figure(fignum)
            kargs["figure"] = fig
            kargs["ax"] = axs.ravel()[j]
            ret.append(d.plot(*args, **kargs))
            plot_extra(i, j, d)
        for n in range(j + 1, plt_x * plt_y):
            axs.ravel()[n].remove()
        return ret


class DataFolder(DataMethodsMixin, DiskBasedFolderMixin, baseFolder):
    """Provide an interface to manipulating lots of data files stored within a directory structure on disc.

    By default, the members of the DataFolder are instances of :class:`Stoner.Data`. The DataFolder emplys a lazy
    open strategy, so that files are only read in from disc when actually needed.

    .. inheritance-diagram:: DataFolder

    """

    def __init__(self, *args, **kargs):
        self.type = kargs.pop("type", make_Data(None))
        super().__init__(*args, **kargs)


class PlotFolder(PlotMethodsMixin, DataFolder):
    """A :py:class:`Stoner.folders.baseFolder` that knows how to ploth its underlying data files."""
