# -*- coding: utf-8 -*-
"""Folder functions for binding as methods to BaseFolder."""
from os import path

from numpy import any as np_any
from numpy import append, array

from ..tools import isiterable


def concatenate(folder, sort=None, reverse=False):
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
    for d in folder[1:]:
        folder[0] += d
    del folder[1:]

    if not isinstance(sort, bool) or sort:
        if isinstance(sort, bool) or sort is None:
            sort = folder[0].setas.cols["xcol"]
        folder[0].sort(order=sort, reverse=reverse)

    return folder


def extract(folder, *metadata, **kargs):
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
        if isinstance(m, str):
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
                if not isiterable(test) or isinstance(test, str):
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
        ret = folder.clone
    else:
        ret = folder

    return ret.walk_groups(_extractor, group=True, replace_terminal=True, walker_args={"metadata": metadata})


def gather(folder, xcol=None, ycol=None):
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

    return folder.walk_groups(_gatherer, group=True, replace_terminal=True, walker_args={"xcol": xcol, "ycol": ycol})
