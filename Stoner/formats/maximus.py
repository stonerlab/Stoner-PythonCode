# -*- coding: utf-8 -*-
"""Support files from the Bessy Maximus instrument."""

import re
from os import path
from pathlib import Path
from warnings import warn

import h5py

from ..compat import string_types

# Imports for use in Stoner package
from ..core.exceptions import StonerLoadError
from ..Image import ImageFile
from ..Image.stack import ImageStack
from ..Image.scan_metadata import ScanMetadataMixin
from ..tools.file import HDFFileManager, file_dialog, get_filename
from .utils.scan import (
    parse_scan_args, raise_scan_error as _raise_error, write_scan_frame,
    initialise_scan_metadata, read_scan_frame, write_common_metadata, read_common_metadata, read_scan_groups,
)

from .utils import maximus as _scan_io

# Preserve historical import paths while sharing the reviewed scan parser.
hdr_to_dict = _scan_io.hdr_to_dict
read_scan = _scan_io.read_scan
_flatten_header = _scan_io.flatten_header
_process_key = _scan_io.process_key
_read_images = _scan_io.read_images
_read_pointscan = _scan_io.read_pointscan


SCAN_NO = re.compile(r"MPI_(\d+)")


class MaximusStack(ScanMetadataMixin, ImageStack):
    """Process an image scan stack from the Bessy Maximus beamline as an ImageStack subclass."""

    _defaults = {"type": ImageFile, "pattern": "*.hdr"}

    def __init__(self, *args, **kwargs):
        """Construct the attocube subclass of ImageStack."""
        args, root_name, scan = parse_scan_args(args, kwargs, SCAN_NO, "MPI_{:039d}", path_type=Path)

        super().__init__(*args, **kwargs)

        initialise_scan_metadata(self, scan)

        if root_name:
            self._load(root_name)

        self.compression = "gzip"
        self.compression_opts = 6

    def _load(self, *args, **kwargs):
        """Load an ImageStack from either an hdf file or textfiles."""
        filename, args, kwargs = get_filename(args, kwargs)
        self.filename = filename
        pth = Path(self.filename)
        if h5py.is_hdf5(self.filename):
            return self.__class__.read_hdf5(self.filename)

        if pth.suffix != ".hdr":  # Passed a .xim or .xsp file in instead of the hdr file.
            pth = pth.parent / f"MPI_{self.scan_no:09d}.hdr"
        stem = pth.parent / pth.stem

        try:
            hdr = _flatten_header(hdr_to_dict(pth))
            if hdr["ScanDefinition.Type"] != "NEXAFS Image Scan":
                raise StonerLoadError("Not an Maximus ImageStack")
        except (StonerLoadError, ValueError, TypeError, IOError) as err:
            raise StonerLoadError("Error loading as Maximus File") from err

        metadata, stack, _ = read_scan(stem)
        self._common_metadata.update(_flatten_header(metadata))
        for ix, point in enumerate(self._common_metadata["ScanDefinition.StackAxis.Points"]):
            image = ImageFile(stack[:, :, ix])
            image.filename = f"{stem}_a{ix:03d}"
            image.metadata[self._common_metadata["ScanDefinition.StackAxis.Name"]] = point
            self.append(image)
        return self

    # The published keyword is format; retain it for caller compatibility.
    def to_xarray(self, *, format="stack"):  # pylint: disable=redefined-builtin
        """Export storage or the detector channel on the recorded scan grid and energy axis."""
        if format == "stack":
            return super().to_xarray()
        if format != "channels":
            raise ValueError("format must be 'stack' or 'channels'")
        headers = self._common_metadata
        name = headers.get("ScanDefinition.Channels.Name")
        if not isinstance(name, str):
            raise ValueError("Channel export requires a single explicitly identified detector")
        dataset = self.export_storage().dataset
        for axis, header in (("y", "PAxis"), ("x", "QAxis")):
            key = f"ScanDefinition.Regions.{header}"
            points = headers[f"{key}.Points"]
            if len(points) != dataset.sizes[axis]:
                raise ValueError("The current grid differs from the scan header; use canonical storage coordinates")
            dataset = dataset.assign_coords({axis: points})
            dataset[axis].attrs = {"units": headers[f"{key}.Unit"], "long_name": headers[f"{key}.Name"]}
        energy_name = headers["ScanDefinition.StackAxis.Name"]
        values = [image.metadata[energy_name] for image in self]
        dataset = dataset.assign_coords({energy_name: ("frame", values)})
        dataset[energy_name].attrs["units"] = headers["ScanDefinition.StackAxis.Unit"]
        dataset = dataset.rename({"intensity": name, "excluded": f"{name}__excluded"})
        dataset[name].attrs = {
            "units": headers.get("ScanDefinition.Channels.Unit", ""),
            "excluded": f"{name}__excluded",
        }
        warn(
            "The native channel Dataset omits typed metadata; use export_storage for lossless interchange",
            UserWarning,
            stacklevel=2,
        )
        return dataset

    def __clone__(self, other=None, attrs_only=False):
        """Do whatever is necessary to copy attributes from self to other.

        Notes:
            We're in the base class here, so we don't call super() if we can't handle this, then we're stuffed!


        """
        other = super().__clone__(other, attrs_only)
        other._common_metadata = self._common_metadata.copy()
        return other

    def _read_image(self, g):
        """Read an image array and return a member of the image stack."""
        if "image" not in g:
            _raise_error(g.parent, message=f"{g.name} does not have a signal dataset !")
        tmp = self.type()  # pylint: disable=E1102
        return read_scan_frame(g, tmp, "image")

    def to_hdf5(self, filename=None):
        """Save the AttocubeScan to an hdf5 file."""
        mode = "r"
        if filename is None:
            filename = path.join(self.directory, f"MPI_{self.scan_no:09d}.hdf5")
        if isinstance(filename, Path):
            filename = str(filename)
        if filename is None or (isinstance(filename, bool) and not filename):  # now go and ask for one
            filename = file_dialog("w", None, "MaximusStack")
            self.filename = filename
        if isinstance(filename, string_types):
            mode = "r+" if path.exists(filename) else "w"
        with HDFFileManager(filename, mode) as f:
            f.attrs["type"] = type(self).__name__
            f.attrs["module"] = type(self).__module__
            f.attrs["scan_no"] = self.scan_no
            f.attrs["groups"] = list(self.groups.keys())
            f.attrs["names"] = self.__names__()
            write_common_metadata(f, self._common_metadata)

            for g, group in self.groups.items():  # Recurse to save groups
                grp = f.require_group(g)
                group.to_hdf5(grp)

            for ch in self.__names__():
                signal = f.require_group(ch)
                data = self[ch]
                write_scan_frame(signal, data, "image", compression=self.compression,
                                 compression_opts=self.compression_opts)
        return self

    @classmethod
    def read_hdf5(cls, *args, **kwargs):
        """Create a new instance from an hdf file."""
        self = cls(regrid=False)
        filename, args, kwargs = get_filename(args, kwargs)
        self.filename = filename
        with HDFFileManager(self.filename, "r") as f:
            self.scan_no = f.attrs["scan_no"]
            sub_grps = f.attrs.get("groups")
            if "names" in f.attrs:
                names = f.attrs["names"]
            else:
                names = []
            grps = read_common_metadata(self._common_metadata, f)
            if sub_grps is None:
                sub_grps = grps
            read_scan_groups(
                self, f, sub_grps, self._read_image, cls.read_hdf5, args, kwargs
            )
            for grp in names:
                g = f[grp]
                self.append(self._read_image(g))
        return self


if __name__ == "__main__":
    # Test by reading all files
    for infile in Path(".").glob("*.hdr"):
        hdr, data, dims = read_scan(infile.stem)
