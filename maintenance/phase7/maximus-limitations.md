# MAXIMUS multi-region loading: known unimplemented feature

Decision recorded 2026-09-12: retain existing loader behaviour. The maintainer
requires definitive facility format information and representative actual data
before implementing multi-region support. This review resolves the maintenance
triage item; it does not implement or claim support for the missing feature.

## Current behaviour

Four former FIXME sites select `ScanDefinition["Regions"][0]`:

- `Stoner/formats/utils/maximus.py`: `read_images` and `read_pointscan`, used by
  the registered data and image loaders.
- `Stoner/formats/maximus.py`: `_read_images` and `_read_pointscan`, in the
  module that also provides `MaximusStack`.

Image readers take the spatial axes from the first region's PAxis and QAxis;
point-scan readers take PAxis from that region. Stacked results use StackAxis
from ScanDefinition. The readers collect matching .xim or .xsp files without
implementing a region-to-file mapping. Selecting the first region does not
establish correct support for a multi-region scan, nor does the code explicitly
reject such inputs. Do not describe multi-region data as safely handled.

## Available real-data evidence

Parsed all three checked-in .hdr files with the current hdr_to_dict helper:

- `sample-data/MPI_210127006.hdr`: NEXAFS Point Scan, one region, with
  `MPI_210127006_0.xsp`.
- `sample-data/maximus_scan/MPI_210127019.hdr`: Image Scan, one region, with
  `MPI_210127019_a.xim`.
- `sample-data/maximus_scan/MPI_210127021/MPI_210127021.hdr`: NEXAFS Image Scan,
  one region, with `MPI_210127021_a000.xim` and `MPI_210127021_a001.xim`.

Multiple stack files are therefore not evidence of multiple regions.
The existing MAXIMUS image and stack tests in tests/Stoner/test_FileFormats.py
exercise the retained examples, including an HDF5 stack round trip. No actual
multi-region fixture or definitive description of its mapping was established
by this repository review. No claim about an external facility specification
is made.

## Evidence required to reopen implementation

- An authoritative format description or confirmation from the facility or
  acquisition-software maintainers covering region selection, file naming,
  ordering, axes and stack semantics.
- Complete actual multi-region exports, including headers and associated data
  files for the scan types to be supported, with permission to retain regression
  fixtures.
- Expected region contents, dimensions, coordinates and metadata, so tests can
  check scientific interpretation as well as successful parsing.

Until then, retain this as a known unimplemented feature. Do not infer a mapping
from synthetic examples, silently merge regions, or change selection/rejection
behaviour as routine cleanup. Existing single-region fixtures must be preserved.

## Validation of this review

Only comments and maintenance documentation changed. AST comparison confirms
both package modules are executable-equivalent to the preceding commit. The
three real headers were parsed to verify region counts; runtime tests and Sphinx
were not rerun for comment-only changes.
