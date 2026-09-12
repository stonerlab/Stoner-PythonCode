"""Reproduce loader exception-boundary findings without modifying runtime source."""
import importlib
import json
import logging
from pathlib import Path
import sys
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Stoner import Data
from Stoner.core.exceptions import StonerLoadError
from Stoner.formats.data.instruments import load_spc
from Stoner.formats.data.zip import load_zipfile
from Stoner.tools import file as file_tools

outdir = ROOT / 'maintenance/runs/exception-audit'
outdir.mkdir(exist_ok=True)
report = {}
for name in ('Stoner.formats.data.generic', 'Stoner.formats.image.generic'):
    mod = importlib.import_module(name)
    stdout, stderr = sys.stdout, sys.stderr
    logger = logging.getLogger('hyperspy.io')
    filters = list(logger.filters)
    try:
        try:
            with mod.catch_sysout():
                raise StonerLoadError('intentional candidate rejection')
        except StonerLoadError:
            pass
        report[name] = {'stdout_restored': sys.stdout is stdout, 'stderr_restored': sys.stderr is stderr,
                        'extra_logging_filters': len(logger.filters) - len(filters)}
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        logger.filters[:] = filters

short_spc = outdir / 'truncated.spc'
short_spc.write_bytes((ROOT / 'sample-data/Raman.spc').read_bytes()[:32])
bad_zip = outdir / 'invalid-member.zip'
with zipfile.ZipFile(bad_zip, 'w') as archive:
    archive.writestr('invalid.txt', 'Not TDI\n')

for label, loader, filename in [('truncated_spc', load_spc, short_spc), ('unsupported_zip_member', load_zipfile, bad_zip)]:
    reached = []
    def candidate(data, path, *args, **kwargs):
        return loader(data, path, *args, **kwargs)
    candidate.name = label
    def next_candidate(data, path, *args, **kwargs):
        reached.append(True)
        return data
    next_candidate.name = 'audit_sentinel'
    opened = []
    class TrackedZip(zipfile.ZipFile):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)
    try:
        with patch.object(file_tools, 'next_filer', return_value=iter([candidate, next_candidate])), \
             patch.object(file_tools, 'get_mime_type', return_value=None), \
             patch.object(zipfile, 'ZipFile', TrackedZip):
            try:
                file_tools.auto_load_classes(filename, 'Data')
                error = None
            except Exception as exc:
                error = {'type': type(exc).__name__, 'message': str(exc), 'fallback_signal': isinstance(exc, StonerLoadError)}
            report[label] = {'error': error, 'next_candidate_reached': bool(reached),
                             'owned_zip_handles_left_open': sum(z.fp is not None for z in opened)}
    finally:
        for archive in opened:
            archive.close()

empty_zip = outdir / 'empty.zip'
with zipfile.ZipFile(empty_zip, 'w'):
    pass
with zipfile.ZipFile(empty_zip, 'r') as borrowed:
    try:
        load_zipfile(Data(), borrowed)
    except Exception as exc:
        report['borrowed_empty_zip'] = {'exception': type(exc).__name__, 'caller_handle_closed': borrowed.fp is None}

# Establish that the dispatcher itself honours the intended signal.
def rejection(data, filename, *args, **kwargs):
    raise StonerLoadError('expected rejection')
rejection.name = 'expected_rejection'
with patch.object(file_tools, 'next_filer', return_value=iter([rejection, next_candidate])), \
     patch.object(file_tools, 'get_mime_type', return_value=None):
    report['dispatcher_control'] = file_tools.auto_load_classes(short_spc, 'Data')['Loaded as']
report['optional_dependencies'] = {}
for name in ('Stoner.formats.data.facilities', 'Stoner.formats.image.facilities', 'Stoner.formats.image.generic'):
    mod = importlib.import_module(name)
    report['optional_dependencies'][name] = {key: getattr(mod, key, None) is not None for key in ('fabio', 'rsciio') if hasattr(mod, key)}
(output := outdir / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(output)
print(json.dumps(report, indent=2))
