"""Measure native reads and operation selections without building distributions."""

import argparse
import json
import hashlib
from pathlib import Path
import platform
import sys
from statistics import median
import time
import tracemalloc

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Stoner

from Stoner import Data, ImageFile, ImageStack
from Stoner.core.selection import selected_image_values


def measure(function, repeats=5):
    """Record warm median time and a separate traced allocation peak."""
    function()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    tracemalloc.start()
    result = function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return dict(median_seconds=median(times), traced_peak_bytes=peak)


def main():
    """Compare native access allocation with explicit masked numerical exports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if Path(Stoner.__file__).resolve() != root / 'Stoner' / '__init__.py':
        raise RuntimeError('Native-interface benchmarks must import this source checkout')
    data = Data(np.arange(400000, dtype=float).reshape(100000, 4))
    image = ImageFile(np.ones((512, 512), dtype=np.int16))
    stack = ImageStack([ImageFile(np.ones((256, 256), dtype=np.int16)) for _ in range(4)])
    cases = [('table', lambda: data.data, data.to_numpy),
             ('image', lambda: image.image, image.to_numpy),
             ('stack', lambda: stack.stack, stack.to_numpy)]
    results = {}
    for name, native, export in cases:
        native_result, export_result = measure(native), measure(export)
        results[name] = dict(native=native_result, masked_export=export_result,
                             allocation_pass=native_result['traced_peak_bytes'] < export_result['traced_peak_bytes'])
    region = np.zeros(image.shape, dtype=bool)
    region[:128, :128] = True
    results['image_selection'] = measure(lambda: selected_image_values(image.image, region))
    results['image_selected_mean'] = measure(lambda: image.mean(where=region))
    digest = hashlib.sha256()
    for path in sorted((root / 'Stoner').rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode() + b'\0')
        digest.update(path.read_bytes().replace(b'\r\n', b'\n') + b'\0')
    results['environment'] = dict(python=platform.python_version(), numpy=np.__version__,
                                  pandas=pd.__version__, xarray=xr.__version__, source_sha256=digest.hexdigest())
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
    return 0 if all(results[name]['allocation_pass'] for name, _, _ in cases) else 1


if __name__ == '__main__':
    raise SystemExit(main())
