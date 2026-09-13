"""Measure paired storage workloads in fresh processes with inspectable JSON.

Run as ``python -m maintenance.storage_prototype.benchmark --output PATH``.
Timing excludes fixture construction, imports and memory instrumentation. Memory
is measured separately with tracemalloc and sampled process RSS, not inferred
from container sizes. No production backend is changed by this tool.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
import tracemalloc

import numpy as np
import pandas as pd
import psutil
import scipy
from scipy.optimize import curve_fit
import xarray as xr

from Stoner import Data
from Stoner.Image import ImageStack
from .model import Stack, Table

ROOT = Path(__file__).resolve().parents[2]
WORKLOADS = ("construct", "load_fixture", "column", "fit", "append_rows", "convert",
             "stack_insert", "image_reduce", "stack_convert", "frame_edit")


def source_hashes():
    """Identify the exact prototype and benchmark sources used by each worker."""
    return {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("model.py", "benchmark.py")}


def environment():
    """Record runtime versions and verify every declared Conda test dependency."""
    from packaging.version import Version
    import re

    records = {}
    for path in (Path(sys.prefix) / "conda-meta").glob("*.json"):
        record = json.loads(path.read_text())
        records[record["name"]] = record["version"]
    dependencies = []
    for line in (ROOT / "tests/test-env.yml").read_text().split("dependencies:", 1)[1].splitlines():
        match = re.match(r"\s+- (?:[\w-]+::)?([\w-]+)(?:\s+>=([\d.]+))?\s*$", line)
        if not match:
            continue
        name, minimum = match.groups()
        actual = records.get(name)
        satisfied = actual is not None and (minimum is None or Version(actual) >= Version(minimum))
        dependencies.append(dict(name=name, minimum=minimum, installed=actual, satisfied=satisfied))
    if not dependencies or not all(item["satisfied"] for item in dependencies):
        raise RuntimeError(f"Incomplete test environment: {dependencies}")
    return dict(python=sys.version, platform=platform.platform(), executable=sys.executable,
                versions={"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                          "xarray": xr.__version__, "psutil": psutil.__version__},
                test_dependencies=dependencies)


def polynomial(x, a, b, c):
    """Supply the same numerical fitting work to both storage adapters."""
    return a * x**2 + b * x + c


def prepare(workload, backend, size):
    """Return a fresh callable and its explicit workload dimensions."""
    rows = 10_000 if size == "small" else 100_000
    rng = np.random.default_rng(1701)
    values = rng.normal(size=(rows, 8))
    values[:, 0] = np.linspace(-2, 2, rows)
    values[:, 1] = polynomial(values[:, 0], 0.1, 0.3, 2) + rng.normal(0, 0.01, rows)
    values[:, 2] = 0.01
    legacy = backend == "legacy"
    make_table = (lambda a: Data(a, setas="xyey....")) if legacy else (lambda a: Table(a, roles="xyey...."))
    dimensions = dict(rows=rows, columns=8)
    if workload == "construct":
        return lambda: make_table(values), dimensions
    if workload == "load_fixture":
        path = ROOT / "tests/Stoner/CoreTest.dat"
        def load():
            loaded = Data(path, setas="xy")
            return loaded if legacy else Table(loaded.data.data, loaded.column_headers, "xy", np.ma.getmaskarray(loaded.data),
                                               loaded.metadata)
        return load, dict(fixture=str(path.relative_to(ROOT)), bridge="existing Stoner loader")
    if workload in ("stack_insert", "image_reduce", "stack_convert", "frame_edit"):
        frames, side = (8, 128) if size == "small" else (16, 512)
        images = rng.integers(0, 4096, (frames, side, side), dtype=np.uint16)
        stack = ImageStack(images) if legacy else Stack(images)
        dimensions = dict(frames=frames, height=side, width=side, dtype="uint16")
        if workload == "stack_insert":
            return lambda: stack.insert(0, images[0]), dimensions
        if workload == "image_reduce":
            return (lambda: stack.imarray.mean()) if legacy else (lambda: stack.to_numpy().mean()), dimensions
        if workload == "frame_edit":
            def edit():
                stack[0][0, 0] = 12
            return edit, dimensions
        return (lambda: np.ma.array(stack.imarray, copy=True)) if legacy else stack.to_numpy, dimensions
    table = make_table(values)
    if workload == "column":
        return (lambda: table.column(3).copy()) if legacy else (lambda: table.column(3)), dimensions
    if workload == "convert":
        return (lambda: np.ma.array(table.data, copy=True)) if legacy else table.to_numpy, dimensions
    if workload == "fit":
        def fit():
            array = np.ma.array(table.data, copy=True) if legacy else table.to_numpy()
            valid = ~np.ma.getmaskarray(array[:, :3]).any(axis=1)
            return curve_fit(polynomial, array.data[valid, 0], array.data[valid, 1],
                             sigma=array.data[valid, 2], absolute_sigma=True)
        return fit, dimensions
    if workload == "append_rows":
        def append():
            for row in values[:25]:
                if legacy:
                    table.__iadd__(row)
                else:
                    table.append(row)
        dimensions["appends"] = 25
        return append, dimensions
    raise ValueError(workload)


def measure(workload, backend, size, repeats):
    """Measure warmed wall time and separately instrument one fresh operation."""
    operation, dimensions = prepare(workload, backend, size)
    operation()
    durations = []
    for _ in range(repeats):
        operation, _ = prepare(workload, backend, size)
        gc.collect()
        start = time.perf_counter()
        result = operation()
        durations.append(time.perf_counter() - start)
        del result, operation
    operation, _ = prepare(workload, backend, size)
    gc.collect()
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    peak = [initial_rss]
    done = threading.Event()
    def sample():
        while not done.wait(0.001):
            peak[0] = max(peak[0], process.memory_info().rss)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    tracemalloc.start()
    try:
        result = operation()
        _, allocated_peak = tracemalloc.get_traced_memory()
        peak[0] = max(peak[0], process.memory_info().rss)
    finally:
        tracemalloc.stop()
        done.set()
        sampler.join()
    return dict(workload=workload, backend=backend, size=size, dimensions=dimensions,
                source_hashes=source_hashes(),
                seconds=durations, median_seconds=statistics.median(durations),
                traced_peak_bytes=allocated_peak, sampled_peak_rss_bytes=peak[0],
                sampled_rss_increase_bytes=peak[0] - initial_rss)


def main():
    """Run fresh-process pairs and save partial evidence after every result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--worker", nargs=3, metavar=("WORKLOAD", "BACKEND", "SIZE"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(measure(*args.worker, args.repeats)))
        return
    if args.output is None:
        parser.error("--output is required")
    report = dict(environment=environment(), repeats=args.repeats, measurements=[], source_hashes=source_hashes(),
                  methodology="Fresh process per pair member; untimed setup; warmup; separate memory pass; 1ms RSS sampling")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ, MPLBACKEND="Agg", QT_QPA_PLATFORM="offscreen")
    for size in ("small", "large"):
        for workload in WORKLOADS:
            if workload == "load_fixture" and size == "large":
                continue
            for backend in ("legacy", "prototype"):
                completed = subprocess.run([sys.executable, "-m", "maintenance.storage_prototype.benchmark",
                                            "--worker", workload, backend, size, "--repeats", str(args.repeats)],
                                           capture_output=True, text=True, check=True, env=child_env, cwd=ROOT)
                measurement = json.loads(completed.stdout.splitlines()[-1])
                measurement["worker_stderr"] = completed.stderr
                if measurement["source_hashes"] != report["source_hashes"]:
                    raise RuntimeError("Sources changed while benchmarking; rerun on unchanged files")
                report["measurements"].append(measurement)
                args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"{size} {workload} {backend}: {report['measurements'][-1]['median_seconds']:.6f}s", flush=True)


if __name__ == "__main__":
    main()
