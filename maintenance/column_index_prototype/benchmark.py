"""Paired warm medians and separately traced allocations for CS4 ownership.

Both designs use the actual CS3 owner algorithms. Table setup, imports, warm-up
and tracing are outside timed samples. Order alternates between pairs. Results
are process-local observations, not a portable speed guarantee.
"""

import argparse
from copy import deepcopy
import gc
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import platform
import re
import statistics
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd
from packaging.version import Version

from Stoner.core.column_schema import _assign_roles
from Stoner.core.storage_owner import DataOwner
from .model import IndexOwner, fixture

ROOT = Path(__file__).resolve().parents[2]


def source_hashes():
    """Fingerprint working sources; HEAD predates the uncommitted CS3 baseline."""
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [ROOT / "Stoner/core" / name for name in
              ("column_schema.py", "storage.py", "storage_owner.py", "setas.py", "storage_bridge.py")]
    result = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    for name in ("_simple_new", "_view", "_getitem_slice", "_shallow_copy", "take", "copy", "_concat"):
        result[f"pandas.Index.{name}"] = hashlib.sha256(inspect.getsource(getattr(pd.Index, name)).encode()).hexdigest()
    return result


def environment():
    """Check all 39 declared Conda specs, with explicit isolated overlay evidence."""
    records = {}
    for path in (Path(sys.prefix) / "conda-meta").glob("*.json"):
        record = json.loads(path.read_text())
        records[record["name"]] = record["version"]
    # The Python 3.12 lower-bound run overlays pandas, pytest-timeout and pytz.
    # Report both installed Conda records and effective distribution versions.
    effective = dict(records)
    overrides = {"pandas": pd.__version__, "pytest-timeout": version("pytest-timeout")}
    effective.update(overrides)
    dependencies = []
    for line in (ROOT / "tests/test-env.yml").read_text().split("dependencies:", 1)[1].splitlines():
        match = re.match(r"\s+- (?:[\w-]+::)?([\w-]+)(?:\s+>=([\d.]+))?\s*$", line)
        if match:
            name, minimum = match.groups()
            actual = effective.get(name)
            dependencies.append(dict(name=name, conda=records.get(name), effective=actual,
                                     minimum=minimum, satisfied=actual is not None and
                                     (minimum is None or Version(actual) >= Version(minimum))))
    if len(dependencies) != 39 or not all(item["satisfied"] for item in dependencies):
        raise RuntimeError(f"Incomplete test dependencies: {dependencies}")
    import pytest_timeout
    return dict(python=sys.version, executable=sys.executable, prefix=sys.prefix,
                platform=platform.platform(), pandas=pd.__version__, numpy=np.__version__,
                pandas_module=pd.__file__, timeout_module=pytest_timeout.__file__,
                pytz=version("pytz") if Version(pd.__version__) < Version("3") else None,
                dependencies=dependencies)


def prepare(owner_type, width, rows):
    owner = owner_type(fixture(width, rows))
    pattern = re.compile("Signal|Error")

    def role_edit():
        schema = owner._state.schema
        owner.roles[:] = _assign_roles(schema.headers, schema.roles, "Signal", "y")

    def rename():
        owner.headers[0] = "Field (T)"

    def resolve(selector):
        return (owner._state.values.columns.find_col(selector) if owner_type is IndexOwner
                else owner._state.schema.resolve(selector))

    return {
        "exact": lambda: resolve("Signal"),
        "substring": lambda: resolve("Sig"),
        "glob": lambda: resolve("Sign?l"),
        "regex": lambda: resolve(pattern),
        "setas_read": lambda: owner._state.schema.groups(owner._state.values.shape),
        "setas_edit": role_edit,
        "rename": rename,
        "reorder": lambda: owner.select(columns=list(range(width - 1, -1, -1))),
        "clone": lambda: deepcopy(owner),
        "repeat_selection": lambda: owner.select(columns=[1, 1, 0]),
    }


def measure(width, rows, repeats, iterations):
    """Alternate paired measurements, excluding fixture setup from both designs."""
    owners = {"composed": DataOwner, "index": IndexOwner}
    operations = {key: prepare(cls, width, rows) for key, cls in owners.items()}
    result = []
    for workload in operations["composed"]:
        count = 1 if workload in {"reorder", "clone", "repeat_selection"} else iterations
        samples = {key: [] for key in owners}
        for key in owners:
            operations[key][workload]()
        for repeat in range(repeats):
            for key in (list(owners) if repeat % 2 == 0 else list(reversed(owners))):
                operation = operations[key][workload]
                gc.collect()
                start = time.perf_counter_ns()
                for _ in range(count):
                    value = operation()
                samples[key].append((time.perf_counter_ns() - start) / count / 1e9)
                del value
        pair = {}
        for key in owners:
            # Separate fresh fixture for allocation measurement; one operation.
            operation = prepare(owners[key], width, rows)[workload]
            gc.collect()
            tracemalloc.start()
            value = operation()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            del value
            pair[key] = dict(seconds=samples[key], median_seconds=statistics.median(samples[key]),
                             traced_peak_bytes=peak)
        result.append(dict(width=width, rows=rows, workload=workload, iterations=count, **pair,
                           index_over_composed=pair["index"]["median_seconds"] /
                           pair["composed"]["median_seconds"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    report = dict(environment=environment(), sources=source_hashes(), repeats=args.repeats,
                  policy="warm medians; alternating paired order; separate one-operation tracemalloc",
                  results=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for width, rows in [(8, 4096), (512, 64)]:
        report["results"] += measure(width, rows, args.repeats, args.iterations)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
