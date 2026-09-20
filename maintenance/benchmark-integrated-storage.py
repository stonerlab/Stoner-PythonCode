"""Compare production storage with an isolated pre-migration checkout.

Reuse the Stage 3 workload definitions and measurement protocol, but run the
public Stoner API from each checkout in separate processes. No prototype results
are used as production evidence. Output belongs under maintenance/runs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ("construct", "load_fixture", "column", "fit", "append_rows", "convert",
             "stack_insert", "image_reduce", "stack_convert", "frame_edit")


def fingerprint(root):
    """Hash the measured production sources independently of Git status."""
    digest = hashlib.sha256()
    for path in sorted((root / "Stoner").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def main():
    """Measure paired public API workloads and report the agreed thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=WORKLOADS)
    parser.add_argument("--sizes", nargs="+", choices=("small", "large"), default=("small", "large"))
    args = parser.parse_args()
    legacy = args.legacy.resolve()
    if not (legacy / "Stoner/core/array.py").is_file() or (ROOT / "Stoner/core/array.py").exists():
        parser.error("Expected distinct legacy-array and migrated production checkouts")
    roots = {"legacy": legacy, "production": ROOT}
    hashes = {name: fingerprint(root) for name, root in roots.items()}
    report = {"sources": hashes, "measurements": [], "comparisons": [], "repeats": args.repeats}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MPLBACKEND="Agg", QT_QPA_PLATFORM="offscreen")
    # Import Stoner from the selected checkout before importing the shared harness.
    worker = (
        "import sys,json; sys.path[:0]=[sys.argv[1],sys.argv[2]]; "
        "import Stoner; from maintenance.storage_prototype.benchmark import measure,environment; "
        "result=measure(sys.argv[3],'legacy',sys.argv[4],int(sys.argv[5])); "
        "result.update(import_path=Stoner.__file__,environment=environment()); print(json.dumps(result))"
    )
    for size in args.sizes:
        for workload in args.workloads:
            if size == "large" and workload == "load_fixture":
                continue
            pair = {}
            for backend, root in roots.items():
                result = subprocess.run(
                    [sys.executable, "-c", worker, str(root), str(ROOT), workload, size, str(args.repeats)],
                    cwd=root, env=env, capture_output=True, text=True,
                )
                if result.returncode:
                    raise RuntimeError(f"{backend} {size} {workload} failed ({result.returncode}): {result.stderr}")
                item = json.loads(result.stdout.splitlines()[-1])
                if Path(item["import_path"]).resolve() != root / "Stoner/__init__.py":
                    raise RuntimeError("Worker imported the wrong checkout")
                if fingerprint(root) != hashes[backend]:
                    raise RuntimeError("Production sources changed during measurements")
                item.update(backend=backend, worker_stderr=result.stderr)
                pair[backend] = item
                report["measurements"].append(item)
            old, new = pair["legacy"], pair["production"]
            dimensions = old["dimensions"]
            if dimensions != new["dimensions"]:
                raise RuntimeError("Paired workload dimensions differ")
            if "rows" in dimensions:
                payload = (dimensions["rows"] + dimensions.get("appends", 0)) * dimensions["columns"] * 9
            elif "frames" in dimensions:
                payload = (dimensions["frames"] + (workload == "stack_insert")) * dimensions["height"] * dimensions["width"] * 3
            else:
                payload = 0
            time_limit = max(2 * old["median_seconds"], old["median_seconds"] + 0.005)
            memory_limit = max(2 * old["traced_peak_bytes"], 3 * payload + 1024**2)
            passed = new["median_seconds"] <= time_limit and new["traced_peak_bytes"] <= memory_limit
            report["comparisons"].append(dict(size=size, workload=workload, passed=passed,
                                              time_limit=time_limit, memory_limit=memory_limit))
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"{size} {workload}: {'PASS' if passed else 'FAIL'} "
                  f"{old['median_seconds'] * 1000:.3f} -> {new['median_seconds'] * 1000:.3f} ms; "
                  f"{old['traced_peak_bytes']} -> {new['traced_peak_bytes']} bytes", flush=True)
    return 0 if all(item["passed"] for item in report["comparisons"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
