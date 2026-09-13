"""Assess paired measurements against proposed, reviewable migration tolerances.

Run with the benchmark JSON path. This verifies evidence completeness and source
identity; passing proposed thresholds does not imply maintainer agreement to them.
"""

import argparse
import hashlib
import json
from pathlib import Path


def assess(path):
    """Print paired timing/memory evidence and return whether thresholds pass."""
    report = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in report["source_hashes"].items():
        actual = hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Measured source differs: {name}")
    pairs = {}
    for item in report["measurements"]:
        if item["source_hashes"] != report["source_hashes"]:
            raise ValueError("Mixed source revisions")
        pair = pairs.setdefault((item["size"], item["workload"]), {})
        if item["backend"] in pair:
            raise ValueError("Duplicate measurement")
        pair[item["backend"]] = item
    if len(pairs) != 19 or any(set(pair) != {"legacy", "prototype"} for pair in pairs.values()):
        raise ValueError("Incomplete paired benchmark")
    passed = True
    print("| Size  | Workload      | Legacy ms | Proto ms | Time ratio | Legacy MiB | Proto MiB | Gate |")
    print("| ----- | ------------- | --------- | -------- | ---------- | ---------- | --------- | ---- |")
    for (size, workload), pair in pairs.items():
        legacy, prototype = pair["legacy"], pair["prototype"]
        dimensions = legacy["dimensions"]
        if prototype["dimensions"] != dimensions:
            raise ValueError("Different paired inputs")
        if "rows" in dimensions:
            payload = (dimensions["rows"] + dimensions.get("appends", 0)) * dimensions["columns"] * 9
        elif "frames" in dimensions:
            frames = dimensions["frames"] + (workload == "stack_insert")
            payload = frames * dimensions["height"] * dimensions["width"] * 3
        else:
            payload = 0
        old_time, new_time = legacy["median_seconds"], prototype["median_seconds"]
        old_memory, new_memory = legacy["traced_peak_bytes"], prototype["traced_peak_bytes"]
        time_ok = new_time <= max(old_time * 2, old_time + 0.005)
        memory_ok = new_memory <= max(old_memory * 2, payload * 3 + 1024**2)
        passed &= time_ok and memory_ok
        gate = "pass" if time_ok and memory_ok else "FAIL"
        print(f"| {size:5} | {workload:13} | {old_time*1000:9.3f} | {new_time*1000:8.3f} | "
              f"{new_time/old_time:10.2f} | {old_memory/1024**2:10.3f} | {new_memory/1024**2:9.3f} | {gate:4} |")
    print("Timing limit: max(2 x legacy median, legacy median + 5 ms).")
    print("Traced allocation limit: max(2 x legacy peak, 3 x input/output payload + 1 MiB).")
    print("RSS remains diagnostic because sampling and allocator reuse limit comparability.")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    arguments = parser.parse_args()
    raise SystemExit(0 if assess(arguments.report) else 1)
