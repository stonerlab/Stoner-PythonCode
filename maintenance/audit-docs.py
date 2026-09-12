"""Report Sphinx warning categories and coverage of the primary public API."""

import argparse
from collections import Counter
import inspect
import json
import importlib
from pathlib import Path
import re
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Stoner


def inventory_objects(build):
    """Read Python object names from a local Sphinx inventory."""
    with (build / "objects.inv").open("rb") as stream:
        header = [stream.readline() for _ in range(4)]
        if not header[0].startswith(b"# Sphinx inventory version 2"):
            raise ValueError("Expected a Sphinx version 2 inventory")
        entries = zlib.decompress(stream.read()).decode().splitlines()
    return {line.split()[0] for line in entries if " py:" in line}


def audit(build, warnings, baseline=None):
    """Compare the built inventory with the runtime API and classify warnings."""
    objects = inventory_objects(build)
    primary = {}
    for name in ("Data", "DataFolder", "ImageFile", "ImageFolder", "ImageStack"):
        cls = getattr(Stoner, name)
        canonical = f"{cls.__module__}.{cls.__name__}"
        primary[name] = {"canonical": canonical, "documented": canonical in objects}
    modules = {
        "Stoner.core.methods", "Stoner.analysis.columns", "Stoner.analysis.features",
        "Stoner.analysis.filtering", "Stoner.analysis.functions", "Stoner.analysis.fitting.functions",
        "Stoner.plot.functions",
    }
    dynamic = sorted(
        name for name, member in inspect.getmembers(Stoner.Data, inspect.isfunction)
        if not name.startswith("_") and member.__module__ in modules
    )
    missing = [name for name in dynamic if f"Stoner.core.data.Data.{name}" not in objects]
    fitting_functions = []
    for suffix in ("generic", "thermal", "tunnelling", "e_transport", "magnetism", "superconductivity"):
        module = importlib.import_module(f"Stoner.analysis.fitting.models.{suffix}")
        fitting_functions.extend(
            f"{module.__name__}.{name}" for name in module.__all__
            if inspect.isfunction(getattr(module, name))
        )
    messages = [line for line in warnings.read_text(encoding="utf-8").splitlines()
                if re.search(r"\b(?:WARNING|ERROR|CRITICAL):", line)]
    categories = Counter()
    untagged = Counter()
    for line in messages:
        tag = re.search(r"\[([\w.]+)\]\s*$", line)
        categories[tag.group(1) if tag else "untagged"] += 1
        if tag is None:
            if "stub file not found" in line:
                untagged["missing_autosummary_stub"] += 1
            elif "duplicate object description" in line:
                untagged["duplicate_object_description"] += 1
            else:
                untagged["other"] += 1
    return {
        "python": sys.version, "stoner_source": Stoner.__file__,
        "warning_messages": len(messages), "warning_categories": dict(categories.most_common()),
        "untagged_warning_causes": dict(untagged.most_common()),
        "primary_classes": primary, "dynamic_data_methods": dynamic,
        "missing_dynamic_data_methods": missing,
        "fitting_functions": sorted(fitting_functions),
        "missing_fitting_functions": sorted(set(fitting_functions) - objects),
        "removed_api_objects": sorted(inventory_objects(baseline) - objects) if baseline else [],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("warnings", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compare", type=Path, help="Baseline HTML build to check for lost API entries")
    parser.add_argument("--require-fitting-functions", action="store_true",
                        help="Require all exported fitting functions (use a case-sensitive filesystem)")
    parser.add_argument("--allow-removed", type=Path,
                        help="JSON file listing explicitly reviewed obsolete inventory names under objects")
    args = parser.parse_args()
    result = audit(args.build, args.warnings, args.compare)
    allowed = set()
    if args.allow_removed:
        if not args.compare:
            parser.error("--allow-removed requires --compare")
        allowed = set(json.loads(args.allow_removed.read_text(encoding="utf-8"))["objects"])
    result["reviewed_obsolete_api_objects"] = sorted(allowed.intersection(result["removed_api_objects"]))
    result["unexpected_removed_api_objects"] = sorted(set(result["removed_api_objects"]) - allowed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))
    if (result["missing_dynamic_data_methods"] or result["unexpected_removed_api_objects"]
            or (args.require_fitting_functions and result["missing_fitting_functions"])
            or not all(item["documented"] for item in result["primary_classes"].values())):
        sys.exit(1)
