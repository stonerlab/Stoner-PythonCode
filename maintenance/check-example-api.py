"""Inventory documentation examples and reject retired Stoner API usage.

This is a static check, not proof that every illustrative fragment can execute.
Run the sample/script tests as well. Reports retain syntax fragments and historical
inline migration references separately, so neither disappears from the inventory.
"""

import argparse
import ast
from dataclasses import asdict, dataclass
import doctest
import json
from pathlib import Path
import re
import textwrap


ROOT = Path(__file__).resolve().parents[1]
OLD_MODULES = {"Core", "Analysis", "Plot", "PlotFormats", "Fit", "FileFormats", "Folders", "compat"}
OLD_CLASSES = {"DataFile", "AnalysisFile", "AnalyseFile", "PlotFile", "DataArray", "ImageArray", "KerrArray"}
OLD_METHODS = {
    "plot_xy", "plot_xyz", "plot_xyuv", "plot_xyzuvw", "plot_voxels", "plot_matrix",
    "contour_xyz", "quiver_plot", "inset", "subplot", "subplot2grid", "nlfit", "mpfit",
    "apply_all", "crop_stack", "correct_drifts", "crop_image", "convert_float", "convert_int",
}
FIT_METHODS = {"curve_fit", "lmfit", "odr", "differential_evolution", "polyfit", "annotate_fit"}
HISTORICAL = {"doc/UserGuide/upgrade.rst", "doc/UserGuide/plot_migration.rst", "doc/release-notes.rst"}
HISTORICAL_LITERALS = {
    ("doc/development/plans/FIT_ACCESSOR_PLAN.md", "d.lmfit(Linear, xcol=0, ycol=1)"),
    ("doc/development/plans/FIT_ACCESSOR_PLAN.md", "Data.lmfit(d, ...)"),
    ("doc/UserGuide/fitting_registry.rst", "d.lmfit(Linear, xcol=0, ycol=1)"),
    ("doc/UserGuide/image_plotting.rst", "image.imshow()"),
    ("doc/UserGuide/plot_recipes.rst", "d.inset(...)"),
}


@dataclass
class Example:
    """Record one source example and its location for compatibility auditing."""

    path: str
    line: int
    kind: str
    code: str
    language: str = "python"


def markup_examples(text, path, offset=0):
    """Collect literal blocks, inline literals, doctests and fenced/TeX listings."""
    lines = text.splitlines()
    if path.endswith(".tex"):
        for match in re.finditer(r"\\begin\{(lstlisting|verbatim)\}(.*?)\\end\{\1\}", text, re.S):
            code = textwrap.dedent(match[2]).strip()
            code = re.sub(r"(?m)^\s*(?:>>>|\.\.\.) ?", "", code)
            start = match.start(2) + len(match[2]) - len(match[2].lstrip("\r\n"))
            yield Example(path, offset + text[:start].count("\n") + 1, "tex", code)
        return
    if path.endswith(".md"):
        for match in re.finditer(r"(?m)^```([^\n]*)\n(.*?)^```", text, re.S):
            yield Example(path, offset + text[:match.start()].count("\n") + 2,
                          "fence", match[2], match[1].strip() or "python")
    else:
        i = 0
        while i < len(lines):
            line = lines[i]
            directive = re.match(r"\s*\.\. (?:code-block|code|sourcecode)::\s*(\S*)", line)
            # A plot with a filename includes a separately inventoried sample.
            plot = re.match(r"\s*\.\. plot::\s*$", line)
            if directive or plot or (line.rstrip().endswith("::") and not line.lstrip().startswith(".. ")):
                base = len(line) - len(line.lstrip())
                start = i + 1
                while start < len(lines) and (
                    not lines[start].strip() or lines[start].lstrip().startswith(":")
                    or lines[start].strip() == "# doctest: +SKIP"
                ):
                    start += 1
                end = start
                while end < len(lines) and (not lines[end].strip()
                                            or len(lines[end]) - len(lines[end].lstrip()) > base):
                    end += 1
                if end > start:
                    code = textwrap.dedent("\n".join(lines[start:end]))
                    yield Example(path, offset + start + 1, "literal", code,
                                  directive[1] or "python" if directive else "python")
                    i = end - 1
            i += 1
    try:
        for example in doctest.DocTestParser().get_examples(text):
            yield Example(path, offset + example.lineno + 1, "doctest", example.source)
    except ValueError:
        pass  # Incomplete prompts are still retained as literal blocks above.
    pattern = r"``([^`\n]+)``" if not path.endswith(".md") else r"(?<!`)`([^`\n]+)`(?!`)"
    for match in re.finditer(pattern, text):
        if "(" in match[1] or "Stoner." in match[1]:
            yield Example(path, offset + text[:match.start()].count("\n") + 1, "inline", match[1])


def examples(root=ROOT):
    """Include current docs, samples, scripts and package docstrings."""
    for directory in ("doc", "scripts"):
        for path in sorted((root / directory).rglob("*")):
            if any(part in {"_build", "plot_cache", ".ipynb_checkpoints", "__pycache__"} for part in path.parts):
                continue
            relative = path.relative_to(root).as_posix()
            if path.suffix in {".rst", ".md", ".tex"}:
                yield from markup_examples(path.read_text(encoding="utf-8-sig"), relative)
            elif path.suffix == ".py" and (directory == "scripts" or "samples" in path.parts):
                yield Example(relative, 1, "script", path.read_text(encoding="utf-8-sig"))
            elif path.suffix == ".ipynb":
                for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
                    if cell["cell_type"] == "code":
                        yield Example(relative, 1, "notebook", "".join(cell["source"]))
    for path in sorted((root / "Stoner").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    yield from markup_examples(textwrap.dedent(doc), path.relative_to(root).as_posix(),
                                               node.body[0].lineno - 1)
    for path in (root / "README.rst", root / "README.md"):
        if path.exists():
            yield from markup_examples(path.read_text(encoding="utf-8-sig"), path.name)


def _attribute_violations(node):
    """Check retired attributes while distinguishing scientific and rendering owners."""
    found = []
    owner = ast.unparse(node.value)
    if node.attr in OLD_METHODS and not owner.startswith(("plt", "pyplot", "figure", "fig", "ax")):
        found.append((node.lineno, f"retired method {node.attr}"))
    if node.attr in {"_get_cols", "_cols", "_col_args"}:
        found.append((node.lineno, f"private compatibility helper {node.attr}"))
    if node.attr in FIT_METHODS and not (
        owner.endswith((".fit", ".plot"))
        or owner in {"np", "numpy", "sp", "scipy", "optimize", "odr"}
        or owner.startswith(("np.", "numpy.", "scipy.", "self._fitter"))
    ):
        found.append((node.lineno, f"direct fitting/annotation method {owner}.{node.attr}"))
    if node.attr == "imshow" and owner in {"im", "image", "binary", "xmcd", "strctural", "structural"}:
        found.append((node.lineno, "legacy image display; use image.plot()"))
    return found


def _call_violations(node):
    """Check dynamic method selection and retired accessor keyword conventions."""
    found = []
    if isinstance(node.func, ast.Name) and node.func.id == "getattr":
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) and node.args[1].value in FIT_METHODS:
            owner = ast.unparse(node.args[0])
            if not owner.endswith(".fit"):
                found.append((node.lineno, "dynamic direct fitting method"))
        if len(node.args) > 1 and ".method" in ast.unparse(node.args[1]):
            if not ast.unparse(node.args[0]).endswith(".fit"):
                found.append((node.lineno, "configuration-selected fitting must use the fit accessor"))
    elif isinstance(node.func, ast.Attribute):
        owner = ast.unparse(node.func.value)
        if owner.endswith((".plot", ".fit")) or ".plot." in owner or ".fit." in owner:
            old = {"xcol", "ycol"}.intersection(k.arg for k in node.keywords)
            if old:
                found.append((node.lineno, "legacy accessor coordinate keywords: " + ", ".join(sorted(old))))
    return found


def violations(code):
    """Find known removed/compatibility surfaces without confusing NumPy or Axes."""
    found = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    for match in re.finditer(r"\bStoner\.(\w+)", code):
        if match[1] in OLD_MODULES | OLD_CLASSES:
            found.append((code[:match.start()].count("\n") + 1, f"legacy Stoner.{match[1]}"))
    for match in re.finditer(r"\bfrom Stoner(?:\.Image(?:\.core|\.kerr)?|\.core(?:\.array)?)? import ([^\n]+)", code):
        if OLD_CLASSES.intersection(re.findall(r"\b\w+\b", match[1])):
            found.append((code[:match.start()].count("\n") + 1, "removed array/class import"))
    if tree is None:
        # Broken fragments still receive the unambiguous retired-name checks.
        for match in re.finditer(r"\.([A-Za-z_]\w*)\s*\(", code):
            if match[1] in OLD_METHODS:
                found.append((code[:match.start()].count("\n") + 1, f"retired method {match[1]}"))
        return found, False
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            found.extend(_attribute_violations(node))
        elif isinstance(node, ast.Call):
            found.extend(_call_violations(node))
    return sorted(set(found)), True


def audit(root=ROOT):
    """Return a complete inventory with explicit static-validation boundaries."""
    inventory, failures = [], []
    for example in examples(root):
        entry = asdict(example)
        entry.pop("code")
        if example.language not in {"python", "py", "python3", "pycon", "default"}:
            entry["status"] = "other-language"
        else:
            issues, parsed = violations(example.code)
            if not parsed and example.kind in {"script", "notebook"}:
                issues.append((1, "invalid Python syntax in executable example"))
            historical = example.kind == "inline" and (
                example.path in HISTORICAL or (example.path, example.code) in HISTORICAL_LITERALS)
            entry["status"] = "historical-reference" if historical and issues else "parsed" if parsed else "fragment"
            entry["findings"] = [dict(line=example.line + line - 1, message=message) for line, message in issues]
            if issues and not historical:
                failures.extend(dict(path=example.path, **issue) for issue in entry["findings"])
        inventory.append(entry)
    return {"examples": inventory, "findings": failures,
            "summary": {"examples": len(inventory), "files": len({e["path"] for e in inventory}),
                        "findings": len(failures),
                        "fragments": sum(e["status"] == "fragment" for e in inventory)}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "maintenance/runs/example-api.json")
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"]))
    for finding in report["findings"]:
        print(f"{finding['path']}:{finding['line']}: {finding['message']}")
    raise SystemExit(bool(report["findings"]))
