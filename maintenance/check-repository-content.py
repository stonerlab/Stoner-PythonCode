"""Check Phase 3's repository-content policy without changing the checkout."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOVED_PREFIXES = (
    ".eggs/",
    ".ipynb_checkpoints/",
    ".idea/",
    "doc/classes/",
    "doc/pypi-docs/",
    "doc/PythonCode_API.chm",
    "prospector-report.txt",
    "tests/Stoner/folders/dask-worker-space/",
    "tests/stoner/prof/",
)
REMOVED_FILES = {
    "doc/PythonCode_API.chm",
    "doc/UserGuide/CheatSheet.pdf",
    "doc/UserGuide/Users_Guide.dvi",
    "doc/UserGuide/Users_Guide.pdf",
}


def git_lines(*args: str) -> list[str]:
    """Return non-empty output lines from a Git command."""
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, check=True, encoding="utf-8"
    )
    return [line for line in result.stdout.splitlines() if line]


def main() -> None:
    """Verify removed output is absent and the plot cache remains versioned."""
    tracked = git_lines("ls-files")
    prohibited = [path for path in tracked if is_prohibited(path)]
    if prohibited:
        raise SystemExit(f"Unexpected tracked generated output: {prohibited}")

    unignored = git_lines("ls-files", "-o", "--exclude-standard")
    prohibited = [path for path in unignored if is_prohibited(path)]
    if prohibited:
        raise SystemExit(f"Unexpected unignored generated output: {prohibited}")

    plot_cache = [path for path in tracked if path.startswith("doc/plot_cache/")]
    if not plot_cache:
        raise SystemExit("The intentional documentation plot cache is not tracked.")

    print(f"Phase 3 repository-content policy passed ({len(plot_cache)} cached plot files retained).")


def is_prohibited(path: str) -> bool:
    """Return whether *path* is a removed or ignored local output."""
    return (
        path.startswith(REMOVED_PREFIXES)
        or path in REMOVED_FILES
        or ".egg-info/" in path
        or (path.startswith("doc/samples/") and "/prof/" in path)
    )


if __name__ == "__main__":
    main()
