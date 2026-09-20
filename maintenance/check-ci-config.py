"""Validate the repository's CI policy without contacting GitHub."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
PINNED_ACTION = re.compile(r"^\s*uses:\s+[^\s@]+@([0-9a-f]{40})(?:\s+#.*)?$", re.MULTILINE)
ANY_ACTION = re.compile(r"^\s*uses:\s+([^\s]+)", re.MULTILINE)


def require(condition: bool, message: str) -> None:
    """Raise a readable configuration error when *condition* is false."""
    if not condition:
        raise SystemExit(message)


def main() -> None:
    """Check action pins, permissions, coverage flags and retired tooling."""
    workflows = {path.name: path.read_text(encoding="utf-8") for path in WORKFLOW_DIR.glob("*.yaml")}
    require(workflows, "No GitHub Actions workflows were found.")

    for name, text in workflows.items():
        actions = ANY_ACTION.findall(text)
        pins = PINNED_ACTION.findall(text)
        require(len(actions) == len(pins), f"{name} contains an action that is not pinned to a commit SHA.")
        require("curl" not in text and "wget" not in text, f"{name} executes a network installer script.")

    tests = workflows["run-tests-action.yaml"]
    for version in ("3.12", "3.13", "3.14"):
        require(version in tests, f"Python {version} is missing from the test matrix.")
    for runner in ("ubuntu-latest", "macos-15-intel"):
        require(runner in tests, f"{runner} is missing from the test matrix.")
    require("windows-latest" not in tests, "The unreliable hosted Windows lane has been restored.")
    require("micromamba-extra-args: pyqt6" in tests, "The macOS widget-test dependency is missing.")
    require("pull_request:" in tests and "workflow_dispatch:" in tests, "The test workflow lacks required triggers.")
    require("contents: read" in tests, "The test workflow must default to read-only contents permission.")
    require("pull-requests: write" not in tests, "The test workflow has unnecessary pull-request write access.")
    require("carryforward:" not in tests, "Coveralls carryforward flags must not mask missing matrix jobs.")
    require(
        "flag-name: run-${{ matrix.python-version }}-${{ matrix.os }}" in tests,
        "Coveralls flags do not match the test matrix.",
    )
    require(
        tests.count("coverage: true") == 1
        and re.search(r'python-version: "3\.14"\s+os: ubuntu-latest\s+coverage: true', tests),
        "Only Ubuntu Python 3.14 should collect coverage.",
    )
    require(
        "- name: Upload coverage to Coveralls\n        if: matrix.coverage == true" in tests
        and "if: matrix.coverage == true && github.event_name != 'pull_request'" in tests,
        "Coverage uploads must use the coverage matrix marker, excluding Codacy on pull requests.",
    )
    require("codacy/codacy-coverage-reporter-action@" in tests, "The pinned Codacy action is missing.")
    require(
        tests.count('--cov-config="$GITHUB_WORKSPACE/pyproject.toml"') == 1
        and 'COLLECT_COVERAGE: ${{ matrix.coverage == true }}' in tests
        and 'if [[ "$COLLECT_COVERAGE" == "true" ]]; then' in tests,
        "The coverage job must conditionally pass the absolute coverage config.",
    )

    docs = workflows["build-docs.yaml"]
    require("contents: read" in docs and "contents: write" not in docs, "Documentation CI is not read-only.")
    require("READTHEDOCS: True" in docs, "Documentation CI does not consume the retained plot cache.")
    require("QT_QPA_PLATFORM: offscreen" in docs, "Documentation CI must initialise Qt without a display.")
    require("MPLBACKEND: Agg" in docs, "Documentation CI must use a non-interactive Matplotlib backend.")
    require("git push" not in docs, "Documentation CI must not push generated cache changes.")

    require(not (ROOT / ".travis.yml").exists(), "The inactive Travis configuration is still present.")
    require("prospector" not in (ROOT / "Makefile").read_text(encoding="utf-8"), "Obsolete Prospector target remains.")

    test_environment = (ROOT / "tests" / "test-env.yml").read_text(encoding="utf-8").lower()
    require("pytesseract" in test_environment and "tesseract" in test_environment, "OCR test tools are incomplete.")
    require("conda-forge::libmagic >=5.48" in test_environment, "The current conda-forge libmagic is not pinned.")

    lower = workflows["check-minimum-dependencies.yaml"]
    require("environment-file: tests/minimum-env.yml" in lower, "The lower dependency environment is not exercised.")
    require("python -m pytest -n 2" in lower, "The lower dependency job must run the full parallel suite.")
    require("contents: read" in lower, "The lower dependency job must be read-only.")
    minimum_environment = (ROOT / "tests" / "minimum-env.yml").read_text(encoding="utf-8")
    minimum_packages = {
        re.sub(r"[-_.]+", "-", name.lower())
        for name in re.findall(r"^  - (?:[\w-]+::)?([\w.-]+)", minimum_environment, re.MULTILINE)
    }
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for dependency in project["project"]["dependencies"]:
        name = re.match(r"[\w.-]+", dependency).group()
        require(
            re.sub(r"[-_.]+", "-", name.lower()) in minimum_packages,
            f"Required runtime dependency {name} is missing from tests/minimum-env.yml.",
        )
    packages = workflows["check-packages.yaml"]
    require(project["project"]["requires-python"] == ">=3.12", "The package minimum must be Python 3.12.")
    require("  - python =3.12" in minimum_environment, "The lower dependency job must use Python 3.12.")
    require("environment-name: stoner-py312-minimum" in lower, "The lower dependency environment name is stale.")
    require("3.11" not in tests + lower + packages, "CI still advertises unsupported Python 3.11.")
    require("python-version: ['3.12', '3.14']" in packages, "Installed-package checks must cover Python endpoints.")
    require("python-version: ${{ matrix.python-version }}" in packages, "Installed-package checks ignore the matrix.")

    print(f"CI policy passed ({sum(len(ANY_ACTION.findall(text)) for text in workflows.values())} pinned actions).")


if __name__ == "__main__":
    main()
