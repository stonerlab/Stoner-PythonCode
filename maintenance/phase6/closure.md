# Phase 6 closure validation (2026-09-12)

Status: validation in progress; Linux documentation build pending.

Corrected the remaining local markup and type references in image helpers,
folder metadata, filtering and fitting docstrings. Documented spline return
behaviour against its implementation. Package AST comparison confirms that
all six changed Python modules differ only in docstrings.

Enabled NumPy-style parsing for inherited third-party docstrings while retaining
Google style for Stoner authors. Scoped scikit-image drawing references by method
name so their shared page has distinct citation targets. Matplotlib's `rc` role
renders as literal configuration-key text. No warning suppression was added.

A trial of Napoleon's ivar output removed real API targets and was rejected.
The existing attribute-target behaviour is retained; duplicate overview/member
descriptions remain visible. One obsolete target, DiskBasedFolderMixin.flatten,
described a nonexistent Boolean attribute. The actual constructor option is flat;
folder flatten methods are unchanged.

All 73 automated documentation examples passed on Python 3.14 (406 warnings,
174.04 seconds pytest time). Of 78 Python files under doc/samples, four are
package markers and plot_folder_demo.py is a main-guarded multiprocessing
demonstration excluded by the runpy harness. No duplicate basenames silently
replace examples. Details: closure-source-examples.json. Test evidence:
maintenance/runs/20260912-171320-focused-e943d924.

Compiled CHM, DVI and PDF manuals remain reproducible, ignored build outputs as
decided in Phase 3. The definitive documentation is the RST/source docstrings;
doc/plot_cache remains the intentional versioned graphics cache. HTML is the
normal documentation deliverable; no compiled manual is required for release.
