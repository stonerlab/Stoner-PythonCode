# Phase 6 closure validation (2026-09-12)

Status: complete. Implementation validated at commit fb21b974f.

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

The first Linux check, workflow run 34705113529 at commit 1cedeb2c9, succeeded
with 81 warnings and no missing stubs. It exposed malformed maths in the public
lorentzian_diff function and a missing Graphviz executable in the documentation
environment. Both are corrected for the repeat build: the function docstring
now agrees with its model class, and doc/docs-env.yml supplies graphviz. The pip
requirements and contributor guide explain the separate executable dependency.

The audit now supports an explicit reviewed warning-count ceiling. New or
increased diagnostics fail; existing warnings remain visible. The Windows
manifest accepts only duplicate descriptions, the intentional case collisions
and unreferenced imported bibliography entries, not arbitrary warning categories.

## Final results

The repeat Linux workflow [34705333120](https://github.com/stonerlab/Stoner-PythonCode/actions/runs/34705333120)
succeeded on Python 3.13 with 79 reviewed warnings: 74 duplicate descriptions
and five unreferenced imported bibliography entries. There are no missing stubs,
markup errors, ambiguous Python references or missing Graphviz warnings.
See closure-linux.json and expected-warnings-linux.json. The existing workflow
does not export its inventory, so Linux verification uses its build diagnostics;
the complete primary/dynamic API inventory comparison is local.

The final Windows Python 3.14.7 / Sphinx 9.1.0 build succeeds with 106 reviewed
warnings: 76 duplicate descriptions, 25 intentional filename collisions and
the same five bibliography entries. This is down from the fresh Phase 6 baseline
of 1,658. No warning suppression was introduced. New warnings or increased counts
fail the audit against expected-warnings-windows.json; negative probes confirm
both failure conditions (closure-gate-check.json).

closure-after.json confirms all five primary classes and 85 dynamic Data methods
remain documented. Only the reviewed nonexistent DiskBasedFolderMixin.flatten
attribute target was removed during closure. All 243 plot-cache hashes remain
unchanged (closure-cache-check.json). All seven changed package modules have
identical executable ASTs to f7259df17; see closure-source-examples.json.

Served the final HTML directory on 127.0.0.1:8766. Browser navigation from the
front page to Data worked, and the corrected DiskBasedFolderMixin page was
inspected in both its rendered DOM and a screenshot. Attribute descriptions,
defaults and links render correctly. Long qualified names still wrap in the
theme's narrow sidebar; this is a presentation limitation, not missing content.

## Reproduce the closure gate

Run a cached build with READTHEDOCS=True, preserve its warning log, then run:

```powershell
python maintenance/audit-docs.py maintenance/runs/phase6-closure-verified maintenance/runs/phase6-closure-verified-warnings.log maintenance/phase6/closure-after.json --compare maintenance/runs/phase6-batch7-final --allow-removed maintenance/phase6/closure-obsolete-attributes.json --expected-warnings maintenance/phase6/expected-warnings-windows.json
```

For a Linux build, use expected-warnings-linux.json and add
`--require-fitting-functions`. This checks every exported fitting function,
including names that differ from model classes only in capitalisation.
Do not approve new warnings automatically when environments change; investigate
their source and update the reviewed ceiling only with evidence.

All Phase 6 tasks and completion criteria are satisfied. Remaining known
documentation limitations are explicitly bounded above. Phase 7 can begin
with the previously reproduced image conversion force_copy defect, followed
by the plan's focused source workstreams; no Phase 7 code change is included here.
