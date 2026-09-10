# Phase 1: checkout workflow — 2026-09-10

Windows Git is the agreed client for this checkout. The maintainer explicitly approved these repository-local settings:

```powershell
git config --local core.fileMode false
git config --local core.autocrlf false
```

The index records all 691 tracked files as executable (`100755`); Windows previously presented them as `100644`. Ignoring filesystem executable-bit differences removes this noise while preserving the indexed modes. Disabling automatic newline conversion preserves the existing mixed LF and CRLF contents.

## Verification

- A disposable shared clone of the exact baseline commit was checked out with these settings: clean status and `git diff --exit-code` success.
- The semantic binary diff of the live checkout, computed with `--ignore-space-at-eol`, had the identical SHA256 before and after changing configuration. See `verification.txt`.
- No source file, indexed mode or line ending was normalized. The existing semantic working changes remain present.
- `Stoner/formats/utils/__init__.py` already had a one-line LF-to-CRLF change. With conversion disabled, Git reports it explicitly; `git diff --ignore-space-at-eol -- Stoner/formats/utils/__init__.py` is empty. Preserve it unless newline normalization is explicitly included in a later change.
- `maintenance/.gitignore` excludes only generated output from the new maintenance checks; durable reports and environment manifests remain visible.
- No Cygwin installation was found at the usual locations or in its setup registry keys. This is verified for Windows Git only. Use a separate checkout for Cygwin rather than alternating clients here.

## Routine use and rollback

Review `git status --short` and `git diff --ignore-space-at-eol`; stage only the intended changes. Existing indexed modes are preserved by ordinary `git add`. Do not use `git add --renormalize` as an incidental maintenance step.

To restore the exact previous local configuration, without changing tracked files:

```powershell
git config --local core.fileMode true
git config --local --unset core.autocrlf
```

The latter restores inheritance of the machine-wide `core.autocrlf=true`. No history rewrite, commit or push was performed.