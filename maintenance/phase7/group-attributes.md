# Folder group attribute inheritance

Updated 2026-09-12 following the maintainer's request to track added and
deleted attributes when creating groups.

`BaseFolder.__clone__(attrs_only=True)` already copied `_instance_attrs`, but
assignment never populated that set. It also replayed original constructor
keyword values, which could restore stale or deleted attributes.

## Changes

- `BaseFolder.__setattr__` records successfully assigned public extra
  attributes in the existing instance-owned set. Private names, class-defined
  attributes/properties, declared defaults, and the existing args/kwargs,
  executor and directory bookkeeping are excluded.
- The existing protected-attribute deletion hook now removes tracking records
  after successful deletion. Failed deletions retain their existing behaviour.
- Attribute-only cloning uses current constructor attribute values only when
  the attribute still exists, and propagates tracked extras to new groups.
- The existing `each` deletion route now delegates to the folder's deletion
  hook; it previously attempted indexed deletion from a set.
- Updated `add_group` documentation, including its in-place return and no-op
  behaviour for existing group keys, and removed the addressed TODO.

Existing groups are not retroactively synchronised when the parent changes.
Attribute values retain the existing attribute-clone assignment semantics;
this change does not introduce deep copies of mutable attribute values.

## Validation

Tests cover constructor settings, updated values, additions, deletions,
re-additions, nested groups, private-state exclusion, protected properties,
the each deletion path, empty child contents, and existing group identity.

Before assignment/deletion tracking, three policy cases failed and three
passed in 5.93 seconds
(`maintenance/runs/20260912-214143-focused-fed45186`).
After the fix, seven focused cases passed on Python 3.14 in 5.61 seconds
(`maintenance/runs/20260912-214404-focused-9cb0d506`). After adding the protected
deletion assertion, the shared Python 3.11 minimal environment passed all seven
cases in 8.06 seconds, with 18 warnings
(`maintenance/runs/20260912-214450-focused-da4badcf`).

The full Python 3.14 serial suite passed: 385 tests, 592 warnings,
417.25 seconds (`maintenance/runs/20260912-214439-serial-ce860e91`). This also
validates the accumulated earlier runtime fixes. The final diff check passes
with the repository's CR-at-EOL setting. Hosted validation remains outstanding.
No changes were made to the read-only stoner_measurement reference checkout.
