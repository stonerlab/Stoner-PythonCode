# Same-dtype image conversion copy

Fixed 2026-09-12. The shared converter calls `np.asarray(image)` before
checking the dtype, so its same-dtype `force_copy=True` path always attempted
`.clone` on a plain ndarray. This was a deterministic bug, not an exceptional
input requiring an AttributeError fallback.

Replaced that access with `np.copy(image)`. Conversion formulae and the default
same-dtype storage-sharing behaviour are unchanged. The public stack wrapper
still returns the same stack object and restores its pixel mask; explicit
copying now provides independent pixel storage. Removed the fixed limitation
from the stack conversion docstring.

Regression tests cover unsigned integer, signed integer and floating-point
inputs, default sharing versus forced independence, and public ImageStack
values, dtype, mask and return identity. All four new cases failed with the
expected AttributeError before the fix (21.93 seconds;
`maintenance/runs/20260912-212420-focused-c914fa56`).

After the fix, Python 3.14 image-core, stack and new regression tests passed:
42 passed, four warnings, 17.88 seconds
(`maintenance/runs/20260912-212458-focused-5ebcae25`).

The four regressions also passed in shared `py311-minimal`: 18 warnings,
21.18 seconds (`maintenance/runs/20260912-212517-focused-b14cac21`).
`git -c core.whitespace=cr-at-eol diff --check` passes. The full suite was not
repeated for this one-line conversion fix; validation targeted the affected
image and stack operations. Hosted validation remains outstanding.

No changes were made to the read-only `stoner_measurement` reference checkout.
