# Next stable release: image accessors and API stability plan

Date: 2026-10-30. Working release series: 0.12.
Status: agreed feature sequence and release-phase policy. The standalone
ImagePlotter alpha is published as 0.12.0a6; the ROI/fit milestones remain future
work. This plan does not authorise subsequent publication by itself.

See [maintenance guidance](MAINTENANCE_PLAN.md), the
[accessor roadmap](DATA_ACCESSOR_PLAN.md), and the detailed
[plot](IMAGE_PLOT_ACCESSOR_PLAN.md), [ROI](IMAGE_ROI_ACCESSOR_PLAN.md) and
[fit](IMAGE_FIT_ACCESSOR_PLAN.md) implementation/testing plans.

## 1. Intended stable-release scope

The intended image feature sequence is:

1. `ImageFile.plot`: common drawing, Axes/property/context conventions with Data.
2. `ImageFile.roi`: current ROI management, reusable descriptions/resolution,
   programmatic and interactive selection, clear and reviewed consumers.
3. `ImageFile.fit`: selected-background plane/quadratic fits and whole-image
   flattening using the shared ROI facility.

Do not start fitting by creating its own independent ROI mechanism, or require
the complete ROI/fitting feature set before shipping the first plotting alpha.
Shared coordinate services belong to the initial plot work and evolve compatibly
through the remaining alphas. Each accessor has its own demonstrable alpha.

The stable release also includes the already published alpha work: storage and
column-schema/selection changes, Data plotting and fitting accessors, fit model
registration/outputs, migrated consumers and fit annotations. Assess the whole
transition from 0.11, not only the final image changes.

The maintainer raised whether the cumulative API break warrants a larger version
than 0.12. Keep that as an explicit version-number checkpoint before API lock.
Compare supported migration paths and readiness for a stronger stability
commitment, rather than choosing a larger number solely from feature count.
No larger version is selected here. Existing published alphas, including
`v0.12.0a5`, remain immutable regardless of that decision.

## 2. Agreed phase policy

| Phase | Permitted work | API rule |
| --- | --- | --- |
| Accessor alphas | Implement/review the planned features, test migration and revise unsettled contracts | Breaking changes can still be considered, documented and tested before the lock |
| Beta | Bug fixes, documentation and small compatible additions that smooth practical use | API locked: no breaking API changes |
| Release candidate | Bug fixes and verification; documentation corrections supporting existing contracts | Full API freeze: no API additions or breaking changes |
| Stable | Publish the validated frozen candidate | Must satisfy the RC contracts and exact-source release gates |

Published and planned alpha milestones, with future versions subject to release preparation:

| Milestone | Candidate version | Required evidence |
| --- | --- | --- |
| Plot accessor alpha | Published 0.12.0a6 | Standalone ImagePlotter; exact-source CI and publication evidence in MAINTENANCE_PLAN.md |
| ROI accessor alpha | 0.12.0a7 | Current state/set/clear, reusable selectors, geometry and reviewed plotting/reduction integration |
| Fit accessor alpha | 0.12.0a8 | Plane/quadratic reference-region fitting, reports and full-grid floating-point correction |
| API-lock review | Before the first beta | End-to-end workflows, compatibility inventory and unresolved contract decisions closed |
| Beta series | b1 onwards in the selected series | Only compatible small additions and fixes; test each new convenience against the locked baseline |
| Full API-freeze review | Before rc1 | Beta rough edges addressed; public/API/schema inventories frozen |
| RC series | rc1 onwards in the selected series | Bug squashes and full regression/publication rehearsal; no new API |

Version numbers are planning targets, not reserved tags or instructions to publish
automatically. Extra corrective alphas may be necessary; do not combine missing
milestones into a nominally complete release or waive their exit gates.

## 3. What counts as API

The lock covers supported public behaviour, not just an inventory of method names:

- Properties, methods, import paths, signatures, positional/keyword acceptance,
  default values, callback conventions and fluent return types.
- Data/Image consistency: callable fitting defaults, model shortcuts, image
  coordinate selectors, plotting Axes returns and retained legacy Figure returns.
- Coordinate units/order/origin/extent, ROI geometry and boundary rules, current
  state/default/explicit/all precedence, mask polarity and frame/region scope.
- Model names/parameter order, fitting report/output semantics, uncertainty and
  covariance conventions, reference levels and correction dtype/mutation rules.
- Supported enums and strings, registry collision/replacement rules, error and
  warning behaviour, metadata namespaces and serialised ROI/fit/provenance schemas.
- Copy/clone/pickle/export behaviour, owner lifetimes, figure/context ownership,
  folder/worker dispatch and documented selection/processing side effects.

Do not recategorise a breaking default change or removal as a convenience or bug
fix. Correcting an implementation to its locked documented contract is a bug fix,
but still needs impact review, regression tests and release notes. If a necessary
change cannot preserve the locked contract, stop the beta/RC progression and
explicitly reopen alpha/API review with the maintainer; do not silently waive the
phase rule. New public enum values, accepted argument types and registry methods
also need review for collisions and effects on existing calls.

Examples of appropriate beta additions: a clearly named convenience method,
optional keyword with an unchanged default, or a compatible alias, when existing
calls/results/state remain unchanged. New optimiser families, major processing
groups or an unreviewed result abstraction are further feature work and should
not enter under the label of polishing rough edges.

RC fixes may change implementation to satisfy frozen contracts, improve validation
or correct docs/tests. They must not introduce new public arguments, defaults,
methods, enum domains, model types or interchange schemas. Dependency/workflow
changes must be necessary fixes and revalidated, not a route to new features.

## 4. API-lock checklist: alpha to beta

- [ ] Each accessor alpha's minimum deliverable and focused/full/hosted gates pass.
- [ ] Real user workflows cover display, ROI create/edit/clear/reuse, reduction/
  histogram, fit, whole-image correction and inspection with common Data conventions.
- [ ] The three accessor plans' open names/defaults/return/reference/selection/
  coordinate/ownership decisions are settled and reflected in executable docs.
- [ ] Review cumulative 0.11 -> candidate breaking changes with a migration table:
  old call, replacement, changed semantics, retained compatibility and remaining limits.
- [ ] Decide the stable version series, compatibility/deprecation policy and the
  status of direct Data fitting, legacy image.imshow and level_image. Do not allow
  a beta/RC removal merely because an older planning document mentioned it.
- [ ] Audit all supported consumers, including folder/stack adapters, image selectors,
  repr hooks, notebooks if tracked, examples and scripts; record excluded/deferred work.
- [ ] Freeze a reviewed public signature/inventory and behavioural baseline;
  explicitly classify internal versus public names and versioned stored records.
- [ ] Document supported image calibration/ROI/frame/uncertainty limits and manual
  GUI/backend evidence. Unsupported behaviour is rejected explicitly.
- [ ] Update compatibility/release notes and user migration guides. No essential
  feature or scientific-contract decision remains deferred into beta.

## 5. Beta and RC gates

For beta, log each proposed small addition with its intended workflow, signature,
unchanged old-call behaviour and collision/default/serialisation assessment.
Demonstrate additive compatibility against the API-lock baseline; keep the exact
new behaviour in tests/examples. Fix regressions promptly without expanding scope.

Before rc1, close or explicitly defer beta additions to a later release, freeze
the final public/API/schema baseline, and review all migration/compatibility text.
No unimplemented convenience methods should remain presented as released APIs.

During RC, run targeted tests for fixes and the full supported release gates on
the exact candidate source. Inspect representative images, masks, contours,
reference regions and corrected surfaces; verify sample workflows, installed
packages, retained assets and cached documentation. Record manual/physical-data
limits separately from automated tests. A fix that changes public semantics must
be assessed against the frozen contract before coding or publication.

## 6. Release evidence and completion

Keep the repository's existing exact-commit matrix, lower-dependency,
installed-distribution and documentation gates. Builds and deployments occur in
GitHub Actions. Tag only a validated committed source version; never retarget a
published tag. Verify public PyPI/Conda files, installed Conda import/version,
versioned documentation and citation/DOI after each authorised release.

Update the active checklist with the next bounded step. Record completed alpha/
beta/RC evidence in commits, releases/PRs and CI links rather than accumulating
separate historical phase reports. A green suite alone does not establish a
frozen or compatible API; the contract/migration review is a separate gate.

Next work: ImageROI R1 inventory/contract review, then the shared descriptions,
current-selection state and resolver, reusing the released plotting/coordinate
foundation. A subsequent alpha publication requires its own authorisation and
exact-source gates; this plan does not automatically create future tags.
