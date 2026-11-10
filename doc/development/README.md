# Developer documentation

Start with the [developer guide](../UserGuide/developer.rst) for contributor
workflows and extension points. Repository instructions for coding agents remain
in the root [AGENTS.md](../../AGENTS.md).

## Guidance and specifications

- [Docstring style](DOCSTRING_STYLE.md): conventions for API documentation.
- [Storage contracts](STORAGE_CONTRACTS.md): scientific behaviour and regression
  evidence.
- [Storage interchange](STORAGE_INTERCHANGE.md): ownership and interchange specification.
- [Maintenance checklist](../../maintenance/MAINTENANCE_PLAN.md): recurring checks,
  release preparation and outstanding follow-ups.
- [Maintenance tools](../../maintenance/README.md): validation and diagnostic commands.

## Active plans

- [Data accessors](plans/DATA_ACCESSOR_PLAN.md).
- [Fitting migration and compatibility](plans/FIT_ACCESSOR_PLAN.md).
- [Release phases and API lock](plans/RELEASE_0_12_PLAN.md).
- [Storage migration and remaining validation](plans/STORAGE_MIGRATION_PLAN.md).
- [String-valued options and enums](plans/STRING_ENUM_PLAN.md).

These Markdown files are repository working references; the published contributor
guide lives in the Sphinx user guide. Keep unresolved decisions in the active
plans and retain lasting contracts here. Remove completed implementation write-ups
once their guidance, regression coverage and release evidence are recorded.

## Markdown checks

Run Markdownlint from the repository root with `markdownlint-cli2` installed:

```powershell
markdownlint-cli2 AGENTS.md "doc/development/**/*.md" "maintenance/*.md" "maintenance/*/README.md" "maintenance/phase8/*.md" scripts/README.md tests/README.md
```

The root `.markdownlint.json` keeps prose at 80 columns. Tables and code blocks
may exceed that width so scientific comparisons remain tabular and executable
commands retain their exact spelling. All other default rules remain enabled.
