# Continuous Integration Notes

## Resolver smoke workflow

The resolver smoke workflow exercises the stubbed ingestion path to confirm
wiring without hitting external services. The job succeeds when at least one
canonical CSV produced by the stubs contains a data row. The workflow does **not**
require DuckDB snapshots or Parquet exports, so their absence will not fail the
run.

During validation the workflow executes `scripts/ci/smoke_assert.py`, which
inspects `data/staging/<period>/canonical` and writes a `smoke-assert.json`
report under `.ci/diagnostics/`. The gate compares the total canonical rows to
`SMOKE_MIN_ROWS` (default: `1`) and records the exit status in
`.ci/exitcodes/gate_rows`. The diagnostics collector bundles both the JSON
report and a rich `SUMMARY.md` inside the diagnostics artifact so reviewers can
quickly confirm row counts per file.

Environment knobs:

- `SMOKE_MIN_ROWS` — minimum canonical rows required for a passing smoke run
  (defaults to `1`).
- `smoke_canonical_dir` input to the diagnostics composite (defaults to
  `data/staging/ci-smoke/canonical`), which propagates to the assertion helper.
