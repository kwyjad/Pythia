# Continuous Integration Notes

## Resolver smoke workflow

The resolver smoke workflow exercises the stubbed ingestion path to confirm
wiring without hitting external services. The job succeeds when at least one
canonical CSV produced by the stubs contains a data row. The workflow does **not**
require DuckDB snapshots or Parquet exports, so their absence will not fail the
run.

During validation the workflow executes `scripts/ci/assert_smoke_outputs.py`,
which inspects `data/staging/<period>/canonical` and writes a
`smoke-assert.json` report under `.ci/diagnostics/`. The diagnostics collector
bundles this report alongside `SUMMARY.md` inside the diagnostics artifact so
reviewers can quickly confirm row counts per file.
