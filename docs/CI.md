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
quickly confirm row counts per file. The summary now always includes the
per-step exit-code breadcrumbs and the final 120 lines of every
`scripts/ci/run_and_capture.sh` invocation (pip-freeze, import probes, pytest),
so triage rarely requires unpacking the artifact.

Environment knobs:

- `SMOKE_MIN_ROWS` — minimum canonical rows required for a passing smoke run
  (defaults to `1`).
- `smoke_canonical_dir` input to the diagnostics composite (defaults to
  `data/staging/ci-smoke/canonical`), which propagates to the assertion helper.

## Fast tests

`resolver-ci-fast.yml` builds a deterministic fixture dataset before pytest
runs. The `fast_exports` session fixture calls
`resolver.tests.fixtures.bootstrap_fast_exports.build_fast_exports()` to copy
`resolver/tests/fixtures/staging/minimal/canonical/facts.csv`, run
`resolver.tools.load_and_derive` offline, and mirror the Parquet exports to CSVs
so the files/csv backend has realistic inputs. Environment variables like
`RESOLVER_DB_PATH`, `RESOLVER_SNAPSHOTS_DIR`, and `RESOLVER_TEST_DATA_DIR` are
set for the duration of the tests, and `fast_staging_dir` points schema checks
at the generated canonical directory. The previously skipped contract and
staging-schema suites now execute inside the fast matrix without requiring live
staging data or network access.
