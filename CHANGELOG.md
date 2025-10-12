# Changelog

## Unreleased

- migrate batch resolve request/response models to Pydantic v2 while preserving validation behaviour and CLI/API compatibility.
- fix DuckDB-backed resolver lookups so `series="new"` reads `facts_deltas.value_new` (and stocks continue to read `facts_resolved.value`) while harmonising the pydantic/duckdb dependency pins.
- fix the DuckDB `series="new"` selector to only select columns that exist on `facts_deltas`, alias missing provenance fields to empty strings, and expose query errors instead of silently reporting "no data".

## v0.9.1 – Stabilization Pack

- Pin the `db` extra to DuckDB `>=1.1,<2.0` plus compatible `pandas`/`pyarrow` and ensure every workflow installs via `pip install -e .[db,test]`.
- Expand CI matrices to exercise DuckDB merge on/off paths, record the installed DuckDB version, and fail if DuckDB suites are skipped.
- Add `scripts/ci/assert_no_skipped_db_tests.py`, snapshot manifest verification tooling, and new contract tests covering semantics + natural keys for `facts_resolved` and `facts_deltas`.
- Document the operational runbook, snapshot manifest expectations, and composite key requirements in the resolver docs.
