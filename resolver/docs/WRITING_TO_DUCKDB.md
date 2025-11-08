# Writing IDMC Exports to DuckDB

This guide explains how to load the IDMC staging exports into a local DuckDB database using the bundled helper CLI and Makefile targets.

## Prerequisites

* Install the project's DuckDB extras (`pip install -e .[db]`), or run `make dev-setup`.
* Ensure the IDMC staging data exists under `resolver/staging/idmc/` (typically via the standard exporter pipeline).

## Environment defaults

The helper CLI respects two environment variables:

```
RESOLVER_DB_URL=./resolver_data/resolver.duckdb
RESOLVER_WRITE_DB=1
```

These defaults live in `resolver/.env.sample`. Copy them into your working environment if you want the exporter to persist to DuckDB automatically:

```bash
cp resolver/.env.sample .env
```

When `RESOLVER_DB_URL` is set the exporter dual-writes to DuckDB even if you
omit `--write-db`. Provide `--write-db 0` (or export `RESOLVER_WRITE_DB=0`) to
skip the database write while keeping the CSV diagnostics. The helper accepts a
filesystem path or a `duckdb:///` URL; paths are canonicalised to absolute URLs
before the exporter runs so the write and verification always point at the same
database file.

## Write the IDMC flow data

Run the Makefile target to export IDMC flow rows and upsert them into DuckDB:

```bash
make idmc.db
```

The command prints a short summary similar to:

```
📄 Exported 5 rows to diagnostics/ingestion/export_preview/facts.csv
✅ Wrote 0 rows to DuckDB (facts_resolved) — total 0
✅ Wrote 5 rows to DuckDB (facts_deltas) — total 5
✅ Verified DuckDB row counts: resolved=0 deltas=5 total=5
Warnings: none
Summary: Exported 5 rows; wrote 5 rows (resolved Δ=0, deltas Δ=5) → totals resolved=0, deltas=5, total=5 to DuckDB @ /abs/path/resolver.duckdb; warnings: 0; exit=0
Sources:
 - idmc_flow
```

A successful run creates (or updates) `resolver_data/resolver.duckdb`. Re-running the target is idempotent—the row counts remain stable while updated values replace older entries. Exit codes mirror that summary: `0` for success, `2` when `--strict` escalates warnings, and `3` when verification or the exporter fails. The helper always prints a `Warnings:` block; `Warnings: none` means the run was clean while any listed entries are compatible with strict-mode exit code `2`.

## Single-source selection

The helper focuses on the canonical `flow.csv` export. When both `flow.csv` and
`idmc_facts_flow.parquet` are staged the CLI copies only the CSV (and an
optional `stock.csv`) into a temporary working directory before calling the
exporter. This avoids double-counting the same rows while leaving the original
staging assets untouched. The skipped Parquet file is recorded as a warning so
you can confirm the behaviour in the `Warnings:` block. Missing `stock.csv`
also appears as a warning; rerun with `--strict` to promote those conditions to
exit code `2`.

## Semantics-aware routing

The exporter now splits the unified facts dataframe by `semantics` before
writing to DuckDB. Rows tagged `stock` land in `facts_resolved` while rows
tagged `new` land in `facts_deltas`. Any other semantics are skipped from the DB
write (the CSV preview still contains them) and the CLI verifies the combined
row count across both tables before declaring success. The summary line reports
per-table deltas so idempotent reruns continue to show `✅ Wrote 0 rows`.

## Composite keys and parity

DuckDB initialisation drops the legacy `(ym, iso3, hazard_code, metric, series_semantics)`
unique indexes and replaces them with composite keys that match the exporter’s
semantics-aware routing. When an `event_id` column is present it participates in
the merge key; otherwise the helper falls back to the full context columns—`iso3`,
`hazard_code`, `metric`, the exact `as_of_date`/`as_of`, `publication_date`, `source_id`,
`ym`, and `series_semantics`. The richer key prevents distinct events in the same month
from being collapsed into a single row and keeps DuckDB row counts aligned with
the CSV preview and parity tests. If you point the helper at a minimal table that
predates some of those columns, the schema healer intersects the requested key
with whatever columns exist and falls back to a unique index on that subset so the
write still succeeds.

## Preventing double matches

The exporter now accepts an `--only-strategy` switch that restricts processing
to a single mapping strategy. The IDMC helper always passes
`--only-strategy idmc-staging`, so only the dedicated staging strategy claims
`flow.csv`/`stock.csv` and any competing matches are logged and dropped. You can
use the same flag when running the exporter directly:

```bash
python resolver/tools/export_facts.py \
  --in resolver/staging/idmc \
  --out diagnostics/ingestion/export_preview \
  --only-strategy idmc-staging
```

The dedicated IDMC strategy guard remains in place so only the intended staging
files are processed; the semantics-aware split removes the need for any
downstream source filtering.

## Inspect the database

Use the helper target to review key table counts:

```bash
make db.inspect
```

Example output:

```
facts_resolved: 5
facts_deltas: 0
manifests: (missing) Catalog Error: Table with name manifests does not exist!
```

To see the configured database location, run:

```bash
make db.path
```

## Common pitfalls

* **Missing or empty `stock.csv`:** the exporter logs a warning but still writes the flow rows. Use `--strict` (or set `RESOLVER_WRITE_STRICT=1`) if you want the process to fail when warnings occur.
* **Parquet staged alongside `flow.csv`:** the helper intentionally skips
  `idmc_facts_flow.parquet` to avoid writing duplicate rows. The skip appears in
  the warnings list.
* **Staging directory absent:** confirm the path in `resolver/staging/idmc/` contains the expected `flow.csv` (and optional `stock.csv`) before running `make idmc.db`.
* **DuckDB not installed:** install the `duckdb` Python package (via `pip install duckdb`) or run the repo's `dev-setup` script.

For lower-level control, you can invoke the CLI directly (paths and URLs are
interchangeable):

```bash
python -m resolver.cli.idmc_to_duckdb \
  --staging-dir resolver/staging/idmc \
  --db-url ./resolver_data/resolver.duckdb \
  --out diagnostics/ingestion/export_preview
```

Passing `--strict` causes the command to exit with a non-zero status if any exporter warnings are produced.
