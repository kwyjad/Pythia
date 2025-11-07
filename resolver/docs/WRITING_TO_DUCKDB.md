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
✅ Wrote 5 rows to DuckDB (facts_resolved) — total 5
Summary: Exported 5 rows; wrote 5 rows (total 5) to DuckDB @ /abs/path/resolver.duckdb; warnings: 0; exit=0
Sources:
 - idmc_flow
```

A successful run creates (or updates) `resolver_data/resolver.duckdb`. Re-running the target is idempotent—the row counts remain stable while updated values replace older entries. Exit codes mirror that summary: `0` for success, `2` when `--strict` escalates warnings, and `3` when verification or the exporter fails.

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
