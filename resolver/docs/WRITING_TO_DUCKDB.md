# Writing IDMC Exports to DuckDB

This guide explains how to load the IDMC staging exports into a local DuckDB database using the bundled helper CLI and Makefile targets.

## Prerequisites

* Install the project's DuckDB extras (`pip install -e .[db]`), or run `make dev-setup`.
* Ensure the IDMC staging data exists under `resolver/staging/idmc/` (typically via the standard exporter pipeline).

## Environment defaults

The helper CLI respects a small set of environment variables:

```
RESOLVER_DB_URL=./resolver_data/resolver.duckdb
RESOLVER_WRITE_DB=1
WRITE_TO_DUCKDB=1
RESOLVER_WRITE_TO_DUCKDB=1
```

These defaults live in `resolver/.env.sample`. Copy them into your working environment if you want the exporter to persist to DuckDB automatically:

```bash
cp resolver/.env.sample .env
```

When `RESOLVER_DB_URL` is set—or you provide `--db` (or the legacy
`--db-url`) on the command line—the exporter dual-writes to DuckDB automatically
unless you explicitly disable it with `--write-db 0` (or `RESOLVER_WRITE_DB=0`).
The `WRITE_TO_DUCKDB` and `RESOLVER_WRITE_TO_DUCKDB` aliases act as defensive
toggles for workflows that expect DuckDB writes (the CI backfill job sets all of
them), but the helper still forwards `--write-db 1` to the exporter so the
DuckDB write is always attempted during wrapper runs. The CLI accepts a
filesystem path or a `duckdb:///` URL; paths are canonicalised to absolute URLs
before the exporter runs so the write, verification, and any follow-up queries
all point at the same database file.

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

### Continuous integration usage

The `resolver-initial-backfill` GitHub Actions workflow reuses the helper to
opportunistically load the export preview into the automated DuckDB bundle. The
step sets the environment toggles above, confirms that `resolver/staging/idmc/`
exists, and then runs `python -m resolver.cli.idmc_to_duckdb` with the staging
directory. The invocation now uses the normalised `--db` flag and is wrapped in
guards so empty staging or minor schema drift logs a warning instead of failing
the workflow; the auxiliary write is best-effort and the freeze stage remains in
charge of the canonical database state.

The preceding "Export canonical facts" step still passes `--write-db 1` and the
job's `RESOLVER_DB_URL` directly to `resolver.tools.export_facts`, seeding the
DuckDB file before snapshot derivation begins. Each
`resolver.tools.freeze_snapshot --write-db 1` invocation inside the snapshot loop
continues to be the authoritative write path for backfill jobs. After the
snapshots are frozen the workflow runs
`python scripts/ci/duckdb_summary.py --db "$BACKFILL_DB_PATH"` and appends the
resulting counts to the GitHub Step Summary so operators can verify the final
row totals directly from the job output.

The exporter invocation used by the workflow looks like this:

```bash
python -m resolver.tools.export_facts \
  --in resolver/staging \
  --out resolver/exports/backfill \
  --config resolver/tools/export_config.yml \
  --write-db 1 \
  --db "$RESOLVER_DB_URL" \
  --append-summary diagnostics/ingestion/summary.md
```

These explicit flags guarantee the derived snapshot writes rows into
`data/resolver_backfill.duckdb` during the backfill run rather than acting as a
preview-only export.

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

The exporter splits the unified facts dataframe by semantics before writing to
DuckDB. Rows tagged `stock` land in `facts_resolved` while rows tagged `new`
land in `facts_deltas`. Any other semantics default to `facts_resolved` so the
persisted rows still match the CSV preview. The CLI verifies the combined row
count across both tables before declaring success and the summary line reports
per-table deltas so idempotent reruns continue to show `✅ Wrote 0 rows`.

## Composite keys and parity

DuckDB initialisation enforces the canonical column lists defined in
`resolver/db/schema_keys.py`:

- `facts_resolved`: `(event_id, iso3, hazard_code, metric, as_of_date, publication_date, source_id, series_semantics, ym)`
- `facts_deltas`: `(event_id, iso3, hazard_code, metric, as_of_date, publication_date, source_id, ym)`

The writer adds any missing key columns as nullable `VARCHAR` fields before
issuing the MERGE so minimal or historic tables are healed automatically, then
recreates the `ux_facts_resolved_series` and `ux_facts_deltas_series` unique
indexes with that exact column order. That guarantees row-for-row parity with
the CSV preview—distinct events in the same month no longer collapse—and keeps
the verification step honest because the CLI sums row counts from both tables
after each run.

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
  --db ./resolver_data/resolver.duckdb \
  --out diagnostics/ingestion/export_preview
```

Passing `--strict` causes the command to exit with a non-zero status if any exporter warnings are produced.
