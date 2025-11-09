# DuckDB Setup for Resolver Development

DuckDB is required for the resolver database parity and idempotency tests. Once
installed, any tests that rely on `pytest.importorskip("duckdb")` will execute
normally instead of being skipped.

## Installing via Poetry

```bash
poetry install --with dev
```

If you only want the resolver package extras, you can install them explicitly:

```bash
poetry install --with dev --with test
```

## Installing via pip

```bash
python -m pip install "duckdb>=1.0,<2"
```

## Verify the installation

```bash
python -c "import duckdb; print(duckdb.__version__)"
```

Once the command above prints a version, run the DuckDB-specific test subset:

```bash
pytest -q resolver/tests -k duckdb
```

These tests will now run in both local development and continuous integration
pipelines.

## IDMC flow semantics

The `resolver.cli.idmc_to_duckdb` wrapper now runs in dry-run mode by default
and only performs inserts when `--write-db` is supplied. It prefers the
canonical `facts.csv` emitted by the exporter (pass `--facts-csv`), falling back
to staging re-exports when the CSV is absent. Rows with
`metric=new_displacements` are still forced to `series_semantics=new`, so they
land in `facts_deltas`, while accompanying stock rows continue to populate
`facts_resolved`. Successful writes log `✅ Wrote N rows to DuckDB`, and the
command exits with a non-zero status when `--write-db` is set but the source
contains zero rows (useful guardrail for automation).

## Initial backfill automation

The `resolver-initial-backfill.yml` workflow invokes
`python -m resolver.cli.idmc_to_duckdb` immediately after exporting the preview
CSV so the generated facts land in the `BACKFILL_DB_PATH` DuckDB file during CI.
The step passes `--facts-csv` and `--write-db`, enabling inserts only when the
canonical CSV exists. The subsequent "Verify DuckDB contents" step queries
DuckDB directly, writes `diagnostics/ingestion/duckdb_counts.md`, and appends the
table/row breakdown to both the GitHub Step Summary and the ingestion summary for
auditing.
