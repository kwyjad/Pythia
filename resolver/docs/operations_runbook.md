# Resolver Operations Runbook

This runbook covers the minimal steps required to reproduce Resolver snapshots,
operate the DuckDB dual-write path, and debug CI matrix failures. Follow these
steps whenever you cut the monthly bundle or investigate a regression.

## Prerequisites

```bash
python -m pip install --upgrade pip
pip install -e .[db,test]
```

The `[db]` extra pins DuckDB (currently `>=1.1,<2.0`) alongside compatible
`pandas`/`pyarrow` builds so parquet exports, manifest verification, and
DuckDB-backed tests run without manual dependency juggling.

## Running the monthly snapshot locally

```bash
# 1. Export canonical facts
python -m resolver.tools.export_facts --in resolver/staging --out resolver/exports

# 2. Resolve precedence and make deltas
python -m resolver.tools.precedence_engine --facts resolver/exports/facts.csv --cutoff YYYY-MM-30
python -m resolver.tools.make_deltas --resolved resolver/exports/resolved.csv --out resolver/exports/deltas.csv

# 3. Freeze the snapshot (writes CSV/Parquet + DuckDB when enabled)
python -m resolver.cli.snapshot_cli make-monthly --ym YYYY-MM --write-db 1

# 4. Verify manifest integrity
python -m resolver.tools.verify_snapshot_manifest resolver/snapshots/YYYY-MM/manifest.json
```

Snapshots land in `resolver/snapshots/YYYY-MM/` and always include a
`manifest.json` summarising file hashes, row counts, the schema version, and the
DuckDB target used for dual-write. The CLI now refuses to finish if the manifest
fails verification, ensuring artifact drift is caught immediately.

## CI expectations

- Every CI workflow installs with `pip install -e .[db,test]`.
- The matrix toggles both DuckDB versions (`0.10.x` and `latest`) and merge
  support (`RESOLVER_DUCKDB_DISABLE_MERGE` set to `0` or `1`).
- After pytest completes the guard `scripts/ci/assert_no_skipped_db_tests.py`
  parses `pytest-junit.xml`; the job fails if DuckDB suites were skipped or if
  unexpected skip reasons appear.
- Snapshot publishing re-runs the manifest verifier before uploading
  artifacts or committing changes.

## Common errors & fixes

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ImportError: No module named duckdb` | Extras not installed | `pip install -e .[db,test]` |
| `DuckDB snapshot write skipped: duckdb_io unavailable` | `duckdb` import guard tripped | Install extras or set `--write-db 0` to disable dual-write |
| `DuckDB merge errors in CI` | `RESOLVER_DUCKDB_DISABLE_MERGE=0` path exercising merge writes | Reproduce locally with `RESOLVER_DUCKDB_DISABLE_MERGE=0 pytest ...` or re-run snapshot CLI with the same env |
| `Manifest mismatch` from verifier | Artifact tampering or stale files in snapshot directory | Re-run the snapshot (`--overwrite` if regenerating) so hashes & row counts realign |

## Reconciling manifest diffs

When the manifest verifier reports hash or row count mismatches:

1. Inspect the listed files in `resolver/snapshots/YYYY-MM/`.
2. Re-run the monthly snapshot with `--overwrite` if legitimate updates were
   made, or delete the directory before re-running to ensure a clean export.
3. Commit the regenerated manifest together with the updated CSV/Parquet files
   (CI will verify the pair).

## Troubleshooting DuckDB merge behaviour

- Force-enable merge writes locally by clearing `RESOLVER_DUCKDB_DISABLE_MERGE`
  (or setting it to `0`) and providing a DuckDB URL:
  ```bash
  export RESOLVER_DB_URL="duckdb:///$(pwd)/resolver/db/resolver.duckdb"
  export RESOLVER_DUCKDB_DISABLE_MERGE=0
  pytest -q resolver/tests/test_contracts_semantics_and_keys.py
  ```
- To emulate the merge-disabled CI job, set `RESOLVER_DUCKDB_DISABLE_MERGE=1`
  before running `pytest` or the snapshot CLI.

Keep this runbook close at hand when operating the monthly release or debugging
CI; it mirrors the guards enforced in automation so local runs match what CI
expects.
