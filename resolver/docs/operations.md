# Operations Run Book

This run book covers the main resolver workflows, including the ReliefWeb PDF branch. It is designed for engineers and analysts running the pipeline locally or in CI.

## Quick start

1. **Ingest connectors (offline stubs):**
   ```bash
   python resolver/ingestion/run_all_stubs.py --retries 2
   ```
2. **Validate staging schemas:**
   ```bash
   pytest -q resolver/tests/test_staging_schema_all.py
   ```
3. **Export and validate facts:**
   ```bash
   python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports
   python resolver/tools/validate_facts.py --facts resolver/exports/facts.csv
   ```
   When `RESOLVER_DB_URL` is set (for example `duckdb:///resolver.duckdb`), the exporter automatically dual-writes the
   normalized facts into DuckDB and creates any missing tables before inserting rows.
4. **Run precedence & deltas (optional for analytics):**
   ```bash
   python resolver/tools/precedence_engine.py --facts resolver/exports/facts.csv --cutoff YYYY-MM-30
   python resolver/tools/make_deltas.py --resolved resolver/exports/resolved.csv --out resolver/exports/deltas.csv
   ```
5. **One-command monthly snapshot (exports + freeze):**
   ```bash
   python -m resolver.cli.snapshot_cli make-monthly --ym YYYY-MM
   ```

   This orchestrates export, validation, precedence, deltas, and snapshot freezing. Dual writes occur automatically when
   `RESOLVER_DB_URL` is configured; pass `--write-db 1` or `--write-db 0` to force-enable or disable the DuckDB write step.
   The freezer also respects `RESOLVER_DB_URL`; when present it writes the snapshot tables into DuckDB automatically,
   ensuring date columns remain ISO-formatted and the schema is created if missing.

## Monthly snapshots in CI

- Workflow: `.github/workflows/publish_snapshot.yml`
- Triggers: manual dispatch (`workflow_dispatch`) and the 1st of each month at 02:00 Europe/Istanbul.
- Command executed: `python -m resolver.cli.snapshot_cli make-monthly --ym <year-month> --write-db 1`
- Artifacts: `resolver/snapshots/YYYY-MM/` uploaded to the run for download. When `SNAPSHOT_PUBLISH_TOKEN` is set the job pushes the snapshot directory back to the repository.

### Required GitHub secrets/vars

| Name | Purpose |
| --- | --- |
| `RESOLVER_DB_URL` | DuckDB connection string used for dual-writes (`duckdb:///...`). |
| `SNAPSHOT_PUBLISH_TOKEN` (optional) | Personal access token with repo write access to push snapshot folders. |
| `RESOLVER_SNAPSHOT_ENV` (optional) | Additional environment flags (e.g., feature toggles) exported before the CLI runs. |

Set repository variables for non-sensitive defaults such as `SNAPSHOT_EXPORT_CONFIG` or staging paths if they differ from repo defaults.

## ReliefWeb PDF local runs

The PDF branch can be exercised without network access by enabling the feature flags and relying on mocked text extraction:

```bash
RELIEFWEB_ENABLE_PDF=1 \
RELIEFWEB_PDF_ALLOW_NETWORK=0 \
RELIEFWEB_PDF_ENABLE_OCR=0 \
python resolver/ingestion/reliefweb_client.py
```

To run the mocked extractor tests:

```bash
pytest -q resolver/tests/ingestion/test_reliefweb_pdf.py
```

### Feature toggles

- `RELIEFWEB_ENABLE_PDF=1|0` — enable or disable the PDF branch entirely
- `RELIEFWEB_PDF_ENABLE_OCR=1|0` — allow OCR fallback when native text is sparse
- `RELIEFWEB_PDF_ALLOW_NETWORK=1|0` — control attachment downloads (CI keeps this at `0`)
- `RELIEFWEB_PPH_OVERRIDE_PATH=/path/to/pph.csv` — optional CSV for household overrides

## Logs and observability

- Ingestion writes plain text and JSONL logs to `resolver/logs/ingestion/` by default.
- Override destinations with `RUNNER_LOG_DIR=/tmp/resolver-logs`.
- Set verbosity via `RUNNER_LOG_LEVEL=DEBUG` and format via `RUNNER_LOG_FORMAT=json`.

## Continuous integration

The `resolver-ci` workflow executes offline smoke tests plus the ReliefWeb PDF unit suite. When the optional Markdown link checker is enabled it runs after the tests and reports broken intra-repo links in the job logs without failing the build.
