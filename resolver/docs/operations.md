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

- Workflow: `.github/workflows/resolver-monthly.yml` (manual dispatch remains available; `.github/workflows/publish_snapshot.yml` is now a fallback for unusual recoveries).
- Schedule: cron `0 2 1 * *` → 02:00 on the first day of the month in Europe/Istanbul. The workflow derives `SNAPSHOT_TARGET_YM` from the previous month in Istanbul time and exports matching `RESOLVER_START_ISO`/`RESOLVER_END_ISO` bounds for the connectors.
- Pipeline: run live connectors (skipping any missing credentials), execute `python -m resolver.cli.snapshot_cli make-monthly --ym $SNAPSHOT_TARGET_YM --write-db 1 --db-url data/resolver.duckdb`, build the rolling 12‑month LLM context bundle, and prepare both directories for artifact upload.
- Artifacts & release: uploads `resolver/snapshots/$SNAPSHOT_TARGET_YM/**` plus `context/` as a single artifact and updates or creates a GitHub Release tagged `resolver-$SNAPSHOT_TARGET_YM` containing a compressed bundle.
- Runbook checks: review the Actions summary table for healthy `resolved_rows` (zero rows usually means a connector was skipped), confirm the context bundle file sizes look reasonable, and verify the release upload succeeded. Skipped connectors log clear warnings so the run still completes green.
- SLO: finish within 75 minutes so the artifact and release are available by 03:30 Istanbul. If the run fails, resolve the connector issue and re-run; when live data is unavailable fall back to `publish_snapshot.yml` and note the variance in the monthly summary.

## Initial backfill workflow

- Workflow: `.github/workflows/resolver-initial-backfill.yml` (manual dispatch only).
- Launch it from the **Actions** tab by choosing **Resolver — Initial Backfill**, overriding the `months_back` input (default `12`) or `only` connector filter as needed.
- The `ingest` job runs `resolver/ingestion/run_all_stubs.py --mode real`, wiring connector secrets from the environment. Connectors without credentials stay in header-only mode so the workflow still succeeds.
- The `derive-freeze` job loops over the computed year-month list, runs `export_facts → precedence_engine → make_deltas`, and freezes each snapshot via `python -m resolver.tools.freeze_snapshot --facts … --month <YYYY-MM> --write-db 1` so DuckDB mirrors remain in sync.
- The `context` job calls `python -m resolver.tools.build_llm_context --months 12 --outdir context/` to produce `facts_last12.jsonl` and `facts_last12.parquet`.
- Final artifacts:
  - `resolver/snapshots/<YYYY-MM>/` with `facts_resolved.*`, optional `facts_deltas.*`, and `manifest.json` per month.
  - `context/facts_last12.jsonl` and `context/facts_last12.parquet` ready for Forecaster ingestion.
    Each row covers a `(ym, iso3, hazard_code, metric)` tuple with `unit`, rounded `value`, and `series="new"`.
    The Forecaster LLM reads the JSONL file directly to hydrate its monthly context embeddings;
    the matching Parquet provides the same schema for analytics spot checks.
- After a run completes download the combined artifact (named `resolver-initial-backfill-<run_id>`) and verify the snapshot manifests plus context bundle row counts before handing off to forecasting ops.

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

## Troubleshooting connectors

Use the GitHub Actions diagnostics table (rendered by `scripts/ci/summarize_connectors.py`) as the first-line triage tool:

- **Status / Reason:** `skipped` rows with `disabled: config` mean the connector’s YAML intentionally disables it; `missing secret …` indicates the corresponding secret wasn’t present in the workflow environment.
- **2xx/4xx/5xx (retries):** A spike in `4xx` responses usually means credentials expired or the request window was rejected—regenerate tokens or narrow the backfill window. Large `5xx` counts point to upstream instability; consider retrying later or temporarily skipping the connector.
- **Rows (f/n/w):** Values of `0/0/0` after a supposedly real run mean only headers were written. Inspect the connector logs under `resolver/logs/ingestion/<connector>.log` to confirm whether the upstream returned empty payloads or parsing failed.
- **Coverage (ym / as_of):** Ensure the `ym_min → ym_max` range matches the requested backfill window. Narrow ranges or `—` entries suggest the connector only delivered part of the period and may need a rerun.
- **Details panel:** Review the top ISO3/hazard samples and `rate_limit_remaining` values. A near-zero rate limit hints that you should increase backoff or reduce concurrency before relaunching the job.

## Continuous integration

The `resolver-ci` workflow executes offline smoke tests plus the ReliefWeb PDF unit suite. When the optional Markdown link checker is enabled it runs after the tests and reports broken intra-repo links in the job logs without failing the build.
