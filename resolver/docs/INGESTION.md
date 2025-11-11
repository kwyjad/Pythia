# Ingestion playbook

## Connector configuration paths

Connector YAMLs are in the middle of a migration toward `resolver/ingestion/config/`, but the shared loader currently
prefers the legacy copies in `resolver/config/` so fast tests and older jobs keep working. A short guide covering the
search order, fallback behaviour, and how to clean up legacy duplicates lives in
[`ingestion/config_paths.md`](ingestion/config_paths.md). Refer to it when adding a new connector or migrating an
existing configuration.

## DTM connector troubleshooting

### Missing `id_or_path`

The DTM client now runs a configuration preflight before any files are read. If
one or more entries in `resolver/ingestion/config/dtm.yml` omit the required
`id_or_path` field, the connector:

1. Writes a machine-readable report to `diagnostics/ingestion/dtm_config_issues.json`.
   Each invalid source is emitted with an `error` of `missing id_or_path` plus any
   other metadata the entry originally declared.
2. Emits a diagnostics row in `diagnostics/ingestion/connectors_report.jsonl`
   with `reason="missing id_or_path"`, `extras.invalid_sources`, and
   `extras.skipped_sources` summarising the skipped entries.
3. Continues with the remaining (valid) sources, writing header-only output when
   every entry was skipped. Pass `--fail-on-missing-config` (or set
   `DTM_STRICT=1`) to make the run exit immediately with code `2` instead.

Fix the report by adding a concrete `id_or_path` (Drive ID, S3 key, or absolute
path) to each invalid entry, then re-run the connector. The diagnostics line will
flip back to `status="ok"` and the config issues file will list `invalid: 0`.

### Debug triage

Set `LOG_LEVEL=DEBUG` to ask the connector for richer breadcrumbs. The resolved
sources snapshot (`diagnostics/ingestion/dtm_sources_resolved.json`) mirrors the
post-preflight source list so you can confirm inferred column names, resolved
paths, and skip reasons locally before a retry.

### Admin0 stock exports

The initial backfill enables the DTM admin0 stock dataset (`idps_present`) in
[`resolver/tools/export_config.yml`](../tools/export_config.yml). Expect
`diagnostics/ingestion/export_preview/facts.csv` to include rows with
`semantics=stock` so DuckDB writes can populate `facts_resolved` alongside the
delta table. Toggle the metric in the export config if you need to isolate
flows-only runs during local debugging.

## IDMC skeleton smoke test

The IDMC connector is currently offline-only and exists to wire fixtures, normalization, and diagnostics. Run the smoke test locally with:

```bash
python -m resolver.ingestion.idmc.cli --skip-network
```

The command appends a diagnostics row to `diagnostics/ingestion/connectors.jsonl` and writes a normalized preview CSV to `diagnostics/ingestion/idmc/normalized_preview.csv`.

When the CLI falls back to the HELIX `idus/last-180-days` feed it now records the
real row counts for fetched, normalized, and written data. The generated
`diagnostics/ingestion/idmc/summary.md` and the per-connector table therefore
show the fallback totals (instead of `0/0/0`) whenever the rescue path produced
rows.

HELIX runs now execute a strict fallback order: the connector first attempts the
GIDD endpoint and, on any HTTP failure or empty payload, immediately downloads
the HELIX dump. Because the dump is a single JSON export the client disables
per-month chunking in HELIX mode and rolls the data up to month-end aggregates
after download. Diagnostics expose the selected endpoint via
`helix_endpoint=gidd|idus_last180` and set `zero_rows_reason` to
`helix_http_error` or `helix_empty` so it is obvious why HELIX produced zero
rows in a given window.

The ingestion summary now captures the real fetch/normalize/write counts even
when HELIX takes the fallback path, populating `rows_fetched`,
`rows_normalized`, and `rows_written` for the diagnostics table. The
`resolver-initial-backfill` workflow cleans both `resolver/staging/` and
`diagnostics/ingestion/` before each run, executes the HELIX single-shot
unconditionally (skipping the legacy `RESOLVER_SKIP_IDMC` gate), and forces the
`gidd` → `idus_last180` fallback order when the primary endpoint is empty. After
exporting the staged files the `idmc_to_duckdb` helper appends a dedicated
DuckDB verification block to the job summary via `--append-summary`, so the Step
Summary always lists the inserted and updated row counts alongside the
connector diagnostics.

Fallback data sources occasionally surface ISO3 identifiers under alternate
column names (for example `CountryISO3`). The connector normalizes these
variants to a canonical `iso3` column before applying any country filters and
skips filtering entirely when a fallback frame is empty. Diagnostics now note
when the ISO filter was applied or skipped so empty fallback payloads no longer
raise `KeyError: 'iso3'` during fast tests.

The companion `idmc_to_duckdb` helper accepts both `--db-url` and the shorter
`--db` flag. Empty stock exports no longer trigger a hard failure—the command
emits a warning, keeps the success banner (`✅ Wrote …`), and returns `0` so it
can run as part of automated pipelines focused on flow updates.
