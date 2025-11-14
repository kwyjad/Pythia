# Ingestion playbook

## Connector configuration paths

Connector YAMLs are in the middle of a migration toward `resolver/ingestion/config/`, but the shared loader currently
prefers the legacy copies in `resolver/config/` so fast tests and older jobs keep working. A short guide covering the
search order, fallback behaviour, and how to clean up legacy duplicates lives in
[`ingestion/config_paths.md`](ingestion/config_paths.md). Refer to it when adding a new connector or migrating an
existing configuration.

## ACLED monthly fatalities

The ACLED connector uses an OAuth password-or-refresh flow documented by ACLED: it POSTs to
`https://acleddata.com/oauth/token`, caches the resulting bearer token, and never logs the credential itself—only metadata such
as expiry—to aid local debugging.【F:resolver/ingestion/acled_auth.py†L1-L167】 The HTTP client calls
`https://acleddata.com/api/acled/read?_format=json` with the bearer header, retries on 429/5xx responses, and writes the first
request’s URL/status to `diagnostics/ingestion/acled/http_diag.json` so pipeline summaries can report the last API interaction.
Successful responses hydrate a `pandas.DataFrame` where `event_date` is normalised to UTC, `fatalities` are coerced to integers,
and the optional `_format` parameter is enforced; downstream, the `monthly_fatalities` helper groups by ISO3 + month to produce
staging rows with consistent provenance fields.【F:resolver/ingestion/acled_client.py†L372-L1223】【F:scripts/ci/summarize_connectors.py†L27-L95】
Running the client from the CLI is as simple as:

```bash
python -c "from resolver.ingestion.acled_client import ACLEDClient; print(ACLEDClient().monthly_fatalities('2024-01-01','2024-02-29').head())"
```

The printed frame includes `iso3`, `month`, `fatalities`, `source`, and `updated_at` (UTC). Pass `countries=["KEN","ETH"]` to
restrict the aggregation to a subset of ISO3 codes when debugging region-specific pipelines.

## EM-DAT (PA)

The EM-DAT integration ships with a small GraphQL client and an offline stub so fast tests never hit the
network accidentally. Live requests are opt-in: pass `--network` (or set `EMDAT_NETWORK=1`) **and** expose a valid
`EMDAT_API_KEY`; otherwise `EmdatClient.fetch_raw()` raises `OfflineRequested` and callers fall back to the bundled stub
DataFrame.【F:resolver/ingestion/emdat_client.py†L466-L580】【F:resolver/tests/test_emdat_client_offline_smoke.py†L6-L36】

- **Endpoint & auth:** `POST https://api.emdat.be/v1` with an `Authorization: <EMDAT_API_KEY>` header on each request.
- **Hazard filters:** the query pins the four PA-relevant classification keys—drought (`nat-cli-dro-dro`), tropical
  cyclone (`nat-met-sto-tro`), riverine flood (`nat-hyd-flo-riv`), and flash flood (`nat-hyd-flo-fla`). Flood subtypes are
  merged downstream into `shock_type="flood"` for Resolver semantics.
- **Metric:** EM-DAT `Total Affected` is coerced to a non-negative integer and surfaced as Resolver “PA”. When the
  source omits `Total Affected`, the client falls back to the sum of `Affected`, `Injured`, and `Homeless` while
  intentionally ignoring deaths so the PA column remains aligned with EM-DAT guidance.
- **File sources:** Configurations that specify `source.type: file` (or declare explicit `sources` blocks) bypass the
  live-network gate. They continue to normalize local CSV inputs and return success even when `EMDAT_NETWORK=1` but no
  API key is present so fast tests stay green.
- **Monthly bucketing:** rows group by `start_year`/`start_month`. Records missing `start_month` are logged as
  `emdat.normalize.missing_month` and excluded from monthly aggregates while remaining visible in diagnostics.
- **Metadata:** `normalize_emdat_pa` carries through `as_of_date`, `publication_date` (preferring `last_update`), and the
  lowest contributing `disno` for traceability.

The export mapping in [`resolver/tools/export_config.yml`](../tools/export_config.yml) reshapes the staging CSV into the
canonical facts layout consumed by `freeze_snapshot`. People Affected totals (`pa`) become the unified `value` column with
`metric=affected` and `series_semantics=new`, while the monthly bucket (`ym`) and Resolver `shock_type` are surfaced as the
facts `ym` and `hazard_code` columns. Aligning the preview with this schema keeps the validator happy and unblocks the
DuckDB freeze step.

Invoke `python -m resolver.cli.resolver_cli emdat-to-duckdb --help` to review the available flags. Two quick-start
examples:

```bash
# Offline stub (default)
python -m resolver.cli.resolver_cli emdat-to-duckdb \
  --from 2021 --to 2021 --countries KEN --db ./resolver_data/emdat.duckdb

# Live pull (requires API key and --network)
EMDAT_API_KEY=*** python -m resolver.cli.resolver_cli emdat-to-duckdb \
  --from 2021 --to 2021 --countries KEN --db ./resolver_data/emdat.duckdb --network
```

Both runs normalise the frame via `normalize_emdat_pa` and upsert with `write_emdat_pa_to_duckdb`. The CLI prints the
standard success banner (`✅ Wrote …`) and emits probe logs (`emdat.probe.ok|…` / `emdat.probe.fail|…`) whenever
`--network` is enabled.【F:resolver/cli/emdat_to_duckdb.py†L1-L156】

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
