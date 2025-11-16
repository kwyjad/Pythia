# Ingestion playbook

## Connector configuration paths

Connector YAMLs are in the middle of a migration toward `resolver/ingestion/config/`, but the shared loader currently
prefers the legacy copies in `resolver/config/` so fast tests and older jobs keep working. A short guide covering the
search order, fallback behaviour, and how to clean up legacy duplicates lives in
[`ingestion/config_paths.md`](ingestion/config_paths.md). Refer to it when adding a new connector or migrating an
existing configuration.

## Writing monthly snapshots to DuckDB

The `freeze_snapshot.py` tool is responsible for materialising monthly facts into DuckDB once CSV artefacts exist on disk.
After generating `facts.csv`, `facts_resolved.csv`, and `facts_deltas.csv` for a given month, the freezer invokes the
internal `_maybe_write_db(...)` helper. This helper:

- Checks whether DuckDB writes are enabled (`--write-db=1` or `RESOLVER_WRITE_DB=1` alongside a DuckDB URL).
- Loads the month’s CSVs into memory and calls `duckdb_io.write_snapshot(...)` to update `facts_resolved`,
  `facts_deltas`, and the `snapshots` metadata table in one transaction.
- Emits row counts, series semantics histograms, and the canonical DB path to `diagnostics/ingestion/freeze_db.json`
  and appends a Markdown block to `diagnostics/ingestion/summary.md`.
- On failure, prints a clear warning and appends a “Freeze Snapshot — DB write” error section to the summary so CI
  diagnostics surface the problem without relying on raw stack traces.

The DuckDB helper mirrors the contracts exercised by the fast tests:

- **Parity:** After the freeze step runs, the DuckDB `facts_resolved` table must contain the same monthly rows as
  the snapshot CSV. This behaviour is asserted by `test_exporter_dual_writes_to_duckdb`.
- **Idempotency:** Running `_maybe_write_db(...)` multiple times for the same month keeps the `snapshots` table at a
  single row per `ym` and preserves the expected `facts_resolved` counts. The dedicated parity test in
  `test_freeze_snapshot_write_db_parity.py` and `test_duckdb_idempotency.py::test_dual_writes_idempotent` enforce it.
- **Flows:** When a deltas CSV is not provided (e.g. EM-DAT preview runs), the helper derives a deltas frame from the
  canonical facts CSV so `facts_deltas` matches the preview row-for-row. The
  `test_emdat_duckdb_write.test_emdat_export_and_freeze_to_duckdb` fast test verifies this behaviour.

### `ym` handling for flows (`facts_deltas`)

For monthly flows (EM-DAT, ACLED, synthetic tests), the `ym` column is the primary key for DuckDB writes. The exporter and
freezer normalise it so downstream helpers always receive a `YYYY-MM` string:

- When `ym` is missing, it is derived from `as_of_date` during export finalisation.
- When `ym` exists but is blank, it is backfilled—first from `as_of_date`, then from `publication_date` if necessary.
- The DuckDB layer (`duckdb_io.write_snapshot(...)`) therefore continues to enforce its strict `YYYY-MM` contract without
  needing connector-specific exceptions.

Fast tests covering the freeze → DuckDB path (`test_exporter_dual_writes_to_duckdb`, `test_dual_writes_idempotent`, and
`test_emdat_export_and_freeze_to_duckdb`) assert this behaviour stays in place.

If DuckDB writes are disabled or the URL is missing, `_maybe_write_db` logs the skip reason and leaves the diagnostics in
place for downstream verification stages.

### Freeze snapshot: CLI vs function behaviour

- CLI usage (`python -m resolver.tools.freeze_snapshot ...`) is optimised for generating snapshot files. Unless `--write-db=1`
  is passed (or `RESOLVER_WRITE_DB=1` is set explicitly), the CLI **does not** touch DuckDB even when `RESOLVER_DB_URL` is
  available in the environment. This guarantees that tests calling the CLI without flags never mutate the database and keeps
  `test_exporter_dual_writes_to_duckdb` focused on the export-time write.
- Programmatic usage (`freeze_snapshot(..., write_db=True, db_url=...)`) continues to call `duckdb_io.write_snapshot(...)` in
  a single transaction. Tests and backfill workflows opt into this path when they need DB writes by either setting the
  argument explicitly or exporting `RESOLVER_WRITE_DB=1`.
- The `_maybe_write_db(...)` helper honours the `RESOLVER_WRITE_DB` environment variable when the function-level `write_db`
  argument is left as `None`, making it easy for CI pipelines to enable DB writes without changing callsites while still
  defaulting to "off" for ad-hoc CLI usage.

### Snapshot parity and EM-DAT flow passthrough

- `freeze_snapshot` now builds the snapshot artefacts (`facts.csv`, `facts.parquet`, and `facts_resolved.*`) directly from the
  month-filtered preview using the same `_prepare_resolved_for_db(...)` / `_prepare_deltas_for_db(...)` helpers as
  `export_facts`. The DuckDB tables therefore match the snapshot rows 1:1 when freeze is run without DB writes, satisfying
  the `test_exporter_dual_writes_to_duckdb` contract.
- When the preview looks like EM-DAT People Affected flows (all metrics are PA metrics and/or the publisher mentions
  EM-DAT/CRED), the freezer enforces a passthrough guard: the number of deltas rows prepared must equal the number of
  preview rows with `series_semantics="new"`. If the counts diverge, the freezer logs a warning and falls back to a
  straight subset of the preview rows so every flow row is written exactly once, as asserted by
  `test_emdat_export_and_freeze_to_duckdb`.
- Two new diagnostics sections (`Freeze snapshot — parity inputs` and `Freeze snapshot — flow passthrough`) record the month,
  preview row counts, prepared deltas counts, and whether passthrough fired. This makes it easy to confirm parity/flow
  behaviour in CI logs without opening the parquet files.

### EM-DAT flows in snapshots

EM-DAT People Affected rows represent monthly flows that must populate `facts_deltas`. The freezer detects EM-DAT PA metrics
(`metric` in `{affected, total_affected, people_affected, in_need, pin, pa}`) and normalises their
`series_semantics`/`semantics` columns to `"new"` when they are missing or blank. When an entire frame looks like canonical
EM-DAT facts (publisher/source fields mention EM-DAT and the expected hazard columns exist) the freezer now reuses the same
`_prepare_resolved_for_db(...)` / `_prepare_deltas_for_db(...)` helpers as the exporter. This ensures:

- The per-month snapshot (`facts.parquet`, `facts_resolved.csv`) contains the same EM-DAT rows that `export_facts` wrote to
  DuckDB, preserving the DB ↔ snapshot parity asserted in `test_exporter_dual_writes_to_duckdb`.
- When `freeze_snapshot(..., write_db=True)` is invoked with the export preview EM-DAT facts, every preview row is written
  into `facts_deltas` with `series_semantics="new"`, as exercised by `test_emdat_export_and_freeze_to_duckdb`.
- ACLED and other connectors remain unchanged because the normalisation and exporter-helper reuse only fire when **all**
  metrics in the frame are recognised EM-DAT flow metrics.

## ACLED monthly fatalities

The ACLED connector uses an OAuth password-or-refresh flow documented by ACLED: it POSTs to
`https://acleddata.com/oauth/token`, caches the resulting bearer token, and never logs the credential itself—only metadata such
as expiry—to aid local debugging.【F:resolver/ingestion/acled_auth.py†L1-L167】 The HTTP client calls
`https://acleddata.com/api/acled/read?_format=json` with the bearer header, retries on 429/5xx responses, and writes the first
request’s URL/status to `diagnostics/ingestion/acled/http_diag.json` so pipeline summaries can report the last API interaction.
Successful responses hydrate a `pandas.DataFrame` where `event_date` is normalised to UTC, `fatalities` are coerced to integers,
and the optional `_format` parameter is enforced; downstream, the `monthly_fatalities` helper groups by ISO3 + month to produce
staging rows with consistent provenance fields.【F:resolver/ingestion/acled_client.py†L1223-L1390】【F:scripts/ci/summarize_connectors.py†L27-L95】
Running the client from the CLI is as simple as:

```bash
python -c "from resolver.ingestion.acled_client import ACLEDClient; print(ACLEDClient().monthly_fatalities('2024-01-01','2024-02-29').head())"
```

The printed frame includes `iso3`, `month`, `fatalities`, `source`, and `updated_at` (UTC). Pass `countries=["KEN","ETH"]` to
restrict the aggregation to a subset of ISO3 codes when debugging region-specific pipelines.

### ACLED authentication precedence

The ACLED ingestion client resolves credentials in the following order:

1. `ACLED_ACCESS_TOKEN` (modern opaque bearer token), if set.
2. `ACLED_TOKEN` (legacy environment variable). When found, it is mirrored into
   `ACLED_ACCESS_TOKEN` for compatibility with downstream helpers.
3. `ACLED_REFRESH_TOKEN`, which triggers the OAuth `refresh_token` grant.
4. `ACLED_USERNAME` and `ACLED_PASSWORD`, which fall back to the OAuth
   `password` grant.

Fast tests in `resolver/tests/test_acled_auth_tokens.py` assert this
precedence, ensuring that environment-provided tokens short-circuit the HTTP
grants, while refresh and password flows still work when no token is supplied.

### Flows vs stocks

Flows such as ACLED fatalities represent per-period totals. They are written to `facts_deltas` with `series_semantics="new"`, meaning each monthly value stands alone rather than representing a change from the previous month. Stocks—including IDP headcounts or people in need—populate `facts_resolved` (and may emit derived deltas when changes are calculated), but they are conceptually distinct from the ACLED-style monthly flows.

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

### Validation: EM-DAT vs ACLED

- The monthly snapshot pipeline (`freeze_snapshot.py`) includes a strict validator for **EM-DAT People Affected** data, which
  checks against the `emdat_pa` schema and fails if required fields or consistency checks are violated.
- For connectors like **ACLED** that provide **flow metrics** (e.g., events, monthly fatalities), the EM-DAT validator is
  *not* applied. These facts are still normalized and written into `facts_deltas` with `series_semantics="new"`, but are not
  required to satisfy EM-DAT-specific schema rules.
- This allows us to backfill and freeze ACLED monthly flows without being blocked by EM-DAT validation rules that do not
  apply to ACLED.

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

All downstream tools invoked by `resolver-initial-backfill` now report
failures directly into `diagnostics/ingestion/summary.md`. Each CLI wraps its
entrypoint with a best-effort call to `scripts.ci.append_error_to_summary`,
emitting sections such as “Export Facts — DB write”, “Precedence Engine — CLI
error”, or “Verify DuckDB Counts — error” that include the DB URL, relevant
row counts, and the exception details. The next failing run therefore points to
the precise step and context that needs attention without digging through the
workflow logs.

### Backfill diagnostics

The derive-freeze job appends stage markers to the ingestion summary so a
single artifact reveals how far the workflow progressed. Each major step (facts
export, DuckDB verification, monthly freeze, and the LLM context build) writes
start/end banners via `scripts.ci.append_stage_to_summary`. When a step
terminates early the matching `failed` marker remains in the log alongside the
structured error block from the underlying CLI, making it obvious which phase
needs attention in the next retry.
