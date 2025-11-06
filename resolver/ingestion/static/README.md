# Static ingestion assets

- `iso3_master.csv` holds the canonical list of ISO3 codes used by ingestion
  connectors. The IDMC client resolves `countries: []` (or a missing `countries`
  block) to this roster so an empty list means “fetch every country” rather than
  “fetch none,” and honours environment overrides via `IDMC_COUNTRIES` or
  `IDMC_COUNTRIES_FILE` when you need to narrow the scope for a run.【F:resolver/ingestion/utils/country_utils.py†L118-L209】【F:resolver/ingestion/idmc/cli.py†L223-L323】 The CLI accepts `--start`/`--end`; when they are omitted it first consults `RESOLVER_START_ISO`/`RESOLVER_END_ISO` (falling back to `BACKFILL_*`). If both the flags and env vars are absent the CLI now leaves the window unset, writes a `reason: "no_window"` why-zero payload, skips `fetch()`, and exits with success for `EMPTY_POLICY=allow|warn` (returning `2` when `EMPTY_POLICY=fail`).【F:resolver/ingestion/idmc/cli.py†L223-L736】 Exporters no longer need to pass `--write-outputs`: setting `RESOLVER_OUTPUT_DIR` automatically enables it, redirects `--out-dir` to `${RESOLVER_OUTPUT_DIR}/idmc`, and writes `resolver/staging/idmc/flow.csv` with the `new_displacements` payload sourced from Helix (`idmc_gidd`) after every normalization run; fixture and cache-only test modes also emit a header-only `flow.csv` so staging checks succeed even when normalization returns zero rows.【F:resolver/ingestion/idmc/cli.py†L340-L438】【F:resolver/ingestion/idmc/staging.py†L9-L26】【F:resolver/ingestion/idmc/cli.py†L720-L910】 CI now appends `--series flow --chunk-by-month --start … --end … --enable-export --write-outputs --allow-fallback`, relying on the defaults for staging writes and the gated HDX CSV rescue when live requests fail. The fallback queries CKAN’s `package_show` for the configured package (`idmc.hdx.package_id` in the YAML or `IDMC_HDX_PACKAGE_ID`), picks the best displacement CSV, rolls it up to monthly `new_displacements`, and records the selected `resource_url` for diagnostics and `why_zero.json`; override the CKAN base with `idmc.hdx.base_url` or `IDMC_HDX_BASE_URL` when pointing at a mirror.【F:.github/workflows/resolver-initial-backfill.yml†L162-L209】【F:resolver/ingestion/idmc/config.py†L63-L224】【F:resolver/ingestion/idmc/client.py†L924-L1415】 Live runs now hit Helix’s `/external-api/gidd/displacements` endpoint, passing the requested ISO3 batch and date window so diagnostics report the exact path; opt into the HDX CSV mirror with `--allow-hdx-fallback` or `IDMC_ALLOW_HDX_FALLBACK=1` when the API fails so backfills keep landing rows and diagnostics show both the HTTP attempt and any rescue path.【F:resolver/ingestion/idmc/cli.py†L212-L323】【F:resolver/ingestion/idmc/client.py†L924-L1415】
- HTTP behaviour is tunable: set `IDMC_HTTP_CONNECT_TIMEOUT_S` /
  `IDMC_HTTP_TIMEOUT_CONNECT`, `IDMC_HTTP_READ_TIMEOUT_S` /
  `IDMC_HTTP_TIMEOUT_READ`, and optionally `IDMC_HTTP_VERIFY=false` to skip TLS
  verification when debugging corporate proxy issues. The client always sends
  `Accept: application/json` and an explicit `User-Agent` and records the
  classified error kind (`connect_timeout`, `ssl_error`, `dns_error`, etc.)
  alongside the HTTP status buckets so diagnostics explain why live fetches are
  empty.【F:resolver/ingestion/idmc/http.py†L1-L266】【F:resolver/ingestion/idmc/client.py†L560-L1232】 The CI reachability probe at
  `scripts/ci/probe_idmc_reachability.py` resolves DNS, performs a TLS handshake
  and a live GET against the Helix displacement endpoint, captures the runner egress IP,
  and writes `diagnostics/ingestion/idmc/probe.json` plus a Markdown summary
  consumed by `scripts/ci/summarize_connectors.py` so GitHub jobs surface the
  exact network failure mode.【F:scripts/ci/probe_idmc_reachability.py†L1-L214】【F:scripts/ci/summarize_connectors.py†L1378-L2071】 A companion probe, `scripts/ci/probe_hdx_reachability.py`, performs a `package_show` request against `preliminary-internal-displacement-updates`, checks the selected CSV resource, and appends an “HDX Reachability” block to the workflow summary so reviewers can see the status codes and URL the fallback relied on.【F:scripts/ci/probe_hdx_reachability.py†L1-L171】【F:scripts/ci/summarize_connectors.py†L1900-L1960】
- The IDMC connector now probes the PostgREST view to select an appropriate
  date column (preferring `displacement_date`) and logs per-chunk HTTP errors
  without failing the run, preserving diagnostics for empty months.【F:resolver/ingestion/idmc/client.py†L204-L360】【F:resolver/ingestion/idmc/client.py†L980-L1352】【F:resolver/ingestion/idmc/cli.py†L952-L1190】

- Normalization coerces date windows and input timestamps to real datetimes.
  When dates cannot be parsed or the upstream feed returns no rows, the
  connector now exits gracefully with zero normalized rows while still writing
  diagnostics, keeping `EMPTY_POLICY=allow|warn` runs healthy.
- `iso3_roster.csv` is a slimmer export that mirrors the master list and is used
  by some legacy scripts.
- `dtm_admin0_fake.csv` ships fixture rows that let the DTM exporter run in
  offline smoke tests.【F:resolver/tools/export_facts.py†L180-L212】
