# Static ingestion assets

- `iso3_master.csv` holds the canonical list of ISO3 codes used by ingestion
  connectors. The IDMC client resolves `countries: []` (or a missing `countries`
  block) to this roster so an empty list means “fetch every country” rather than
  “fetch none,” and honours environment overrides via `IDMC_COUNTRIES` or
  `IDMC_COUNTRIES_FILE` when you need to narrow the scope for a run.【F:resolver/ingestion/utils/country_utils.py†L118-L209】【F:resolver/ingestion/idmc/cli.py†L223-L323】 The CLI accepts `--start`/`--end`; when they are omitted it first consults `RESOLVER_START_ISO`/`RESOLVER_END_ISO` (falling back to `BACKFILL_*`). If both the flags and env vars are absent the CLI now leaves the window unset, writes a `reason: "no_window"` why-zero payload, skips `fetch()`, and exits with success for `EMPTY_POLICY=allow|warn` (returning `1` when `EMPTY_POLICY=fail`).【F:resolver/ingestion/idmc/cli.py†L223-L716】 Exporters no longer need to pass `--write-outputs`: setting `RESOLVER_OUTPUT_DIR` automatically enables it, redirects `--out-dir` to `${RESOLVER_OUTPUT_DIR}/idmc`, and writes `resolver/staging/idmc/flow.csv` with the `new_displacements`/`idmc_idu` payload after every normalization run.【F:resolver/ingestion/idmc/cli.py†L340-L438】【F:resolver/ingestion/idmc/export.py†L9-L47】 CI now only appends `--series flow --chunk-by-month --network-mode live`, relying on the defaults for staging writes.【F:.github/workflows/resolver-initial-backfill.yml†L149-L207】 Live runs build PostgREST filters (`displacement_date=gte.*`, `displacement_date=lte.*`) and batch ISO3 codes; opt into the HDX CSV mirror with `--allow-hdx-fallback` or `IDMC_ALLOW_HDX_FALLBACK=1` when the API fails so backfills keep landing rows and diagnostics show both the HTTP attempt and any rescue path.【F:resolver/ingestion/idmc/cli.py†L212-L320】【F:resolver/ingestion/idmc/client.py†L506-L719】【F:resolver/ingestion/idmc/client.py†L958-L1362】
- HTTP behaviour is tunable: set `IDMC_HTTP_CONNECT_TIMEOUT_S`,
  `IDMC_HTTP_READ_TIMEOUT_S`, and optionally `IDMC_USER_AGENT`
  (falling back to `RELIEFWEB_USER_AGENT`). Defaults stay at 5 s/25 s and both
  the reachability probe and `summary.md` capture the effective values plus any
  timeout counters so HDX fallbacks (enabled explicitly via
  `--allow-hdx-fallback`/`IDMC_ALLOW_HDX_FALLBACK`) are obvious when PostgREST
  times out.【F:resolver/ingestion/idmc/http.py†L15-L253】【F:resolver/ingestion/idmc/cli.py†L560-L1280】【F:scripts/ci/probe_idmc_reachability.py†L1-L210】
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
