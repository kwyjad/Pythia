# Static ingestion assets

- `iso3_master.csv` holds the canonical list of ISO3 codes used by ingestion
  connectors. The IDMC client resolves `countries: []` (or a missing `countries`
  block) to this roster so an empty list means “fetch every country” rather than
  “fetch none,” and honours environment overrides via `IDMC_COUNTRIES` or
  `IDMC_COUNTRIES_FILE` when you need to narrow the scope for a run.【F:resolver/ingestion/utils/country_utils.py†L118-L209】【F:resolver/ingestion/idmc/cli.py†L223-L323】 The CLI also accepts `--start`/`--end` (or `RESOLVER_START_ISO`/`RESOLVER_END_ISO`, falling back to `BACKFILL_*`) so CI workflows can inject the desired window while leaving local runs free to choose their own span.【F:resolver/ingestion/idmc/cli.py†L300-L384】 When no explicit window is provided the CLI logs a warning and returns a zero-row outcome unless `EMPTY_POLICY=fail`, keeping runs green while still producing diagnostics.【F:resolver/ingestion/idmc/cli.py†L590-L676】 Resolver workflows additionally pass `--enable-export --write-outputs --series flow,stock` and `IDMC_NETWORK_MODE=live` so staged `flow.csv` (and `stock.csv` when present) come straight from live API responses.【F:.github/workflows/resolver-initial-backfill.yml†L149-L167】【F:resolver/ingestion/idmc/cli.py†L629-L655】 Live runs now build PostgREST filters (`displacement_start_date=gte.*`, `displacement_end_date=lte.*`) and batch ISO3 codes; opt into the HDX CSV mirror with `--allow-hdx-fallback` or `IDMC_ALLOW_HDX_FALLBACK=1` when the API fails so backfills keep landing rows and diagnostics show both the HTTP attempt and the rescue path.【F:resolver/ingestion/idmc/cli.py†L212-L320】【F:resolver/ingestion/idmc/client.py†L506-L719】【F:resolver/ingestion/idmc/client.py†L958-L1362】
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
