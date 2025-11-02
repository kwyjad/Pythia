# Public APIs & Contracts

## resolver.query.db_reader

- `fetch_deltas_point(conn, *, ym, iso3, hazard_code, cutoff, preferred_metric)`
- `fetch_resolved_point(conn, *, ym, iso3, hazard_code, cutoff, preferred_metric)`

## resolver.db.conn_shared

- `normalize_duckdb_url(db_url)`
- `get_shared_duckdb_conn(db_url, *, force_reopen=False)`
- `clear_cached_connection(db_url)`

## resolver.ingestion.idmc

- `config.IdmcConfig` now exposes `api.base_url`, named `endpoints`, and `cache`
  controls (`dir`, `ttl_seconds`, `force_cache_only`).
- `client.fetch(cfg, *, skip_network=False, window_days=30, only_countries=None,
  base_url=None, cache_ttl=None)` returns the offline fixtures plus an
  `idus_view_flat` DataFrame and diagnostics with probe/http/cache fields.
- `http.http_get(url, *, headers=None, timeout=10.0, retries=2, backoff_s=0.5)`
  performs a guarded GET request (adds `Authorization` when `IDMC_API_TOKEN` is
  set) and surfaces retry diagnostics.
- `cache.cache_key`, `cache.cache_get`, and `cache.cache_put` provide a small
  file-backed cache for HTTP responses used by the connector and CLI.

### Upstream source reference

| Source | Base URL | Endpoints | Auth mode | Terms URL |
| --- | --- | --- | --- | --- |
| IDMC (IDU) | `https://www.internal-displacement.org` | `idus_json`, `idus_geo` (REST) | Optional bearer token via `IDMC_API_TOKEN` | _Pending legal confirmation_ |

## TypeScript / JavaScript

(none)

## scripts.ci.summarize_connectors

- `build_markdown(...)` always renders the legacy diagnostics matrix header and divider
  even when no connector rows are available. The header is:

  `| Connector | Mode | Status | Reason | HTTP 2xx/4xx/5xx (retries) | Fetched | Normalized | Written | Kept | Dropped | Parse errors | Logs | Meta rows | Meta |`

  Counts are emitted as individual integer columns (`Fetched`, `Normalized`, `Written`,
  `Kept`, `Dropped`, `Parse errors`) defaulting to `0` when diagnostics omit the values.
  `Status` prefers the raw connector status (`extras.status_raw`) when provided.
- The "Config used" section always includes two literal lines: `Config source: <value>`
  and `Config: <path>`.
- Config path resolution order: connector metadata (`extras.config.*`),
  `diagnostics/ingestion/**/why_zero.json`, `diagnostics/ingestion/**/manifest.json`,
  the DTM fallback (`resolver/config/dtm.yml`), and finally `unknown` when nothing can be
  discovered.
- `resolver/diagnostics/ingestion/summarize_connectors.py` re-exports the CI summarizer
  helpers so imports and diagnostics CLI entrypoints share a single implementation.

- **Boolean env parsing:** All boolean-like flags are read via `getenv_bool`, treating `0/false/no/off/""` as False.
- **Superreport meta-row display:** Aggregated meta row count of `0` renders as **“—”** (em dash), not the numeral `0`.
