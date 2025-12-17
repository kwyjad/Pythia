# Public APIs & Contracts

## FastAPI (pythia.api)

- **Authentication**: Protected endpoints expect `Authorization: Bearer <token>`.
  - Server token is sourced from `PYTHIA_API_TOKEN` (preferred) or `PYTHIA_API_KEY` (legacy fallback).
  - Legacy header `X-Pythia-Token` is still accepted for backwards compatibility.
  - If no token environment variable is set, auth is disabled (for local development only).
- `GET /v1/question_bundle`
  - Query params: `question_id` (required), `hs_run_id`, `forecaster_run_id`, `include_llm_calls` (bool, default false), `include_transcripts` (bool, default false), `limit_llm_calls` (int, default 200).
  - Returns a bundle containing: the question row, HS run/scenarios/country report, ensemble SPD rows plus per-model SPD rows, question context/resolutions, and optional `llm_calls` (including transcripts only when requested).

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
- `client.fetch(cfg, *, network_mode="live", window_days=30, only_countries=None,
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
