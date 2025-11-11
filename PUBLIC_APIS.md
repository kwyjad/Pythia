# Public APIs & Contracts

## resolver.query.db_reader

- `fetch_deltas_point(conn, *, ym, iso3, hazard_code, cutoff, preferred_metric)`
- `fetch_resolved_point(conn, *, ym, iso3, hazard_code, cutoff, preferred_metric)`

## resolver.cli.idmc_to_duckdb

The CLI returns deterministic exit codes that downstream automation can depend on:

| Exit code | Meaning |
| --- | --- |
| `0` | Success (warnings allowed when `--strict` is unset) |
| `1` | Unhandled error while running the CLI |
| `2` | Strict mode with warnings |
| `4` | `--write-db` enabled but no facts were written |

## resolver.db.conn_shared

- `normalize_duckdb_url(db_url)`
- `get_shared_duckdb_conn(db_url, *, force_reopen=False)`
- `clear_cached_connection(db_url)`

## resolver.io.files_locator

- `discover_files_root(preferred: Optional[pathlib.Path] = None) -> pathlib.Path` honours caller-provided directories, then `RESOLVER_FILES_ROOT`, before falling back to fast-fixture exports or `resolver/tests/data`.

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

## resolver.api.app

- FastAPI application exposing `/resolve` and `/resolve_batch` routes. Each request accepts a `backend` parameter (`"db"` or `"files"`) so parity tests can compare results across storage backends.

## TypeScript / JavaScript

(none)
