# IDMC (IDU) Connector

The IDMC connector now supports a cautious online mode while preserving the
deterministic offline fixtures that power the fast test suite.

## Quick start

```bash
python -m resolver.ingestion.idmc.cli --window-days=14 --only-countries=SDN
```

The command:

- runs a reachability probe (DNS/TCP/TLS/HEAD) to explain potential failures;
- attempts to download the IDU `idus_view_flat` payload with retries and
  exponential backoff;
- falls back to the file cache or bundled fixtures when the network is
  unavailable; and
- normalizes the legacy CSV fixtures for compatibility with existing charts.

## Flags and environment overrides

| Flag | Purpose |
| --- | --- |
| `--skip-network` | Force offline mode (fixtures/cache only). |
| `--window-days` | Client-side window applied to `displacement_date` (default `30`). |
| `--only-countries=COD,SDN` | Filter to a comma-separated list of ISO3 codes. |
| `--base-url` | Override the IDU base URL (defaults to `https://backend.idmcdb.org`). |
| `--cache-ttl` | Override the cache TTL in seconds for this run. |

Environment variables provide the same controls:

- `IDMC_BASE_URL`, `IDMC_REACHABILITY_TIMEOUT`
- `IDMC_CACHE_DIR`, `IDMC_CACHE_TTL_S`, `IDMC_FORCE_CACHE_ONLY`
- `IDMC_API_TOKEN` (optional bearer token; IDU does not currently require it)

## Diagnostics and caching

- Raw HTTP responses are cached under `.cache/idmc/` and mirrored to
  `diagnostics/ingestion/idmc/raw/` when downloaded.
- The reachability probe summary and HTTP/cache counters are appended to
  `diagnostics/ingestion/connectors.jsonl` alongside the normalized preview.
- Use `--skip-network` or `IDMC_FORCE_CACHE_ONLY=1` to keep runs deterministic.

Fixtures remain available under `resolver/ingestion/idmc/fixtures/` for offline
testing and CI.
