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
- normalizes the IDU feed into Resolver’s monthly **new displacement** series.

The first ten normalized rows are saved to
`diagnostics/ingestion/idmc/normalized_preview.csv` and the drop reason
histogram is mirrored to `diagnostics/ingestion/idmc/drop_reasons.json` for
inspection.

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

## Normalization rules

- **ISO3 resolution:** values are upper-cased and validated via the Resolver
  country catalogue. Rows without a valid ISO3 are dropped (`no_iso3`).
- **Date precedence:** `displacement_date` → `displacement_start_date` →
  `displacement_end_date`. Values are coerced to month-end (`as_of_date`).
- **Value column:** `figure` is parsed as a numeric value; rows without a valid
  figure are removed (`no_value_col`).
- **Windowing:** if configured, `as_of_date` must fall within the inclusive
  `date_window` (`date_out_of_window`).
- **Duplicates:** rows are deduplicated on `(iso3, as_of_date, metric)` by
  keeping the maximum value observed (`duplicates_dropped`).
- **Output schema:** `metric=idp_displacement_new_idmc`,
  `series_semantics=new`, `source=IDMC`.

## Diagnostics payload

Each CLI run appends a JSON line to `diagnostics/ingestion/connectors.jsonl`
with:

- `http` counters (`requests`, `retries`, last status, latency percentiles) and
  cache hit/miss counts;
- `probe` results from the reachability check;
- `timings` for probe/fetch/normalize/total spans (ms);
- a `drop_reasons` histogram and links to the saved artifacts in `samples`;
- a `zero_rows` rescue block when no rows survive filtering, including the
  selectors (`only_countries`, `window_days`) and a human-readable note.

Use `--skip-network` or `IDMC_FORCE_CACHE_ONLY=1` to keep runs deterministic.
Fixtures remain available under `resolver/ingestion/idmc/fixtures/` for offline
testing and CI.
