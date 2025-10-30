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

## Feature flags

IDMC honours the shared [connector feature flags](flags.md) while keeping runs
deterministic offline. CLI arguments override environment variables which, in
turn, win over static config defaults.

| CLI flag | Environment variable | Notes |
| --- | --- | --- |
| `--skip-network` | `IDMC_FORCE_CACHE_ONLY=1` | Skip HTTP probes and fall back to cache/fixtures. |
| `--strict-empty` | — | Exit with status code `2` when zero rows survive. |
| `--no-date-filter` | `IDMC_NO_DATE_FILTER=1` | Disable both server and client date filters. |
| `--window-days=<int>` | `IDMC_WINDOW_DAYS=<int>` | Trailing window (days) applied to `displacement_date`. |
| `--only-countries=COD,SDN` | `IDMC_ONLY_COUNTRIES="COD,SDN"` | Parsed into an uppercase ISO3 list for client filtering. |
| `--series=flow,stock` | `IDMC_SERIES="flow"` | Defaults to `flow`; selecting `stock` currently yields zero rows for IDU. |
| `--base-url=<url>` | `IDMC_BASE_URL=<url>` | Override the IDU endpoint root (useful for staging mirrors). |
| `--cache-ttl=<seconds>` | `IDMC_CACHE_TTL_S=<seconds>` | Override the cache lifetime for online runs. |

`run_flags` in `diagnostics/ingestion/connectors.jsonl` echoes the resolved
values (`skip_network`, `only_countries`, `series`, etc.) and the `debug` block
reports derived context such as `selected_countries_count` or the cache mode
used for the run.

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
  keeping the maximum value observed (`dup_event`).
- **Non-negative guardrail:** negative values are dropped before deduping
  (`negative_value`).
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

## Exports

IDMC’s monthly **flow** series can be exported as resolution-ready facts when
the Resolver export gate is enabled. Exports stay disabled in fast tests unless
explicitly opted in.

1. Enable the feature flag:

   ```bash
   export RESOLVER_EXPORT_ENABLE_IDMC=1  # or RESOLVER_EXPORT_ENABLE_FLOW=1
   ```

2. Run the connector with output writing enabled:

   ```bash
   python -m resolver.ingestion.idmc.cli \
     --skip-network \
     --write-outputs \
     --out-dir artifacts/idmc
   ```

   The `--out-dir` argument defaults to `artifacts/idmc` (override with
   `IDMC_OUT_DIR`).

The run writes `idmc_facts_flow.csv` and, when optional dependencies are
available, `idmc_facts_flow.parquet` to the requested directory. Rows follow the
shared facts contract documented in `SCHEMAS.md` and include:

- `iso3`
- `as_of_date`
- `metric` (`idp_displacement_new_idmc`)
- `value`
- `series_semantics` (`new`)
- `source` (`IDMC`)

Connectors diagnostics echo the export status (enabled/disabled), row counts,
and output paths under the `export` key.
