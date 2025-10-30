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
| `--map-hazards` | `IDMC_MAP_HAZARDS=1` | Append Resolver hazard metadata (code/label/class) and related diagnostics. |

`run_flags` in `diagnostics/ingestion/connectors.jsonl` echoes the resolved
values (`skip_network`, `only_countries`, `series`, etc.) and the `debug` block
reports derived context such as `selected_countries_count` or the cache mode
used for the run.

## Scale-up guidance

Online runs can now be tuned for larger windows and polite throughput. The
following flags keep the connector responsive while avoiding rate-limit bans:

```bash
python -m resolver.ingestion.idmc.cli \
  --window-days=60 \
  --rate=0.5 \
  --max-concurrency=1 \
  --max-bytes=$((10 * 1024 * 1024)) \
  --chunk-by-month
```

- `--rate` / `IDMC_REQ_PER_SEC` wraps all HTTP calls in a token bucket. Setting
  the value to `0` disables throttling, but the default `0.5` keeps peak load
  within IDMC’s expectations.
- `--max-concurrency` / `IDMC_MAX_CONCURRENCY` controls the chunk fan-out. The
  connector remains conservative (`1`) by default; higher values should only be
  used when mirroring traffic through an internal cache.
- `--max-bytes` / `IDMC_MAX_BYTES` streams large responses directly to
  `.cache/idmc/` so JSON payloads no longer sit twice in memory.
- `--chunk-by-month` partitions wide windows into month-sized requests. This
  primarily improves cache segmentation and parsing behaviour; the upstream API
  still responds with a flat feed.

For wider windows (e.g. 180 days) prefer `--rate=0.3` and keep concurrency at
`1`. Diagnostics now include `performance`, `rate_limit`, and `chunks` blocks
with request counts, wire/body bytes, throughput, planned waits, and per-month
timings. The environment flag `IDMC_TEST_NO_SLEEP=1` remains available for fast
offline tests: it records planned waits without actually sleeping.

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

## Hazard mapping (optional)

An opt-in post-process tags each normalized row with Resolver’s canonical hazard
codes. Enable it via `--map-hazards` or `IDMC_MAP_HAZARDS=1`. When active the
normalized frame gains three additional columns:

- `hazard_code`
- `hazard_label`
- `hazard_class`

The current heuristic covers the following codes:

| Code | Label | Class |
| --- | --- | --- |
| `FL` | Flood | natural |
| `DR` | Drought | natural |
| `TC` | Tropical Cyclone | natural |
| `HW` | Heat Wave | natural |
| `PHE` | Public Health Emergency | epidemic |
| `CU` | Civil Unrest | human-induced |
| `ACE` | Armed Conflict – Escalation | human-induced |

Disaster rows are mapped via keyword matching on the IDU hazard metadata
(`hazard_category`, `hazard_type`, etc.). Conflict rows default to `ACE` unless
riot/protest cues promote them to `CU`. Internal displacement feeds do not
currently emit cross-border influx (`DI`) records. Rows that do not match any
rule expose `hazard_class` based on the upstream `displacement_type` and remain
otherwise null for code/label.

Diagnostics add a `hazards` block with `enabled`, `unmapped_hazard` counts and a
path to `hazard_unmapped_preview` when applicable. The preview CSV lists a small
sample of unmatched rows for manual inspection.

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

## Attribution & Provenance

Every CLI run writes a machine-readable provenance manifest to
`diagnostics/ingestion/idmc/manifest.json`. The manifest captures the connector
arguments, the resolved environment snapshot (`IDMC_*` variables), reachability
probe outputs, HTTP/cache rollups, normalization counts, export status, and a
default attribution block (source name, placeholder terms URL, citation, license
note, and method note). When export previews are enabled, a sibling
`*.manifest.json` sits next to the generated facts CSV so downstream tooling can
trace lineage without cross-referencing logs.

Secrets are redacted automatically. Any key containing `token`, `secret`, or
`password` (and known headers like `Authorization`) is replaced with
`***REDACTED***` in manifests and in the shared
`diagnostics/ingestion/connectors.jsonl` log. The manifest therefore records that
`IDMC_API_TOKEN` was supplied without ever persisting the raw credential.

The manifest blocks of interest are:

- `run`: CLI command, parsed arguments, sanitized environment, effective base
  URL, resolved endpoints, timings, and mode (`offline`/`online`).
- `reachability`: DNS/TCP/TLS probe data including the observed egress IP when
  available.
- `http`: Request counters, retry summaries, status codes, latency percentiles,
  and cache hits/misses surfaced by the client diagnostics.
- `cache`: Cache directory/mode information.
- `normalize`: Row counts before/after normalization plus drop-reason histogram.
- `export`: Export enablement, feature flags, written paths, and manifest
  pointers for exported artifacts.
- `notes`: Supplemental flags such as the zero-rows rescue block.

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

## Precedence candidates (opt-in)

Use the precedence adapter to emit Resolver-standard candidates for IDMC and,
optionally, run the config-driven selector locally. The feature is fully
offline and does not modify existing workflows.

```bash
python -m resolver.ingestion.idmc.cli \
  --skip-network \
  --write-candidates \
  --candidates-out diagnostics/ingestion/idmc/idmc_candidates.csv
```

- `--write-candidates` (or `IDMC_WRITE_CANDIDATES=1`) emits
  `idmc_candidates.csv` with rich metadata columns (source system, collection
  type, coverage, freshness).
- `--run-precedence` (or `IDMC_RUN_PRECEDENCE=1`) reuses those candidates to run
  the config in `tools/precedence_config.yml` and writes the selected rows to
  `--precedence-out` (defaults to `diagnostics/ingestion/idmc/idmc_selected.csv`).
- `--precedence-config` overrides the config path when experimenting with custom
  rules.

Diagnostics list both artifact paths under `samples` and summarise row counts in
the `exports` block of `connectors.jsonl`.
