# DTM displacement connector

The DTM connector fetches monthly displacement **flows** (newly displaced people) by
calling the official DTM API or, when credentials are not available, by generating a
synthetic offline export that exercises the full pipeline. The connector always writes
structured diagnostics and appends a line to `diagnostics/connectors_report.jsonl`
whether it runs, skips, or encounters an error.

## Quick start

```bash
# Optional: use DEBUG logging
export LOG_LEVEL=INFO

# Run the offline smoke path (no network requests)
python -m resolver.ingestion.dtm_client --offline-smoke

# Run with real credentials (requires dtmapi>=3.0.0)
export DTM_API_KEY="<your key>"
python -m resolver.ingestion.dtm_client
```

### Command-line flags

- `--debug` – enable verbose logging for local debugging.
- `--offline-smoke` – synthesise a small dataset without importing `dtmapi`.
- `--no-date-filter` – fetch every row regardless of `RESOLVER_START_ISO` / `RESOLVER_END_ISO`.
- `--strict-empty` – exit with code `1` when the run produced zero rows in online mode.
- `--output PATH` – override the default CSV destination (`resolver/staging/dtm_displacement.csv`).

## Credentials

The connector looks for credentials in this order:

1. `DTM_API_KEY`
2. `DTM_SUBSCRIPTION_KEY`

Logs only show the final four characters of the detected key. If neither variable is
present the run exits with status `skipped` and reason `auth_missing` while still
writing the connectors report line.

## Diagnostics bundle

Each invocation creates or updates:

| Path | Description |
| --- | --- |
| `resolver/staging/dtm_displacement.csv` | Canonical monthly displacement flows. |
| `resolver/staging/dtm_displacement.csv.meta.json` | Schema + row-count metadata. |
| `diagnostics/sample_dtm_displacement.csv` | Up to 20 canonical rows for quick inspection. |
| `diagnostics/ingestion/dtm/summary.json` | Run summary: window, counts, negative-flow flag, mode, and status. |
| `diagnostics/ingestion/dtm/request_log.jsonl` | Per-request timing (only in online/API mode). |
| `diagnostics/connectors_report.jsonl` | One JSON object per run with status, reason, and elapsed seconds. |

Missing directories are created automatically before any file is written, which keeps
CI smoke tests green even when diagnostics were never initialised.

## Skip reasons

| Reason | When it appears | Exit code |
| --- | --- | --- |
| `sdk_missing` | `dtmapi` is not importable. | 0 |
| `auth_missing` | Neither `DTM_API_KEY` nor `DTM_SUBSCRIPTION_KEY` is configured. | 0 |
| `discovery_empty` | Country discovery returned zero ISO3 codes when `api.countries` is empty. | 0 |
| `disabled` | `resolver/ingestion/config/dtm.yml` has `enabled: false`. | 0 |
| `error` | Any unexpected exception. | 1 |

A skipped run still emits `summary.json` with the captured reason and appends a new
connectors report line so CI dashboards stay accurate.

## Stock-to-flow conversion

The API returns displacement **stocks**. Rows are converted into monthly flows by:

1. Deduplicating each `(iso3, admin1, month_start)` combination and keeping the latest
   `reportingDate`.
2. Calculating `flow_t = stock_t - stock_{t-1}` and dropping the first observation per
   admin area.
3. Preserving negative deltas (returns) and flagging them in `summary.json` via
   `has_negative_flows`.
4. Aggregating admin1 flows into an admin0 row by summing the monthly totals and using
   the latest `as_of` timestamp from the children.

The canonical CSV schema lives in
`resolver/ingestion/schemas/dtm_displacement.schema.yml` and is documented in
[`SCHEMAS.md`](../../SCHEMAS.md#stagingdtm_displacement).

## Troubleshooting

- **Offline smoke succeeds but live mode skips** – confirm that `dtmapi>=3.0.0` is
  installed (`poetry install --extras ingestion`) and that the API key environment
  variable is set in the same shell.
- **`discovery_empty`** – set `api.countries` in `resolver/ingestion/config/dtm.yml` to
  an explicit ISO3 list or re-run after the DTM API outage is resolved.
- **Negative flows look suspicious** – the connector keeps returns by design. Inspect
  `diagnostics/sample_dtm_displacement.csv` and the upstream reporting dates to verify
  the decrease.
- **Strict empty failures** – rerun without `--strict-empty` to keep diagnostics while
  investigating the upstream data gap.
