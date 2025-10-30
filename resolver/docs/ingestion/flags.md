# Ingestion connector feature flags

Resolver connectors share a small set of command-line flags and environment
variables to keep offline fixtures deterministic while still allowing targeted
runs in CI or local debugging. CLI arguments always win over environment
variables, which in turn override static values in `resolver/ingestion/config/*`.

## Common switches

| CLI flag | Environment variable | Purpose |
| --- | --- | --- |
| `--skip-network` | `IDMC_FORCE_CACHE_ONLY=1` | Force fixture/cache mode and bypass HTTP probes. |
| `--strict-empty` | — | Exit with status code `2` when zero rows are written. |
| `--no-date-filter` | `IDMC_NO_DATE_FILTER=1` | Disable server/client-side date filtering. |
| `--window-days=<int>` | `IDMC_WINDOW_DAYS=<int>` | Limit IDU monthly flow rows to a trailing window (inclusive). |
| `--only-countries=COD,SDN` | `IDMC_ONLY_COUNTRIES="COD,SDN"` | Filter client-side to a comma-separated list of ISO3 codes. |
| `--series=flow,stock` | `IDMC_SERIES="flow"` | Restrict normalisation to specific connector series. |

`IDMC_FORCE_CACHE_ONLY` falls back to fixtures when set to a truthy value
(`1`, `true`, `yes`) and defaults to live mode otherwise. The `*_ONLY_COUNTRIES`
pattern uppercases ISO3 codes automatically; blank entries are ignored.

## Diagnostics snapshot

Every connector run writes a JSON line to
`diagnostics/ingestion/connectors.jsonl`. The `run_flags` block echoes the
parsed switches (after applying environment overrides) and the `debug` block
summarises derived decisions such as the selected country/series set or cache
mode. This payload is the first stop when triaging zero-row runs in CI.

## Example

```bash
export IDMC_ONLY_COUNTRIES="COD,SDN"
python -m resolver.ingestion.idmc.cli --skip-network --series=flow
```

The CLI records `run_flags.only_countries == ["COD", "SDN"]` and
`debug.selected_series == ["flow"]`, making the run reproducible in local
investigations or nightly automation.
