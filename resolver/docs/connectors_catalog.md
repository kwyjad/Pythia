# Connectors Catalog

This catalog documents the primary Resolver connectors, their configuration sources, entry points, and staging outputs. All connectors emit CSVs with the canonical staging header documented in [Data contracts](data_contracts.md).

| Connector | Config file | Requires secret? | Secret name(s) | Notes |
|---|---|---|---|---|
| ACLED | [`ingestion/config/acled.yml`](../ingestion/config/acled.yml) | Yes | `ACLED_REFRESH_TOKEN` (or `ACLED_TOKEN` / `ACLED_USERNAME`+`ACLED_PASSWORD`) | Pulls a 450‑day window at 1,000 records per request and skips unless ACLED credentials are present.【F:resolver/ingestion/config/acled.yml†L1-L5】【F:resolver/ingestion/run_all_stubs.py†L268-L277】 |
| DTM | [`ingestion/config/dtm.yml`](../ingestion/config/dtm.yml) | No | – | Consumes file-based sources and converts displacement stocks to monthly flows via the `diff_nonneg` rule.【F:resolver/ingestion/config/dtm.yml†L2-L6】 |
| EM-DAT | [`ingestion/config/emdat.yml`](../ingestion/config/emdat.yml) | No | – | Reads licensed EM-DAT extracts from the configured file path and allocates multi-month events linearly.【F:resolver/ingestion/config/emdat.yml†L2-L9】 |
| GDACS | [`ingestion/config/gdacs.yml`](../ingestion/config/gdacs.yml) | No | – | Fetches a 365-day event window with five retry attempts and 2s backoff to stay within API limits.【F:resolver/ingestion/config/gdacs.yml†L2-L8】 |
| HDX | [`ingestion/config/hdx.yml`](../ingestion/config/hdx.yml) | No | – | Queries HDX with HXL tag preference, up to 500 datasets per run, filtering by humanitarian plan topics.【F:resolver/ingestion/config/hdx.yml†L1-L18】 |
| IFRC GO | [`ingestion/config/ifrc_go.yml`](../ingestion/config/ifrc_go.yml) | Optional | `GO_API_TOKEN` | Covers a 45-day window across Field Reports, Appeals, and Situation Reports; an optional token header helps when rate-limited.【F:resolver/ingestion/config/ifrc_go.yml†L2-L17】【F:resolver/ingestion/ifrc_go_client.py†L5-L14】 |
| IPC | [`ingestion/config/ipc.yml`](../ingestion/config/ipc.yml) | No | – | Enables specific IPC feeds via YAML and defaults to unauthenticated downloads.【F:resolver/ingestion/config/ipc.yml†L1-L5】 |
| ReliefWeb | [`ingestion/config/reliefweb.yml`](../ingestion/config/reliefweb.yml) | Optional | `RELIEFWEB_APPNAME` | Pulls a 30-day window with 0.6 s pauses between 100-row pages; the connector falls back to the configured app name when no secret is set.【F:resolver/ingestion/config/reliefweb.yml†L20-L25】【F:resolver/ingestion/reliefweb_client.py†L1211-L1221】 |
| IDMC (IDU) | [`config/idmc.yml`](../config/idmc.yml) | Optional | `IDMC_API_TOKEN` | CI runs probe the public endpoint, apply polite defaults (`--chunk-by-month`, `--write-candidates`, `IDMC_REQ_PER_SEC≈0.3`, `IDMC_MAX_CONCURRENCY=1`), and export flow facts when `RESOLVER_EXPORT_ENABLE_FLOW` resolves truthy; set `RESOLVER_SKIP_IDMC=1` to bypass when no token is available.【F:.github/workflows/resolver-monthly.yml†L35-L118】【F:.github/workflows/resolver-initial-backfill.yml†L72-L164】【F:scripts/ci/run_connectors.py†L23-L120】 |
| UNHCR (Population) | [`ingestion/config/unhcr.yml`](../ingestion/config/unhcr.yml) | No | – | Requests monthly asylum applications with a 60-day window, 500-row page limit, and two-page cap per run.【F:resolver/ingestion/config/unhcr.yml†L2-L19】 |
| UNHCR ODP | [`ingestion/config/unhcr_odp.yml`](../ingestion/config/unhcr_odp.yml) | Yes | `UNHCR_ODP_USERNAME`, `UNHCR_ODP_PASSWORD`, `UNHCR_ODP_CLIENT_ID`, `UNHCR_ODP_CLIENT_SECRET` | Scrapes monthly sea-arrival widgets (max one page) and only runs when the full credential set is present.【F:resolver/ingestion/config/unhcr_odp.yml†L2-L11】【F:resolver/ingestion/run_all_stubs.py†L278-L291】 |
| WFP mVAM | [`ingestion/config/wfp_mvam_sources.yml`](../ingestion/config/wfp_mvam_sources.yml) | No | – | Harvests indicators from the configured source list; authentication block is empty by default.【F:resolver/ingestion/config/wfp_mvam_sources.yml†L1-L3】 |
| WHO PHE | [`ingestion/config/who_phe.yml`](../ingestion/config/who_phe.yml) | No | – | Pulls disease-specific feeds defined in YAML, with empty `auth` placeholders for optional credentials.【F:resolver/ingestion/config/who_phe.yml†L1-L9】 |
| WorldPop | [`ingestion/config/worldpop.yml`](../ingestion/config/worldpop.yml) | No | – | Loads the unadjusted national totals product for the listed years to supply denominator data.【F:resolver/ingestion/config/worldpop.yml†L1-L14】 |

### Shared conventions

Across staging outputs the following columns are standard:

- `as_of`, `source`, `event_id`, `country_iso3`, `month`, `hazard_type`, `metric`, `value`
- Provenance fields such as `source_type`, `source_url`, and `definition_text`
- Tier metadata (`tier`, `confidence`) used by the precedence engine

Refer to the [data contracts](data_contracts.md) and generated [SCHEMAS.md](../../SCHEMAS.md) for exhaustive field definitions.
