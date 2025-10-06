# Connectors Catalog

This catalog documents the primary Resolver connectors, their configuration sources, entry points, and staging outputs. All connectors emit CSVs with the canonical staging header documented in [Data contracts](data_contracts.md).

| Connector | Config | Entrypoint | Output CSV | Notes |
|---|---|---|---|---|
| ACLED | [`ingestion/config/acled.yml`](../ingestion/config/acled.yml) | [`ingestion/acled_client.py`](../ingestion/acled_client.py) | `staging/acled.csv` | Conflict events aggregated to humanitarian impact metrics. |
| DTM | [`ingestion/config/dtm.yml`](../ingestion/config/dtm.yml) | [`ingestion/dtm_client.py`](../ingestion/dtm_client.py) | `staging/dtm.csv` | Pulls displacement monitors, normalises ISO3 and hazard taxonomy. |
| EMDAT | [`ingestion/config/emdat.yml`](../ingestion/config/emdat.yml) | [`ingestion/emdat_client.py`](../ingestion/emdat_client.py) | `staging/emdat.csv` | Disaster loss archive; often used for historical baselines. |
| FEWS NET | (stubbed) | [`ingestion/fews_stub.py`](../ingestion/fews_stub.py) | `staging/fews.csv` | Stub produces canonical header for offline smoke tests. |
| GDACS | [`ingestion/config/gdacs.yml`](../ingestion/config/gdacs.yml) | [`ingestion/gdacs_client.py`](../ingestion/gdacs_client.py) | `staging/gdacs.csv` | Global Disaster Alert and Coordination System feeds for rapid-onset events. |
| HDX | [`ingestion/config/hdx.yml`](../ingestion/config/hdx.yml) | [`ingestion/hdx_client.py`](../ingestion/hdx_client.py) | `staging/hdx.csv` | Fetches curated humanitarian indicators from HDX datasets. |
| IFRC GO | [`ingestion/config/ifrc_go.yml`](../ingestion/config/ifrc_go.yml) | [`ingestion/ifrc_go_client.py`](../ingestion/ifrc_go_client.py) | `staging/ifrc_go.csv` | Handles GO API pagination and report metadata quirks. |
| IPC | [`ingestion/config/ipc.yml`](../ingestion/config/ipc.yml) | [`ingestion/ipc_client.py`](../ingestion/ipc_client.py) | `staging/ipc.csv` | Integrated Food Security Phase Classification (IPC/CH) feed. |
| UNHCR (Population) | [`ingestion/config/unhcr.yml`](../ingestion/config/unhcr.yml) | [`ingestion/unhcr_client.py`](../ingestion/unhcr_client.py) | `staging/unhcr.csv` | Global population statistics, includes asylum/refugee cohorts. |
| UNHCR ODP | [`ingestion/config/unhcr_odp.yml`](../ingestion/config/unhcr_odp.yml) | [`ingestion/unhcr_odp_client.py`](../ingestion/unhcr_odp_client.py) | `staging/unhcr_odp.csv` | Operational Data Portal situational reports. |
| WFP mVAM | [`ingestion/config/wfp_mvam.yml`](../ingestion/config/wfp_mvam.yml) | [`ingestion/wfp_mvam_client.py`](../ingestion/wfp_mvam_client.py) | `staging/wfp_mvam.csv` | Mobile Vulnerability Analysis and Mapping indicators. |
| WHO PHE | [`ingestion/config/who_phe.yml`](../ingestion/config/who_phe.yml) | [`ingestion/who_phe_client.py`](../ingestion/who_phe_client.py) | `staging/who_phe.csv` | Public health emergency surveillance, includes case counts. |
| WorldPop | [`ingestion/config/worldpop.yml`](../ingestion/config/worldpop.yml) | [`ingestion/worldpop_client.py`](../ingestion/worldpop_client.py) | `staging/worldpop.csv` | Population denominators for coverage checks. |
| ReliefWeb (API) | [`ingestion/config/reliefweb.yml`](../ingestion/config/reliefweb.yml) | [`ingestion/reliefweb_client.py`](../ingestion/reliefweb_client.py) | `staging/reliefweb.csv` | Handles WAF 202 challenges; header-only output when blocked. |
| ReliefWeb (PDF) | [`ingestion/config/reliefweb.yml`](../ingestion/config/reliefweb.yml) (PDF toggles) | [`ingestion/reliefweb_client.py`](../ingestion/reliefweb_client.py) | `staging/reliefweb_pdf.csv` | Scores & downloads PDFs, native parsing then OCR fallback, HH→people conversion, tier-2 monthly deltas. |

### Shared conventions

Across staging outputs the following columns are standard:

- `as_of`, `source`, `event_id`, `country_iso3`, `month`, `hazard_type`, `metric`, `value`
- Provenance fields such as `source_type`, `source_url`, and `definition_text`
- Tier metadata (`tier`, `confidence`) used by the precedence engine

Refer to the [data contracts](data_contracts.md) and generated [SCHEMAS.md](../../SCHEMAS.md) for exhaustive field definitions.
