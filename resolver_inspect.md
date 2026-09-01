# Resolver DuckDB — Comprehensive Inspection Report

_Generated: 2026-09-01 08:26:12 UTC_
_Database size: 15.3 MB (16,003,072 bytes)_

## Data Freshness — At a Glance

_Newest data point and row count per source, with a staleness flag computed against this report's generation time. ✅ current · ⚠️ past its refresh cadence · ❌ empty · — undated/unparseable. The parenthesised name is the column the 'Newest' value came from. Thresholds are per-source refresh cadences, not hard SLAs._

| Source | Rows | Newest | Status |
|--------|-----:|--------|--------|
| Resolution facts · observations | 1 | 2026-08 (ym) | ✅ 31d |
| ACLED monthly fatalities | _n/a_ | (no table) | — |
| ACLED political events | _n/a_ | (no table) | — |
| GDELT conflict indicators | _n/a_ | (no table) | — |
| NMME seasonal forecasts | _n/a_ | (no table) | — |
| ReliefWeb reports | _n/a_ | (no table) | — |
| ACAPS INFORM severity | _n/a_ | (no table) | — |
| ACAPS daily monitoring | _n/a_ | (no table) | — |
| ACAPS risk radar | _n/a_ | (no table) | — |
| ACAPS humanitarian access | _n/a_ | (no table) | — |
| HDX Signals | _n/a_ | (no table) | — |
| ENSO state | _n/a_ | (no table) | — |
| Seasonal TC outlooks | _n/a_ | (no table) | — |

## Pipeline Stage Completeness

_Row presence per pipeline stage, in run order. Tells you at a glance whether this DB is resolver-only or a full end-to-end run. ✓ populated · ✗ empty (stage not run / no output)._

| # | Stage | Table | Rows | |
|--:|-------|-------|-----:|:-:|
| 1 | Resolver facts | `facts_resolved` | 1 | ✓ |
| 2 | HS runs | `hs_runs` | _(no table)_ | ✗ |
| 3 | HS triage | `hs_triage` | _(no table)_ | ✗ |
| 4 | Questions | `questions` | _(no table)_ | ✗ |
| 5 | Forecasts (raw) | `forecasts_raw` | _(no table)_ | ✗ |
| 6 | Forecasts (ensemble) | `forecasts_ensemble` | _(no table)_ | ✗ |
| 7 | Sibyl forecasts | `sibyl_forecasts` | _(no table)_ | ✗ |
| 8 | Resolutions (ground truth) | `resolutions` | _(no table)_ | ✗ |
| 9 | Scores | `scores` | _(no table)_ | ✗ |
| 10 | Calibration weights | `calibration_weights` | _(no table)_ | ✗ |
| 11 | Calibration advice | `calibration_advice` | _(no table)_ | ✗ |
| 12 | Source coverage map | `source_coverage` | _(no table)_ | ✗ |

## Inject Readiness — 122-Country Target List

_Per-source coverage of the HS target list (`horizon_scanner/hs_country_list.txt`): how many target countries have at least one row in each prompt-inject source. 'Expectation' calibrates the reading — global sources should cover ~all targets; crisis- or hazard-scoped sources are partial by design. Expand the details block to see which target countries are missing._

| Inject source | Target coverage | Missing | Expectation |
|---------------|----------------:|--------:|-------------|
| ACLED monthly fatalities | _no table_ | — | ~all |
| GDACS event history (facts_resolved event_occurrence) | 0/122 | 122 | most (only countries with recorded events) |
| Food security Phase 3+ (FEWS NET/IPC) | 0/122 | 122 | partial (monitored countries only) |
| Conflict forecast · VIEWS | _no table_ | — | ~all |
| Conflict forecast · conflictforecast.org | _no table_ | — | ~all |
| Conflict forecast · ACLED CAST | _no table_ | — | partial (CAST coverage) |
| NMME seasonal forecasts | _no table_ | — | ~all |
| ReliefWeb reports | _no table_ | — | most (report-generating situations) |
| ACLED political events | _no table_ | — | most (countries with recent events) |
| GDELT conflict indicators | _no table_ | — | most (media-covered countries) |
| ACAPS INFORM severity | _no table_ | — | partial (crisis countries only) |
| HDX Signals | _no table_ | — | partial (signal-emitting countries) |
| Seasonal TC context cache | _no table_ | — | partial (TC-exposed countries only) |
| CrisisWatch (latest edition) | _no table_ | — | partial (ICG-monitored countries) |

<details>
<summary>Missing target countries per source</summary>

- **GDACS event history (facts_resolved event_occurrence)** (122): AFG, AGO, ARM, AZE, BDI, BEN, BFA, BGD, BIH, BLR, BOL, BRA, BWA, CAF, CHL, CHN, CIV, CMR, COD, COG, COL, COM, CPV, CRI, CUB, CYP, DJI, DOM, DZA, ECU, EGY, ERI, ESH, ETH, FJI, GAB, GEO, GHA, GIN, GMB, GNB, GNQ, GTM, GUY, HND, HRV, HTI, IDN, IND, IRN, IRQ, ISR, JOR, KAZ, KEN, KGZ, KHM, KOR, LAO, LBN, … (+62 more)
- **Food security Phase 3+ (FEWS NET/IPC)** (122): AFG, AGO, ARM, AZE, BDI, BEN, BFA, BGD, BIH, BLR, BOL, BRA, BWA, CAF, CHL, CHN, CIV, CMR, COD, COG, COL, COM, CPV, CRI, CUB, CYP, DJI, DOM, DZA, ECU, EGY, ERI, ESH, ETH, FJI, GAB, GEO, GHA, GIN, GMB, GNB, GNQ, GTM, GUY, HND, HRV, HTI, IDN, IND, IRN, IRQ, ISR, JOR, KAZ, KEN, KGZ, KHM, KOR, LAO, LBN, … (+62 more)

</details>

## 1. Table Inventory

**Total tables: 2**

File 0.0 MB | free list 0.0 MB (0 blocks) | unattributed 0.0 MB (free list, catalog, metadata)

| Table | Rows | Columns | Est. MB | % of file |
|-------|-----:|--------:|--------:|----------:|
| `facts_resolved` | 1 | 7 | — | — |
| `haz_raw_reliefweb_docs` | 3,000 | 2 | — | — |

_Sizes are estimated from DISTINCT storage blocks per table (metadata only, no table scan). A large free list means the file is holding space no live data needs — DuckDB reuses freed blocks but never truncates, so only a copy to a fresh database reclaims it._

## 2. Resolver Core Tables

### facts_resolved
Total rows: **1**

| hazard_code | metric | rows | countries | min_ym | max_ym |
|-------------|--------|-----:|----------:|--------|--------|
| FL | affected | 1 | 1 | 2026-08 | 2026-08 |


### facts_deltas
_Table does not exist._

### emdat_pa
_Table does not exist._

### acled_monthly_fatalities
_Table does not exist._

## 2.5 PA Resolution Machine (haz_* tables)

_The machine has not run against this DB (no haz_* tables). It is populated by resolver_update.yml Phase 2.5 (live months) and the nightly Hazard Backcast workflow._

## 3. Horizon Scanner Tables

### hs_runs
_Table does not exist._

### hs_triage
_Table does not exist._

### hs_hazard_tail_packs
_Table does not exist._

### hs_adversarial_checks
_Table does not exist._

### hs_country_reports
_Table does not exist._

### hs_scenarios
_Table does not exist._

## 4. Questions & Forecasts

### questions
_Table does not exist._

### question_research
_Table does not exist._

### forecasts_raw
_Table does not exist._

### forecasts_ensemble
_Table does not exist._

### scenarios
_Table does not exist._

## 5. Scoring & Calibration

### resolutions
_Table does not exist._

### scores
_Table does not exist._

### calibration_weights
_Table does not exist._

### calibration_advice
_Table does not exist._

### bucket_definitions
_Table does not exist._

### bucket_centroids
_Table does not exist._

## 6. LLM Calls & Telemetry

### llm_calls
_Table does not exist._

### Batch-API staged-pipeline state
_Tables do not exist (pre-batch-pipeline DB)._

### question_run_metrics
_Table does not exist._

## 7. Structured Data Connectors

### seasonal_forecasts
_Table does not exist._

### conflict_forecasts
_Table does not exist._

### reliefweb_reports
_Table does not exist._

### acled_political_events
_Table does not exist._

### acaps_inform_severity
_Table does not exist._

### acaps_inform_severity_trend
_Table does not exist._

### acaps_risk_radar
_Table does not exist._

### acaps_daily_monitoring
_Table does not exist._

### acaps_humanitarian_access
_Table does not exist._

_`ipc_phases` is a legacy table from the retired `pythia/ipc_phases.py` connector (dead code) — deliberately not tracked here; the active IPC path writes `facts_resolved` via `resolver/connectors/ipc_api.py`._

### crisiswatch_entries (ICG CrisisWatch)
_Table does not exist._

### HDX Signals
_Note: HDX Signals data is cached as a local CSV file, not stored in DuckDB. HDX Signals health cannot be checked from the DB alone. Check the `hs_country_evidence` artifact for HDX Signals availability._

## 8. Game-Theoretic & Prediction Market Tables

### gtmc1_runs
_Table does not exist._

### gtmc1_actors
_Table does not exist._

### pm_checks
_Table does not exist._

## 9. Provenance & Metadata

### run_provenance
_Table does not exist._

### meta_runs
_Table does not exist._

### manifests
_Table does not exist._

### snapshots
_Table does not exist._

## 10. Full Schema Reference

<details>
<summary>Click to expand full column definitions for all tables</summary>

### facts_resolved

| # | column | type | notnull | default | pk |
|--:|--------|------|:-------:|---------|:--:|
| 0 | `ym` | VARCHAR |  |  |  |
| 1 | `iso3` | VARCHAR |  |  |  |
| 2 | `hazard_code` | VARCHAR |  |  |  |
| 3 | `metric` | VARCHAR |  |  |  |
| 4 | `value` | DOUBLE |  |  |  |
| 5 | `publisher` | VARCHAR |  |  |  |
| 6 | `created_at` | TIMESTAMP |  |  |  |

### haz_raw_reliefweb_docs

| # | column | type | notnull | default | pk |
|--:|--------|------|:-------:|---------|:--:|
| 0 | `record_id` | VARCHAR |  |  |  |
| 1 | `payload_json` | VARCHAR |  |  |  |

</details>

## 11. Database-Wide Statistics

- **Total tables:** 2
- **Total rows across all tables:** 3,001
- **Database file size:** 15.3 MB

