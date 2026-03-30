# CLAUDE.md

This file provides guidance for Claude Code when working with the Pythia codebase. CLAUDE.md must be kept as a comprehensive, up-to-date description of the Pythia system. **Update this file at the end of every session where Claude Code changes the repo** — include any new architectural decisions, environment variables, failure modes, workflow changes, or component behavior that future sessions need to know.

## Project overview

Pythia is an end-to-end AI forecasting system for humanitarian crises. It scans countries for emerging hazards, produces triage signals, generates research briefs, runs LLM ensembles for probabilistic forecasts (SPDs), scores and calibrates them, and serves outputs via a FastAPI API and Next.js dashboard.

## Post-Edit Documentation Requirements

After making any code changes, evaluate whether the following files need updating and update them if so:

- **CLAUDE.md** – Update if you've changed architecture, environment variables, failure modes, component behavior, workflow steps, or database schema. This is the authoritative reference for future Claude Code sessions.
- **README.md** – Update if you've changed setup steps, dependencies, usage instructions, file structure, or how to run the project.
- **docs/fred_overview.md** – This is a plain-English description of the system for non-technical readers. Update it if you've changed what the system does, how it behaves, its inputs/outputs, or its overall logic. Avoid technical jargon; explain changes in terms of what the system now does differently from a user perspective. Assume readers understand forecasting and humanitarian data, but not code.

Do not update these files for trivial changes (e.g. formatting, minor refactors with no behavioral change). Use your judgment.

**Core pipeline:**
```
Resolver facts/base rates -> Horizon Scanner per-hazard pipeline (RC grounding → RC → triage grounding → triage)
  -> Structured data connectors (ReliefWeb, ACAPS, IPC, ACLED political, ENSO, seasonal TC, HDX Signals, ACLED CAST, GDACS, GDACS event history, FEWS NET IPC)
  -> Adversarial checks (RC L1+) -> Forecaster SPD ensemble
  -> DuckDB -> FastAPI API -> Next.js Dashboard + CSV/Excel exports
```

**Post-forecast pipeline:**
```
compute_resolutions (ground truth from Resolver) -> resolutions table
  -> compute_scores (Brier, log loss, CRPS per horizon) -> scores table
  -> compute_calibration (weights per hazard/metric) -> calibration_weights
  -> generate_calibration_advice (per-hazard/metric prompt guidance) -> calibration_advice
```

## Repository layout

```
Pythia/
  horizon_scanner/     # Country/hazard triage via LLM (per-hazard RC + triage, tail packs, reliefweb)
    enso/              #   ENSO state/forecast scraper (IRI/CPC Quick Look)
    seasonal_tc/       #   Seasonal TC forecast scrapers (TSR, NOAA CPC, BoM) + country-basin mapping
    rc_prompts.py      #   Per-hazard RC prompt builders (ACE, DR, FL, HW, TC)
    hs_triage_prompts.py  # Per-hazard triage prompt builders
    rc_grounding_prompts.py  # RC-specific grounding queries
    hs_triage_grounding_prompts.py  # Triage-specific grounding queries
    hdx_signals.py     #   HDX Signals (OCHA automated crisis monitoring) connector
    conflict_forecasts.py  # Unified conflict forecast loader (VIEWS + CF.org + ACLED CAST)
  forecaster/          # LLM ensemble for SPD forecasts (structured data + SPD phases)
  resolver/            # Fact ingestion, resolution, and ground truth DB
    connectors/        #   Connector protocol + registry (ACLED, IDMC, IFRC Montandon, GDACS, FEWS NET IPC) + forecast registry (VIEWS, CF.org, ACLED CAST)
    ingestion/         #   Source-specific fetch/normalise clients (acled_client, idmc/)
    transform/         #   Adapters, normalisation, source resolution
    tools/             #   Pipeline orchestrator (run_pipeline.py), precedence, deltas, enrichment
    db/                #   DuckDB schema + helpers
  pythia/
    api/               # FastAPI service (50+ endpoints)
    db/                # DuckDB schema definitions + migrations (schema.py is authoritative)
    tools/             # Post-forecast compute scripts (resolutions, scores, calibration, calibration advice)
    web_research/      # Shared retriever backends (Gemini, OpenAI, Claude, Exa, Perplexity)
    prediction_markets/  # Prediction market signal retriever (Metaculus, Polymarket, Manifold)
    acaps.py           # ACAPS unified connector (INFORM, Risk Radar, Daily Monitoring, Access)
    acled_political.py # ACLED event-level political data connector
    adversarial_check.py # Counter-evidence checks for RC Level 1+ cases
    ipc_phases.py      # IPC food security phase connector (legacy, requires API key — currently unused)
    food_security.py   # Unified food security loader (FEWS NET primary, IPC API fallback)
    fewsnet_food_security.py  # Backward-compat shim → food_security.py
    market_snapshot.py # Manifold prediction market snapshot utility
    tests/             # Pythia-specific tests
  web/                 # Next.js 14 dashboard (TypeScript, Tailwind)
  tests/               # Cross-module integration tests
  docs/                # Component documentation
  scripts/             # Operational scripts
  .github/workflows/   # CI + scheduled pipelines
```

## Key files

- `pythia/db/schema.py` — Authoritative Pythia DuckDB schema (tables, migrations, bucket defs)
- `resolver/db/schema.sql` — Resolver DuckDB schema (facts_resolved, facts_deltas, snapshots)
- `pythia/buckets.py` — SPD bucket definitions (PA, FATALITIES, PHASE3PLUS_IN_NEED). `BUCKET_SPECS` maps metric names to `BucketSpec` tuples. DR_PHASE3_BUCKETS has 5 buckets for IPC Phase 3+ population (<100k, 100k-1M, 1M-5M, 5M-15M, >=15M).
- `pythia/config.yaml` — Central configuration (LLM profiles, hazards, bucket specs)
- `horizon_scanner/horizon_scanner.py` — HS main entrypoint (~1700 LOC)
- `horizon_scanner/regime_change.py` — RC scoring (score = likelihood x magnitude, 4 levels)
- `horizon_scanner/regime_change_llm.py` — Per-hazard RC LLM pipeline (2-pass: both Gemini Flash by default)
- `horizon_scanner/triage.py` — Per-hazard triage LLM pipeline (2-pass: Gemini Pro + Gemini Flash, ACLED low-activity filter, RC-promoted skip for L1+ hazards)
- `horizon_scanner/rc_prompts.py` — Per-hazard RC prompt builders (ACE, DR, FL, TC with calibration anchors; HW builder retained as dead code)
- `horizon_scanner/hs_triage_prompts.py` — Per-hazard triage prompt builders (scoring anchors, RC context injection)
- `horizon_scanner/rc_grounding_prompts.py` — RC-specific grounding queries (TRIGGER/DAMPENER/BASELINE signals)
- `horizon_scanner/hs_triage_grounding_prompts.py` — Triage-specific grounding queries (SITUATION/RESPONSE/FORECAST/VULNERABILITY)
- `horizon_scanner/enso/enso_module.py` — ENSO state/forecast scraper (IRI/CPC, 7-day cache)
- `horizon_scanner/seasonal_tc/seasonal_tc_runner.py` — Seasonal TC forecast orchestrator (TSR + NOAA CPC + BoM)
- `horizon_scanner/seasonal_tc/__init__.py` — Country-to-basin mapping + cached TC context reader
- `horizon_scanner/hdx_signals.py` — HDX Signals connector (OCHA automated crisis monitoring, indicator-to-hazard mapping)
- `horizon_scanner/conflict_forecasts.py` — Unified conflict forecast loader (VIEWS + conflictforecast.org + ACLED CAST)
- `forecaster/cli.py` — Forecaster main runner (~7300 LOC). Supports both SPD (5-bucket × 6-month) and binary (per-month probability) forecast pipelines. Binary questions (metric=EVENT_OCCURRENCE) are detected and routed to `_run_binary_forecast_for_question()`, which uses `build_binary_event_prompt()` for prompting and `parse_binary_response()` for output parsing. Binary storage convention: bucket_1 = P(yes), bucket_2 = P(no) = 1-P(yes), buckets 3-5 = 0. History builders: `_build_history_summary` dispatches to `_load_fewsnet_phase3_history` for DR/PHASE3PLUS_IN_NEED (null-aware, coverage-tracked), `_build_natural_hazard_seasonal_profile` for natural hazard PA, `_build_conflict_base_rate` for ACE, and `_build_gdacs_event_history` for FL/DR/TC event occurrence (binary 1/0 + alertlevel from facts_resolved). `_format_base_rate_for_prompt` handles `fewsnet_phase3` type with null-aware formatting showing "null" for FEWS NET analysis cycle gaps. GDACS event history is loaded in `_load_structured_data()` for FL/DR/TC hazards and injected into both SPD prompts (via `_format_gdacs_event_history_for_prompt` in prompts.py) and binary prompts (via `gdacs_event_history` parameter on `build_binary_event_prompt`).
- `pythia/api/app.py` — FastAPI application (~3500 LOC). Phase 4 additions: `_get_risk_index_binary()` serves EVENT_OCCURRENCE probabilities (bucket_1 = P(event)), `/v1/diagnostics/resolution_rates` endpoint returns resolution rate by (hazard, metric). Risk index supports EVENT_OCCURRENCE (binary), PHASE3PLUS_IN_NEED (SPD), PA, and FATALITIES metrics. KPI scopes accept all four metric scopes. **Resolver accordion endpoints**: `/v1/resolver/source_inventory` returns per-source metadata (label, category, last_updated, row counts); `/v1/resolver/source_data` returns rows for a specific source with curated or all columns. Source registry (`_SOURCE_REGISTRY`) maps 22 source keys to tables, filters, curated columns. `facts_resolved` sources filtered by `publisher` column (IFRC, IDMC, ACLED, GDACS / JRC, FEWS NET). Last-updated dates use ingestion-time columns (`created_at`, `fetched_at`) not data-period columns.
- `web/src/app/questions/[questionId]/BinaryPanel.tsx` — Binary event forecast display: per-month probability bars, resolution outcomes (event/no event), per-horizon Brier scores.
- `pythia/tools/compute_resolutions.py` — Resolves forecasts against Resolver ground truth. Supports PA, FATALITIES, EVENT_OCCURRENCE, and PHASE3PLUS_IN_NEED metrics. Source-aware null handling: FATALITIES+ACE/ACO and EVENT_OCCURRENCE default to 0 when no data; PA and PHASE3PLUS_IN_NEED leave horizons unresolved (null) when no data exists. FEWS NET IPC resolution via `_try_fewsnet_ipc()` queries only `phase3plus_in_need` metric (not projections), guarded to DR hazard only.
- `pythia/tools/compute_bucket_centroids.py` — Computes data-driven bucket centroids from historical facts_resolved. Metric-aware SQL filter: PHASE3PLUS_IN_NEED queries `phase3plus_in_need`, PA queries `affected/people_affected/pa/displaced`, FATALITIES queries `fatalities`.
- `pythia/tools/compute_scores.py` — Brier/log/CRPS scoring per horizon
- `pythia/tools/compute_calibration_pythia.py` — Calibration weights + LLM advice
- `resolver/connectors/protocol.py` — Connector protocol (21-column canonical schema contract)
- `resolver/connectors/__init__.py` — Connector registry (discover_connectors: ACLED, IDMC, IFRC Montandon, GDACS, FEWS NET IPC) + FORECAST_REGISTRY (VIEWS, CF.org, ACLED CAST)
- `resolver/connectors/fewsnet_ipc.py` — FEWS NET IPC Phase 3+ population connector (DR hazard; Current Situation + Most Likely scenarios)
- `resolver/connectors/acled_cast.py` — ACLED CAST connector (event-count forecasts by type: total/battles/ERV/VAC)
- `resolver/tools/run_pipeline.py` — Pipeline orchestrator (fetch -> validate -> enrich -> precedence -> deltas -> DuckDB)
- `resolver/tools/enrich.py` — Enrichment (registry lookups, ym derivation, defaults)
- `resolver/tools/precedence_config.yml` — Precedence tier policy
- `pythia/prediction_markets/retriever.py` — Prediction market signal retriever (currently disabled by default via `PYTHIA_PM_RETRIEVER_ENABLED=0`; Metaculus returns 403, Polymarket returns 422, each times out at 30s)
- `scripts/print_forecaster_ensemble.py` — Ensemble diagnostic script (must be invoked as `python -m scripts.print_forecaster_ensemble`, not directly)
- `pythia/acaps.py` — ACAPS unified connector (4 datasets: INFORM Severity, Risk Radar, Daily Monitoring, Humanitarian Access)
- `pythia/food_security.py` — Unified food security loader (FEWS NET primary, IPC API fallback); provides `load_food_security()`, `format_food_security_for_prompt()`, `format_food_security_for_spd()`. Routes: tries FEWS NET (publisher='FEWS NET') first, falls back to IPC (publisher='IPC'). Wired into RC, triage, and SPD prompts.
- `pythia/fewsnet_food_security.py` — Backward-compat shim that re-exports from `food_security.py`. Deprecated; existing callers still work.
- `pythia/ipc_phases.py` — Legacy IPC food security connector (requires IPC_API_KEY — dead code; replaced by `resolver/connectors/ipc_api.py`)
- `resolver/connectors/ipc_api.py` — IPC API connector for non-FEWS NET countries. Fetches Phase 3+ data from api.ipcinfo.org, excludes FEWS NET countries, writes to facts_resolved. Env: `IPC_API_KEY`, `IPC_API_MONTHS` (default 24), `IPC_API_REQUEST_DELAY` (default 1.0).
- `pythia/acled_political.py` — ACLED event-level political data (protests, riots, strategic developments); wired into bulk ingest via `_bulk_fetch_acled_political` in `ingest_structured_data.py`
- `pythia/adversarial_check.py` — Counter-evidence searches for RC Level 1+ (devil's advocate)
- `horizon_scanner/reliefweb.py` — ReliefWeb humanitarian reports connector
- `forecaster/hazard_prompts.py` — Hazard-specific reasoning guidance for SPD prompts. DR/PHASE3PLUS_IN_NEED dispatches to `_DR_PHASE3` (FEWS NET IPC-specific); DR/PA still uses `_DR` (IFRC Montandon). FEWS NET projection context injected into DR/PHASE3PLUS_IN_NEED prompts via `_load_fewsnet_projection()` in prompts.py.
- `forecaster/scoring.py` — Scoring utilities: `multiclass_brier()`, `log_score()`, `binary_brier()` (Brier score for binary forecasts: `(forecast_p - outcome)^2`)
- `forecaster/trace_validation.py` — Diagnostic validation of structured reasoning traces from SPD ensemble members. Checks prior consistency, delta arithmetic, and magnitude consistency. Produces `trace_quality_score` (0-1) per model. Never blocks forecasts.
- `scripts/ci/snapshot_prompt_artifact.py` — Prompt version snapshot script
- `scripts/refresh_crisiswatch.py` — Playwright-based CrisisWatch scraper (monthly, writes `crisiswatch_latest.json`)
- `pythia/tools/generate_calibration_advice.py` — Per-hazard/metric calibration advice generation
- `tools/compare_gdacs_ifrc.py` — GDACS vs IFRC Montandon PA comparison diagnostic (side-by-side coverage, ratios, blind spots)
- `scripts/create_questions_from_triage.py` — Creates forecast questions from HS triage output. Supported hazards: ACE (FATALITIES+PA), CU (PA), DR (PHASE3PLUS_IN_NEED+EVENT_OCCURRENCE), FL (PA+EVENT_OCCURRENCE), TC (PA+EVENT_OCCURRENCE), DI (PA). HW excluded (no resolution source). DR/PA for FEWS NET countries is remapped to metric=PHASE3PLUS_IN_NEED with IPC Phase 3+ wording (via `_build_dr_fewsnet_question_wording`). DR/PA blocked for non-FEWS NET countries (via `resolver/data/fewsnet_countries.json`; fail-open if file missing). EVENT_OCCURRENCE questions generated for ALL countries (GDACS global coverage). Binary question wording references GDACS Orange/Red alerts. Epoch-specific question_ids with `_{epoch_label}` suffix.
- `forecaster/binary_prompts.py` — Binary event prompt builder for EVENT_OCCURRENCE questions. Contains: `build_binary_event_prompt()` (6-section prompt: role/task, base rate, current situation, GDACS event history, hazard reasoning, output instructions; accepts optional `gdacs_event_history` dict), `build_binary_base_rate()` (queries facts_resolved for seasonal event rates), `get_binary_hazard_reasoning_block()` (DR/FL/TC-specific reasoning), `parse_binary_response()` (JSON parser with code fence stripping, probability clamping to [0.01, 0.99]).
- `scripts/db/update_bucket_centroids.py` — Seeds bucket definitions and centroids into DuckDB from `pythia/buckets.py` BUCKET_SPECS. Iterates all metrics (PA, FATALITIES, PHASE3PLUS_IN_NEED).
- `tools/analyze_fewsnet_distribution.py` — Diagnostic script to analyze FEWS NET Phase 3+ data distribution in facts_resolved. Computes distribution stats, histogram by DR_PHASE3_BUCKETS, country-level summary, and bucket balance assessment.
- `resolver/data/fewsnet_countries.json` — List of 48 FEWS NET-monitored country ISO3 codes. Used by question generator to restrict DR/PA questions to FEWS NET countries. Written by the FEWS NET IPC connector during backfill.

## Databases

Two DuckDB databases:

**Pythia DB** (`PYTHIA_DB_URL`): system of record
- `hs_runs`, `hs_triage` — Horizon Scanner outputs (triage scores, RC fields)
- `hs_hazard_tail_packs` — RC-triggered hazard evidence packs
- `hs_adversarial_checks` — Counter-evidence for RC Level 1+ (devil's advocate)
- `seasonal_forecasts` — NMME country-level temp/precip anomalies (monthly from CPC)
- `conflict_forecasts` — VIEWS + conflictforecast.org + ACLED CAST conflict predictions (PK: source, iso3, hazard_code, metric, lead_months, forecast_issue_date)
- `questions`, `question_research` — Seeded questions + research briefs
- `forecasts_raw`, `forecasts_ensemble` — Per-model + aggregated SPDs. Both include `reasoning_trace_json TEXT` column storing structured reasoning traces (prior, updates with deltas, point estimate, RC assessment) as JSON. NULL for binary forecasts, Track 2, or models that don't emit traces.
- `resolutions` — Ground truth values per (question_id, horizon_m). Not all 6 horizons may have rows — source-aware null handling skips unresolvable horizons.
- `scores` — Brier/log/CRPS per (question, horizon, model)
- `calibration_weights`, `calibration_advice` — Per hazard/metric weights + LLM advice
- `bucket_centroids` — SPD bucket centroids per (hazard_code, metric, bucket_index); seeded from `pythia/buckets.py`, updated via EMA from resolution data
- `bucket_definitions` — SPD bucket boundary definitions per (metric, bucket_index)
- `reliefweb_reports` — ReliefWeb humanitarian situation reports
- `acled_political_events` — ACLED event-level political data
- `ipc_phases` — IPC food security phase populations
- `acaps_inform_severity`, `acaps_inform_severity_trend` — ACAPS INFORM severity scores
- `acaps_risk_radar` — ACAPS forward-looking risk assessments
- `acaps_daily_monitoring` — ACAPS analyst-curated daily updates
- `acaps_humanitarian_access` — ACAPS humanitarian access scores
- `crisiswatch_entries` — ICG CrisisWatch monthly arrows + alerts (PK: iso3, year, month)
- `hdx_signals` — HDX Signals (OCHA automated crisis monitoring) persisted from CSV (PK: iso3, indicator, signal_date)
- `enso_state` — ENSO state/forecast snapshots from IRI/CPC (PK: fetch_date)
- `seasonal_tc_outlooks` — Seasonal TC forecasts from TSR/NOAA CPC/BoM (PK: basin, source, fetched_at)
- `seasonal_tc_context_cache` — Pre-formatted per-country TC context text (PK: iso3)
- `llm_calls` — Full telemetry (cost, tokens, latency, errors)

**Resolver DB** (`resolver/db/schema.sql`): fact ingestion
- `facts_resolved` — Precedence-resolved facts (unique on ym, iso3, hazard_code, metric, series_semantics). Includes `alertlevel` column (GDACS-specific, NULL for other sources).
- `facts_deltas` — Monthly flow changes
- `snapshots`, `manifests`, `meta_runs` — Pipeline metadata

## Resolver architecture

The Resolver was refactored in PR #610 to a connector-based architecture. Defunct legacy connectors (DTM, EM-DAT ingestion, HDX, IPC, ODP, ReliefWeb, UNHCR, WFP, WHO, WorldPop) were removed. The GDACS connector was re-implemented as a new Connector protocol source.

**Connector protocol** (`resolver/connectors/protocol.py`): Every data source implements a `Connector` protocol with a `name` attribute and a `fetch_and_normalize()` method that returns a DataFrame with exactly 21 canonical columns (event_id, iso3, hazard_code, metric, value, as_of_date, publisher, etc.). Connectors may include supplementary columns beyond the canonical set (e.g. `alertlevel` for GDACS) by passing `extra_columns` to `validate_canonical`; `run_pipeline` auto-detects and passes these through.

**Active connectors** (`resolver/connectors/__init__.py` REGISTRY):
- `acled` — ACLED conflict/fatalities data (wraps `resolver/ingestion/acled_client`)
- `idmc` — IDMC internal displacement data (wraps `resolver/ingestion/idmc/`)
- `ifrc_montandon` — IFRC Go connector (stubbed, not yet active)
- `gdacs` — GDACS disaster population exposure (FL, DR, TC). No auth required. **Two data sources**: (1) static RSS feeds (`xml/rss_fl_3m.xml`, `xml/rss_tc_3m.xml`) for FL/TC in ≤3-month window (fast, has population data); (2) JSON search API (`gdacsapi/api/events/geteventlist/SEARCH`) + per-event RSS enrichment for >3-month backfill and **always for DR** (DR RSS feed returns 404). The original `rss.aspx?profile=ARCHIVE` endpoint is broken. Depth controlled by `GDACS_MONTHS` (default 3; use 135 for full backfill to 2015). Multi-country events use population-weighted allocation. TC zero-fills no-event months; FL/DR do not. **Two metrics per country-month-hazard**: `in_need` (population exposed, existing) and `event_occurrence` (binary 1/0 based on alertlevel: Orange/Red=1, Green=0). The `alertlevel` column (Green/Orange/Red) is stored as a supplementary column in `facts_resolved` (NULL for non-GDACS sources). Entry point: `resolver/ingestion/gdacs.py` (for `run_connectors.py`); also integrated into `pythia/tools/ingest_structured_data.py`. Env vars: `GDACS_MONTHS` (default 3), `GDACS_REQUEST_DELAY` (default 1.0s), `GDACS_FORCE_RSS`/`GDACS_FORCE_JSON` (override auto-detection).
- `fewsnet_ipc` — FEWS NET IPC Phase 3+ population estimates (DR hazard). No auth required. Fetches from `https://fdw.fews.net/api/ipcpopulationsize.csv`. Two metrics: `phase3plus_in_need` (Current Situation, used for resolution) and `phase3plus_projection` (Most Likely, context for prompts). ISO2→ISO3 conversion via pycountry. Deduplicates by latest `reporting_date` per (iso3, scenario, month). Writes discovered country list to `resolver/data/fewsnet_countries.json` for question generator consumption. Entry point: `resolver/ingestion/fewsnet_ipc.py`; also integrated into `pythia/tools/ingest_structured_data.py`. Env vars: `FEWSNET_MONTHS` (default 12; 120 for backfill to 2016), `FEWSNET_REQUEST_DELAY` (default 1.0s).
- `ipc_api` — IPC API Phase 3+ population estimates for non-FEWS NET countries (DR hazard). Requires `IPC_API_KEY`. Fetches from `https://api.ipcinfo.org/population`. Same two metrics as fewsnet_ipc. Excludes countries in `resolver/data/fewsnet_countries.json` (FEWS NET always takes priority). Writes `resolver/data/ipc_countries.json` for question generator. Entry point: `resolver/ingestion/ipc_api.py`; also integrated into `pythia/tools/ingest_structured_data.py`. Env vars: `IPC_API_KEY` (required), `IPC_API_MONTHS` (default 24), `IPC_API_REQUEST_DELAY` (default 1.0s).

**Pipeline orchestrator** (`resolver/tools/run_pipeline.py`):
```
discover_connectors() -> fetch_and_normalize() per connector
  -> validate_canonical(extra_columns auto-detected) -> enrich() + derive_ym()
  -> precedence_engine (tiered source resolution, publisher used as source) -> make_deltas()
  -> write to DuckDB (facts_resolved + facts_deltas, supplementary columns preserved)
```

**Precedence policy** (`resolver/tools/precedence_config.yml`):
- Tier 0: IFRC Montandon/ACLED (highest priority; IFRC Montandon is the active source for natural hazard PA: FL, DR, TC)
- Tier 1: IDMC
- Tier 2: EM-DAT (historical read-only, no active connector; replaced by IFRC Montandon)

**Transform adapters** (`resolver/transform/adapters/`): ACLED and IDMC adapters normalize source-specific schemas to the common format used by the precedence engine.

**NMME seasonal forecasts** (`resolver/ingestion/nmme.py` + `resolver/tools/ingest_nmme.py`):
Separate from the Connector pipeline. Fetches NMME ensemble mean anomalies from CPC FTP, computes area-weighted country averages using xarray + regionmask (`countries_10` resolution for 170+ countries), and writes to the `seasonal_forecasts` table in Pythia DB. Injected into HS triage/RC prompts via `horizon_scanner/seasonal_context.py` (`climate_data` kwarg) and into forecaster prompts via `research_json["nmme_seasonal_outlook"]`. **Now integrated into `pythia/tools/ingest_structured_data.py`** as the `nmme` source (NMME failures are caught as warnings, non-fatal, since FTP files are published ~9th-10th of each month). The standalone `python -m resolver.tools.ingest_nmme` entry point still works independently. The standalone `ingest-nmme.yml` workflow is deprecated (schedule disabled, manual dispatch retained). Uses `decode_times=False` in `xr.open_dataset()` calls because the new CPC multi-lead files use `months since 1960-01-02 21:00:00` as time units, which xarray cannot decode.

**Conflict forecast connectors** (`resolver/connectors/views.py`, `resolver/connectors/conflictforecast.py`, `resolver/connectors/acled_cast.py`):
Separate from the Connector pipeline (use `FORECAST_REGISTRY`, not `REGISTRY`). VIEWS connector fetches ML-based fatality predictions from the VIEWS API (`views_predicted_fatalities`, `views_p_gte25_brd`, leads 1–6). conflictforecast.org connector fetches news-based risk scores from Backendless API (`cf_armed_conflict_risk_3m`, `cf_armed_conflict_risk_12m`, `cf_violence_intensity_3m`). ACLED CAST connector fetches event-count forecasts via OAuth2 API (`cast_total_events`, `cast_battles_events`, `cast_erv_events`, `cast_vac_events`, 6-month lead), aggregated from admin1 to country level. All three write to `conflict_forecasts` table. `fetch_and_store` deduplicates before writing (keeps only the latest `forecast_issue_date` per source), and `_write_to_db` prunes old vintages (keeps only the 2 most recent issue dates per source). Loaded into ACE prompts via `horizon_scanner/conflict_forecasts.py`. The conflictforecast.org connector uses suffix-priority column selection: when multiple columns match a metric pattern (e.g. `ons_armedconf_03`), it prefers `_all` (combined forecast) over `_text`/`_hist` sub-models, and explicitly skips `_target` (ground truth, NaN for future) and `_naive` (baseline). Also includes a Backendless metadata skip set and a median-value sanity check (onset metrics only; intensity metrics use log-scale values). **Now integrated into `pythia/tools/ingest_structured_data.py`** as sources `views`, `conflictforecast`, `acledcast` (with `conflict` as a convenience alias for all three). The standalone `python -m resolver.tools.fetch_conflict_forecasts` entry point still works independently.

**ENSO state and forecast** (`horizon_scanner/enso/enso_module.py`):
Scrapes IRI/CPC ENSO Quick Look page for current ENSO state, Niño 3.4 anomaly, 9-season probabilistic forecast, multi-model plume averages, and IOD state. Cached as JSON with 7-day expiry. Injected into RC and triage prompts for DR, FL, HW, TC hazards via `get_enso_prompt_context()`. Refreshed by `.github/workflows/refresh-enso.yml`. DB-first pattern: `fetch_and_store_enso()` persists to `enso_state` table during backfill; `get_enso_prompt_context()` reads from DB first, falls back to live scrape.

**Seasonal TC forecasts** (`horizon_scanner/seasonal_tc/`):
Aggregates basin-level seasonal TC forecasts from TSR (PDF extraction), NOAA CPC (press release scraping), and BoM (outlook scraping) across 8 basins. Country-to-basin mapping in `__init__.py`. Cached as JSON; `get_seasonal_tc_context_for_country(iso3)` returns prompt-ready text. Refreshed by `.github/workflows/refresh-seasonal-tc.yml`. DB-first pattern: `fetch_and_store_seasonal_tc()` persists to `seasonal_tc_outlooks` + `seasonal_tc_context_cache` tables during backfill; `get_seasonal_tc_context_for_country()` reads from DB first, falls back to cached JSON. The HS workflow no longer downloads the TC artifact separately — TC data is in the DB artifact.

**HDX Signals** (`horizon_scanner/hdx_signals.py`):
Downloads OCHA's HDX Signals CSV from CKAN API. Indicator-to-hazard mapping (acled_conflict→ACE, ipc_food_insecurity→DR, etc.). Filtered by country, hazard, recency (180 days). Injected into RC and triage prompts for all hazards via `format_hdx_signals_for_prompt()`. DB-first pattern: `bulk_fetch_and_store_hdx_signals()` persists signals to `hdx_signals` table during backfill; `format_hdx_signals_for_prompt()` reads from DB first, falls back to live CSV download.

**ICG CrisisWatch** (`horizon_scanner/crisiswatch.py` + `scripts/refresh_crisiswatch.py`):
Monthly fetch of ICG CrisisWatch data. **Primary source**: local JSON file (`horizon_scanner/data/crisiswatch_latest.json`) produced by the Playwright scraper (`scripts/refresh_crisiswatch.py`), which runs monthly via `refresh-crisiswatch.yml` workflow, loading `https://www.crisisgroup.org/crisiswatch` in headless Chromium, parsing with BeautifulSoup, and committing structured JSON. The `/crisiswatch/print` endpoint is broken (stale Oct 2019 data); the main page renders correctly. **Fallback**: Gemini grounding calls (`_call_gemini_grounding` in `crisiswatch.py`) — only attempted when the JSON file is missing or empty. Called once per HS run, cached in-memory, persisted to `crisiswatch_entries` DuckDB table. Injected into ACE RC prompts (via `crisiswatch_context` + deprecated `icg_on_the_horizon`), ACE triage prompts, and ACE SPD prompts. Country name → ISO3 mapping via hardcoded `_ICG_COUNTRY_ISO3` dict. JSON staleness: if `fetched_at` is >45 days old, a warning is logged but data is still used.

Run the pipeline: `python -m resolver.tools.run_pipeline [--connectors acled idmc] [--db path/to/resolver.duckdb]`

## Per-horizon architecture

Forecasts cover a 6-month window. Each question has `window_start_date` and `target_month` (= month 6).

- **Resolutions**: `compute_resolutions` resolves each horizon independently against Resolver's `facts_resolved.created_at` for ordering. Source-aware null handling: FATALITIES+ACE/ACO defaults to 0 (ACLED continuous coverage), EVENT_OCCURRENCE defaults to 0 (GDACS binary), PA and PHASE3PLUS_IN_NEED leave horizons unresolved when no data exists. Unresolvable hazards: DI, HW.
- **Scoring**: `compute_scores` scores each (question, horizon_m) pair separately. SPD questions (PA, FATALITIES) get multiclass Brier, log loss, and CRPS. Binary questions (EVENT_OCCURRENCE) get single Brier score: `(forecast_p - outcome)^2` where forecast_p = bucket_1 probability from ensemble. PHASE3PLUS_IN_NEED uses SPD scoring. Missing resolution horizons are naturally excluded via JOIN.
- **Calibration**: `compute_calibration_pythia` aggregates scores across horizons to produce per-model weights. Groups by (hazard_code, metric), so EVENT_OCCURRENCE questions form their own calibration pool separate from PA/FATALITIES.

## Phase 4: Dashboard + Calibration Integration

**API endpoints (Phase 4):**
- `/v1/risk_index?metric=EVENT_OCCURRENCE` — Returns binary P(event) as the risk value (0-1). No centroid multiplication. Per-capita returns raw probability unchanged (already a rate).
- `/v1/risk_index?metric=PHASE3PLUS_IN_NEED` — SPD-based risk index for IPC Phase 3+ population. Supports `normalize=true` for per-capita (fraction of population in Phase 3+).
- `/v1/diagnostics/resolution_rates` — Returns resolution rates by (hazard_code, metric): total/resolved/skipped questions with rate. Supports `hazard_code` filter.
- `/v1/diagnostics/kpi_scopes?metric_scope=EVENT_OCCURRENCE` — KPI scopes now accept EVENT_OCCURRENCE and PHASE3PLUS_IN_NEED alongside PA and FATALITIES.
- `/v1/performance/scores?metric=EVENT_OCCURRENCE` — Performance scores work for all metrics; binary questions produce Brier score only (no log/CRPS).

**Frontend views (Phase 4):**
- `RiskView` type expanded: `PA_EIV`, `PA_PC`, `FATALITIES_EIV`, `FATALITIES_PC`, `EVENT_OCCURRENCE`, `PHASE3PLUS_EIV`, `PHASE3PLUS_PC` (7 total).
- RiskIndexMap uses fixed probability thresholds (0-5%, 5-15%, 15-30%, 30-50%, 50-100%) for EVENT_OCCURRENCE instead of Jenks breaks.
- RiskIndexTable displays probabilities as percentages for binary metrics.
- BinaryPanel.tsx: Per-month probability bars with resolution outcomes and Brier scores for question detail page.
- PerformancePanel: Resolution Coverage Summary section with color-coded rates (green >90%, yellow 50-90%, red <50%). Metric filter includes EVENT_OCCURRENCE and PHASE3PLUS_IN_NEED.
- QuestionsTable: Binary questions display as "Binary (event)", PHASE3PLUS_IN_NEED as "Phase 3+".
- **Resolver page**: Redesigned as a source-level accordion data explorer grouped by category (Resolution Data, Conflict Forecasts, Weather and Climate, Situation Reports, Other Alerts, Other). Each data source gets a collapsible row with summary metadata (name, last updated, global/country row counts) and a lazy-loaded data table on expand. Shared country selector at top filters all country-filterable sources. Old tab-based layout (`tabs/` directory) removed. Component: `web/src/app/resolver/ResolverClient.tsx`.

## Regime Change (RC) scoring

RC detects departures from historical base rates (distinct from triage_score which measures overall risk).

- `score = likelihood x magnitude`, clamped [0, 1]
- **Levels** (env-overridable, likelihood-only thresholds): L0 (likelihood < 0.15), L1 (likelihood >= 0.15), L2 (likelihood >= 0.35), L3 (likelihood >= 0.55). Env vars: `PYTHIA_HS_RC_LEVEL1_LIKELIHOOD`, `PYTHIA_HS_RC_LEVEL2_LIKELIHOOD`, `PYTHIA_HS_RC_LEVEL3_LIKELIHOOD`.
- **Track assignment**: RC level > 0 → Track 1 (full ensemble), RC level 0 + priority tier → Track 2 (single Gemini Flash model), otherwise no SPD.
- **Triage skip for RC L1+**: Hazards with RC level ≥ 1 are automatically promoted to Track 1, so triage LLM calls are skipped (saving 3 LLM calls per hazard: 1 grounding + 2 triage passes). These hazards receive synthetic `_RC_PROMOTED_DEFAULTS` (triage_score=0, status="rc_promoted"). Downstream `_write_hs_triage()` independently assigns track=1 from RC level, so triage output has no effect. Three triage skip patterns exist: ACE low-activity, seasonal skip, and RC-promoted.
- L1+ triggers hazard tail pack generation and adversarial evidence checks. L2+ additionally forces `need_full_spd = TRUE`.
- **CRITICAL sync constraint**: The RC level threshold in `_select_tail_pack_hazards` (horizon_scanner.py) and the re-check threshold inside `adversarial_check.py` must match. If they drift, candidates will be passed to adversarial checks but silently rejected.
- Tail packs are disabled by default (`HS_TAIL_PACKS_ENABLED` defaults to `"0"`). Enable via `PYTHIA_HS_HAZARD_TAIL_PACKS_ENABLED=1`.
- Distribution check warns when too many assessments exceed expected proportions
- RC prompt templates include a softened distribution anchor paragraph for small-country runs (where few comparative countries are included)
- **RC model**: Both passes default to `gemini-3-flash-preview` (overridable via `PYTHIA_RC_MODEL_PASS1` / `PYTHIA_RC_MODEL_PASS2`)
- **RC data sources for ACE**: Conflict forecasts (VIEWS, conflictforecast.org, ACLED CAST) and ICG CrisisWatch "On the Horizon" flags are injected into ACE RC prompts. CrisisWatch is fetched once per run in `main()` and explicitly threaded through `_run_hs_for_country` → `run_rc_for_country` → `_run_rc_for_single_hazard` (falls back to in-memory cache if not passed).
- **RC grounding queries**: Use hazard-specific search labels (e.g. "armed conflict escalation signals" for ACE, "flood risk river levels" for FL) instead of generic terms, to improve Gemini Google Search result relevance
- See `docs/hs_regime_change.md` for full details

## Track 1 / Track 2 forecast system

The forecaster routes questions into two tracks based on RC level:

- **Track 1** (full ensemble): Multi-model ensemble producing both `ensemble_bayesmc_v2` and `ensemble_mean_v2` aggregation rows. Used for RC level > 0 (higher-RC or higher-complexity questions). Higher cost.
- **Track 2** (single model): Single `track2_flash` model. Used for RC level 0 questions with priority tier. No ensemble aggregation — produces a single forecast row per question. Lower cost. Reasoning trace is simplified (prior + rc_assessment only; updates array may be empty).

Track 1 questions always receive scenarios, even when their triage tier is "quiet" (which occurs for RC-promoted hazards whose `_RC_PROMOTED_DEFAULTS` set `triage_score=0`).

Track 2 questions have no `ensemble_bayesmc_v2` or `ensemble_mean_v2` rows in `forecasts_ensemble`. The legacy CI step "Verify Forecaster wrote both aggregation methods" (`verify_forecaster_aggregations.py`) was removed because it hard-exited when all questions routed to Track 2. This is a resolved issue — the step should not be re-added.

## Testing

```bash
# Run all pythia tests (skip fastapi-dependent tests if not installed)
python3 -m pytest pythia/tests/ --ignore=pythia/tests/test_api_question_bundle.py -v

# Run cross-module tests (skip fastapi/openai-dependent tests)
python3 -m pytest tests/ --ignore-glob='tests/test_api_*' --ignore=tests/test_web_research.py -v

# Run RC calibration tests
python3 -m pytest resolver/tests/test_rc_calibration.py -v

# Run resolver tests
python3 -m pytest resolver/tests/ -v
```

**pytest config** (`pytest.ini`): 120s timeout, thread-based, log_cli enabled.
**Markers**: `@pytest.mark.db`, `@pytest.mark.allow_network`, `@pytest.mark.slow`, `@pytest.mark.nightly`.

Some test files require `fastapi` or `openai` which may not be installed locally — exclude them with `--ignore` as shown above.

## Common environment variables

| Variable | Purpose |
|----------|---------|
| `PYTHIA_DB_URL` | DuckDB connection URL |
| `PYTHIA_LLM_PROFILE` | LLM profile: `prod` |
| `PYTHIA_TEST_MODE` | Pipeline test mode flag (1/true/yes = stamp all output as test data) |
| `PYTHIA_LLM_CONCURRENCY` | Max concurrent LLM calls |
| `PYTHIA_API_TOKEN` | Admin API auth token |
| `PYTHIA_WEB_RESEARCH_ENABLED` | Enable shared retriever (0/1) |
| `HS_MAX_WORKERS` | HS concurrent country workers |
| `FORECASTER_RESEARCH_MAX_WORKERS` | Research phase concurrency |
| `FORECASTER_SPD_MAX_WORKERS` | SPD phase concurrency |
| `PYTHIA_HS_RC_LEVEL*_*` | RC threshold overrides |
| `PYTHIA_HS_RC_DIST_WARN_*` | RC distribution warning thresholds |
| `PYTHIA_HS_HAZARD_TAIL_PACKS_ENABLED` | Enable hazard tail packs (0/1, default 0) |
| `PYTHIA_ADVERSARIAL_CHECK_ENABLED` | Enable adversarial checks for RC L1+ (0/1, default 1) |
| `PYTHIA_GROUNDING_PRIMARY_BACKEND` | Primary grounding backend: `brave` (default), `openai`, or `gemini` |
| `BRAVE_SEARCH_API_KEY` | Brave Search API subscription token (required for Brave grounding) |
| `PYTHIA_BRAVE_MAX_RPS` | Max Brave Search API requests per second (default: 15, paid limit is 20) |
| `PYTHIA_HS_RESEARCH_WEB_SEARCH_ENABLED` | Enable web search for RC evidence packs (0/1, default 1) |
| `PYTHIA_PM_RETRIEVER_ENABLED` | Enable prediction market retriever (0/1, default 0 — see known failure modes) |
| `PYTHIA_PREDICTION_MARKETS_ENABLED` | Legacy alias; prefer `PYTHIA_PM_RETRIEVER_ENABLED` |
| `PYTHIA_GOOGLE_SPD_TIMEOUT_FLASH_SEC` | Gemini Flash SPD timeout (default 300s) |
| `PYTHIA_GOOGLE_SPD_TIMEOUT_PRO_SEC` | Gemini Pro SPD timeout (default 300s) |
| `ANTHROPIC_MAX_OUTPUT_TOKENS` | Anthropic max output tokens for SPD calls (default 16384) |
| `PYTHIA_ANTHROPIC_SPD_MAX_TOKENS` | Anthropic SPD-specific max tokens override (default: ANTHROPIC_MAX_OUTPUT_TOKENS) |
| `ACAPS_EMAIL` / `ACAPS_PASSWORD` | ACAPS API credentials |
| `GDACS_MONTHS` | Number of months of GDACS history to fetch (default 3; 135 for full backfill to 2015) |
| `GDACS_REQUEST_DELAY` | Seconds between GDACS API requests (default 1.0) |
| `GDACS_FORCE_RSS` | Force static RSS feed path ("1" to enable; overrides auto-detection) |
| `GDACS_FORCE_JSON` | Force JSON search API path ("1" to enable; overrides auto-detection) |
| `FEWSNET_MONTHS` | Number of months of FEWS NET IPC history to fetch (default 12; 120 for backfill to 2016) |
| `FEWSNET_REQUEST_DELAY` | Seconds between FEWS NET API retries (default 1.0) |
| `IPC_API_KEY` | API key for api.ipcinfo.org (required for IPC connector) |
| `IPC_API_MONTHS` | Number of months of IPC API history to fetch (default 24; IPC analyses are infrequent) |
| `IPC_API_REQUEST_DELAY` | Seconds between IPC API retries (default 1.0) |

| `PYTHIA_HS_ONLY_COUNTRIES` | Restrict HS to specific countries (comma-separated ISO3s) |
| `PYTHIA_HS_FALLBACK_MODEL_SPECS` | Fallback model for HS triage (default from profile `hs_fallback`) |
| `PYTHIA_HS_LLM_MAX_ATTEMPTS` | Max retry attempts for HS triage LLM calls (default 3) |
| `PYTHIA_HS_GEMINI_TIMEOUT_SEC` | Gemini timeout for HS triage (default 120s) |
| `PYTHIA_DUCKDB_MEMORY_LIMIT` | DuckDB memory limit for API (default "150MB") |
| `PYTHIA_DUCKDB_THREADS` | DuckDB thread count for API (default "2") |
| `PYTHIA_MAX_CONCURRENT_HEAVY` | Max concurrent heavy API queries (default "2") |
| `PYTHIA_CORS_ALLOW_ORIGINS` | CORS allowed origins for API (default "*") |
| `PYTHIA_SYNC_FROM_ARTIFACTS` | Enable DB sync from GitHub artifacts on API startup |
| `GPT5_CALL_TIMEOUT_SEC` | GPT-5.x SPD call timeout (default 300s) |
| `GEMINI_CALL_TIMEOUT_SEC` | Gemini SPD call timeout (default 300s) |
| `GROK_CALL_TIMEOUT_SEC` | Grok (XAI) SPD call timeout (default 300s) |
| `PYTHIA_SPD_ENSEMBLE_SPECS` | Override SPD ensemble at runtime (comma-separated `provider:model_id`) |
| `PYTHIA_BLOCK_PROVIDERS` | Comma-separated provider names to exclude from ensemble |
| `PYTHIA_RC_MODEL_PASS1` / `PYTHIA_RC_MODEL_PASS2` | Override RC LLM model per pass |
| `PYTHIA_CREDIT_RETRY_PAUSE_{PROVIDER}` | Credit-retry pause in seconds per provider (default: OpenAI=900, Anthropic=300, Google=600) |
| `PYTHIA_CREDIT_RETRY_MAX_{PROVIDER}` | Credit-retry max attempts per provider (default: 3 for all) |

Provider API keys: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `KIMI_API_KEY`, `DEEPSEEK_API_KEY`.

## Test mode

Test mode runs the identical pipeline as production. The only difference is that every row written during a test run carries `is_test = TRUE` in a dedicated boolean column. The website hides test data by default; users can opt in via the "Test ON/OFF" toggle in the navigation bar.

**Architecture:**
- `pythia/test_mode.py` — single source of truth; `is_test_mode()` reads `PYTHIA_TEST_MODE` env var
- `is_test BOOLEAN DEFAULT FALSE` column on all pipeline tables (hs_runs, hs_scenarios, hs_triage, hs_country_reports, hs_hazard_tail_packs, hs_adversarial_checks, questions, forecasts_ensemble, forecasts_raw, llm_calls, question_research, question_run_metrics, scenarios, question_context, run_provenance, resolutions, scores, eiv_scores)
- API endpoints accept `?include_test=true` query parameter (default: hidden)
- Calibration (`compute_calibration_pythia`, `generate_calibration_advice`) excludes test data automatically
- Downloads include an `is_test` column
- `scripts/mark_test_runs.py` retroactively marks existing runs as test data
- `.github/workflows/mark_test_runs.yml` workflow wraps the script for CI use

**Triggering test mode:**
- Set `PYTHIA_TEST_MODE=1` in environment (or `true`/`yes`)
- `run_horizon_scanner.yml` has a `test_mode` checkbox input
- `manual_test.yaml` always sets `PYTHIA_TEST_MODE=1`

## Build and run

```bash
# Install dependencies
pip install -r python_library_requirements.txt

# Initialize DB schema
python3 -c "from pythia.db.schema import ensure_schema; ensure_schema()"

# Run Resolver pipeline (fetch facts from ACLED + IDMC -> DuckDB)
python3 -m resolver.tools.run_pipeline
# Or specific connectors: python3 -m resolver.tools.run_pipeline --connectors acled idmc

# Ingest ALL structured data (conflict forecasts, ACAPS, IPC, ReliefWeb, NMME)
python3 -m pythia.tools.ingest_structured_data
# Filter by source: python3 -m pythia.tools.ingest_structured_data --sources conflict acaps ipc
# Valid sources: views, conflictforecast, acledcast, acaps_inform_severity,
#   acaps_risk_radar, acaps_daily_monitoring, acaps_humanitarian_access, ipc, reliefweb, acled_political, nmme, gdacs, fewsnet_ipc, ipc_api
# Aliases: acaps (all 4 ACAPS sources), conflict (views+conflictforecast+acledcast)

# Standalone entry points (still work independently):
python3 -m resolver.tools.ingest_nmme                        # NMME only
python3 -m resolver.tools.fetch_conflict_forecasts            # conflict forecasts only
python3 -m resolver.tools.fetch_conflict_forecasts --sources views conflictforecast_org acled_cast

# Refresh CrisisWatch data (requires Playwright: pip install playwright && playwright install chromium)
python3 -m scripts.refresh_crisiswatch [--output PATH] [--verbose]

# Run Horizon Scanner
python3 -m horizon_scanner.horizon_scanner

# Run Forecaster
python3 -m forecaster.cli --mode pythia

# Post-forecast pipeline
python3 -m pythia.tools.compute_resolutions
python3 -m pythia.tools.compute_scores
python3 -m pythia.tools.compute_calibration_pythia
python3 -m pythia.tools.generate_calibration_advice

# API server
uvicorn pythia.api.app:app --reload --port 8000

# Dashboard
cd web && npm install && npm run dev
```

## Prompt editing

Before editing any prompt source files, always run `bash scripts/snapshot_prompts.sh` first. This archives the current prompts before changes so the About page can show historical versions. Commit the snapshot alongside the prompt edits.

Key prompt files:
- `forecaster/prompts.py` — Forecaster SPD prompt builder
- `horizon_scanner/rc_prompts.py` — Per-hazard RC prompt builders (ACE, DR, FL, HW, TC)
- `horizon_scanner/hs_triage_prompts.py` — Per-hazard triage prompt builders
- `horizon_scanner/rc_grounding_prompts.py` — RC-specific grounding queries (per-hazard)
- `horizon_scanner/hs_triage_grounding_prompts.py` — Triage-specific grounding queries (per-hazard)
- `pythia/web_research/backends/brave_search.py` — Brave Search API grounding backend (primary, deterministic, $5/1K queries). Wired to Brave circuit breaker: checks `is_tripped()` at entry, records success/failure per query.
- `pythia/web_research/brave_circuit_breaker.py` — Thread-safe circuit breaker for Brave Search API. Trips after 3 consecutive failures; short-circuits all subsequent Brave calls when tripped. Module-level singleton, reset per HS run.
- `pythia/web_research/backends/gemini_grounding.py` — Gemini grounding backend (second fallback)

## Overview editing

Before editing `docs/fred_overview.md`, always run `bash scripts/snapshot_overview.sh` first. This archives the current overview before changes so the About page can show historical versions. Commit the snapshot alongside the overview edits.

## Code conventions

- Copyright header on every file: `# Pythia / Copyright (c) 2025 Kevin Wyjad`
- License: Pythia Non-Commercial Public License v1.0
- DuckDB is the only database backend (no Postgres/SQLite)
- `pythia/db/schema.py` is authoritative for Pythia tables; `resolver/db/schema.sql` for Resolver tables
- Config loaded via `pythia.config.load()` which reads `pythia/config.yaml`
- LLM providers abstracted through `forecaster/providers.py` (OpenAI, Google, Anthropic, XAI, Kimi, DeepSeek)
- All LLM calls logged to `llm_calls` table with cost, tokens, latency, error tracking. This includes RC/triage LLM passes (logged by providers module), grounding calls (logged explicitly in `_run_grounding_for_hazard` via `log_hs_llm_call` with `hazard_code=grounding_{hazard}`), and adversarial checks (logged by `web_research` module for evidence fetches and by providers module for synthesis calls).
- Env vars override config defaults; threshold env vars use `_env_float()` pattern
- Structured data connectors follow a standard pattern: `fetch_*()` → `store_*()` → `load_*()` → `format_*_for_prompt()` / `format_*_for_spd()`
- **Reasoning traces** are required in SPD output JSON for Track 1 (full ensemble). The `reasoning_trace` object captures the prior SPD, sequential evidence updates with numeric deltas, point estimate, and RC assessment. Track 2 requires only `prior` and `rc_assessment`. Traces are stored in `reasoning_trace_json` columns on `forecasts_raw` and `forecasts_ensemble`, validated by `forecaster/trace_validation.py` (diagnostic only), and analyzed by `generate_calibration_advice.py` for prior anchoring diagnostics.
- **Question-level web research pipeline is deprecated**: The `fetch_evidence_pack` / `_build_question_evidence_queries` / `_merge_question_evidence_packs` flow is bypassed. SPD prompts now receive structured data directly via `_load_structured_data()`: conflict forecasts, ReliefWeb, HDX Signals, HS grounding evidence, ACAPS, IPC, ACLED political, NMME, ENSO, seasonal TC, GDACS event history (FL/DR/TC), adversarial checks. The `question_research` table is no longer populated by the pipeline (only placeholder rows). Env vars `PYTHIA_RETRIEVER_ENABLED`, `PYTHIA_WEB_RESEARCH_ENABLED`, `PYTHIA_SPD_WEB_SEARCH_ENABLED`, `PYTHIA_FORECASTER_SELF_SEARCH` are set to "0" in the workflow.
- HS pipeline is per-hazard: each hazard gets its own RC grounding, RC call, triage grounding, and triage call (2-pass each: RC uses Gemini Flash for both passes; triage uses Gemini Pro for Pass 1 + Gemini Flash for Pass 2)
- **Grounding backend order**: Brave Search API is the primary grounding backend (direct, deterministic, $5/1K queries); OpenAI GPT-4.1-mini web search is the first fallback; Gemini Google Search grounding is the second fallback. Override with `PYTHIA_GROUNDING_PRIMARY_BACKEND=openai` or `=gemini` to change the primary. All three backends are attempted in order until one returns sources. RC, triage, and adversarial grounding all use this three-tier chain. Triage grounding calls are logged to `llm_calls` with `hazard_code=TRIAGE_GROUNDING_{hazard}` (RC grounding logged as `grounding_{hazard}`).
- RC and triage grounding use different signal categories and recency windows (RC: TRIGGER/DAMPENER/BASELINE; triage: SITUATION/RESPONSE/FORECAST/VULNERABILITY)
- **Grounding recency windows**: RC grounding uses moderate windows (ACE=30d, DR=60d, FL=30d, HW=30d, TC=60d) balancing change signal detection with sufficient search coverage for a monthly forecast cycle. Triage grounding uses wider windows (ACE=60d, DR=90d, FL=60d, HW=60d, TC=60d) for the operational picture. Both are defined in their respective `RECENCY_DAYS` dicts.
- **Grounding source steering**: All 10 grounding prompts (5 RC + 5 triage) include a `PRIORITIZE THESE SOURCES` section with hazard-specific source priority lists and a `RECENCY FILTER` instruction. RC prompts prioritize wire services and specialist sources for novelty; triage prompts elevate OCHA/humanitarian sources for the operational picture.

## LLM credit-retry safety net

When LLM providers return billing/quota-exhaustion errors (distinct from transient rate limits), a credit-retry wrapper pauses and retries, waiting for auto-recharge. Implemented in `forecaster/providers.py` inside `call_chat_ms`, wrapping the existing transient-retry inner loop.

**Provider-specific config** (`_CREDIT_RETRY_CONFIG`):
- OpenAI: 900s pause, 3 retries (worst-case 45min)
- Anthropic: 300s pause, 3 retries (worst-case 15min)
- Google: 600s pause, 3 retries (worst-case 30min)
- Kimi/DeepSeek: excluded (no credit retry)

**Detection** (`_is_billing_error`): Conservative per-provider rules that distinguish billing errors from rate limits. OpenAI: 429 + quota/billing keywords but NOT "rate limit". Anthropic: 400/403 + insufficient/billing/blocked. Google: 429 + RESOURCE_EXHAUSTED + quota/billing. Returns False when uncertain.

**Env-var overrides**: `PYTHIA_CREDIT_RETRY_PAUSE_{PROVIDER}` (seconds), `PYTHIA_CREDIT_RETRY_MAX_{PROVIDER}` (count). E.g. `PYTHIA_CREDIT_RETRY_PAUSE_OPENAI=600` reduces OpenAI pause to 10 minutes.

**Usage tracking**: `credit_retries_used`, `credit_retry_pauses_sec`, `billing_error_detected` fields added to the usage dict when billing errors are detected.

## Brave Search circuit breaker

Thread-safe circuit breaker (`pythia/web_research/brave_circuit_breaker.py`) that tracks consecutive Brave API failures across all call sites. When 3 consecutive raw API calls return errors, the breaker trips and all subsequent Brave calls are short-circuited (return immediately with `circuit_breaker_tripped` error). The fallback chain (OpenAI → Gemini) takes over grounding.

**Safety gate**: When the breaker is tripped, `_write_hs_triage` checks whether each hazard received grounding evidence (via `_hazard_has_grounding()` querying `llm_calls`). Hazards without grounding are blocked from forecasting (`need_full_spd = False`, `data_quality.brave_budget_gate = "blocked_no_grounding"`). Hazards that completed grounding before the breaker tripped proceed normally.

**Lifecycle**: Breaker resets at the start of each HS run (`main()` → `reset_brave_breaker()`). Stats logged at run end. `HS_BRAVE_BREAKER_TRIPPED` emitted as workflow output.

## Known failure modes

- **RC degradation without data sources**: RC assessments degrade severely when both `PYTHIA_HS_RESEARCH_WEB_SEARCH_ENABLED=0` and structured data connectors (ACLED, facts_deltas) are unavailable simultaneously. All country-hazard pairs may return RC Level 0 (baseline), even for active crisis countries. Always ensure at least one of web search or structured data is available for RC grounding.
- **Prediction market retriever**: Currently disabled (`PYTHIA_PM_RETRIEVER_ENABLED=0`). Metaculus returns 403, Polymarket returns HTTP 422, and each call times out at 30s — adding ~6 minutes of wasted wall time per run. Do not re-enable until upstream APIs are fixed.
- **DuckDB cache key mismatch** (fixed): The DuckDB connection pool was using raw URL strings rather than resolved paths as cache keys, causing 100+ `cache_event=miss` entries and a new connection per DB access. Fixed by normalizing to resolved paths.
- **API stale DB connection after sync** (fixed): `_ensure_read_connection()` in `app.py` cached a singleton DuckDB connection forever. When `maybe_sync_latest_db()` atomically replaced the DB file on disk (`os.replace()`), the open connection still read from the old file descriptor (Unix inode semantics). New pipeline runs were invisible to the dashboard until the API process restarted. Fixed by adding `_maybe_refresh_db()` to `_con()`: it periodically (every 60s) calls `maybe_sync_latest_db()` and, if a new DB was downloaded (signalled via `db_was_refreshed()` flag in `db_sync.py`), closes the old connection and opens a new one.
- **DuckDB WAL replay crash on startup** (fixed): After `_maybe_refresh_db()` replaced the DB file on disk while the old connection was still open, the stale `.wal` file from the old connection's `ensure_schema` run persisted. On next process restart, `duckdb.connect(read_only=False)` replayed the WAL which contained a duplicate `ALTER TABLE ADD COLUMN fetched_at`, crashing with `CatalogException: Column with name fetched_at already exists!`. Fixed by catching `CatalogException` in `_open_duckdb_connection()`, deleting the stale WAL file, and retrying the connection. Safe because the API's authoritative data comes from the synced DB file, not the WAL.
- **DuckDB WAL replay crash on artifact download** (fixed): Downloaded `pythia-resolver-db` artifacts could include stale `.wal` files, or WAL files were left behind between candidate iterations in the canonical DB discovery loop. DuckDB's WAL replay crashed with `InternalError: Failure while replaying WAL file … Calling DatabaseManager::GetDefaultDatabase with no default database set`. Fixed in two places: (1) `scripts/ci/db_signature.py` now calls `_remove_stale_wal()` to delete any `.wal` file before opening the DB; (2) all 11 workflow files that download and validate DB artifacts now explicitly `rm -f data/resolver.duckdb.wal` after copying the DB and before opening it.
- **ACAPS iso3 list values** (fixed): The ACAPS API sometimes returns `iso3` as a list instead of a string. All bulk-fetch functions in `pythia/tools/ingest_structured_data.py` (`_bulk_fetch_inform_severity`, `_bulk_fetch_risk_radar`, `_bulk_fetch_daily_monitoring`, `_bulk_fetch_humanitarian_access`, `_bulk_fetch_reliefweb`) now defensively coerce list values to strings before calling `.upper()`.
- **Adversarial checks / tail packs silenced by PYTHIA_WEB_RESEARCH_ENABLED=0** (fixed): The deprecated `PYTHIA_WEB_RESEARCH_ENABLED=0` flag (which disables the old question-level web research pipeline) was also blocking `fetch_evidence_pack` calls in adversarial checks and hazard tail packs. Both now temporarily override the env var to `"1"` around their `fetch_evidence_pack` calls, so they work independently of the deprecated flag.
- **Kimi kimi-k2.5 temperature constraint** (fixed): The kimi-k2.5 model only accepts `temperature=1`. The `_call_openai_compatible` function in `forecaster/providers.py` now clamps temperature to 1.0 for models in `_KIMI_FIXED_TEMPERATURE_MODELS`.
- **ACLED CAST expired token passthrough** (fixed): `get_access_token()` in `resolver/ingestion/acled_auth.py` now validates JWT expiry via `_jwt_is_valid()` before using environment-provided tokens. Expired tokens fall through to the refresh/password grant flow instead of being returned as-is.
- **ACAPS Humanitarian Access field names** (fixed): The ACAPS Humanitarian Access API does not return an `iso3` field in its response records (unlike INFORM Severity, Risk Radar, etc.). The bulk fetcher in `pythia/tools/ingest_structured_data.py` tries multiple field names (`iso3`, `iso`, `country_iso3`, `country_code`, `country_iso`) and falls back to a nested `country.iso3` dict lookup. A sample-record-keys log line is emitted on first record to aid debugging if the schema changes again.
- **conflictforecast.org wrong value column** (fixed): `_transform_csv()` in `resolver/connectors/conflictforecast.py` was selecting the `_target` column (ground truth, NaN for future periods) instead of `_all` (combined model forecast), producing 0 rows. Fixed by adding suffix-priority column selection (`_all` > `_text` > `_hist`, skip `_target`/`_naive`), a Backendless metadata skip set, and a median-value sanity check (onset metrics only).
- **NMME regionmask low country count** (fixed): `resolver/ingestion/nmme.py` used `countries_110` (coarsest Natural Earth resolution), which resolved only ~12 countries. Changed to `countries_10` (highest resolution) with `natural_earth_v5_1_2` preferred over `v5_0_0` for 170+ country coverage.
- **IDMC displacement hazard code mismatch** (fixed): `_load_idmc_conflict_flow_history_summary` and `_build_conflict_base_rate` in `forecaster/cli.py` filtered by `hazard_code = ?` (passing `'ACE'`), but IDMC writes data with `hazard_code='IDU'`. Fixed to use `IN (?, 'IDU')` to match both.
- **Gemini SPD timeouts** (fixed): Gemini Flash and Pro SPD calls timed out at 90s/120s respectively. Raised both to 300s in `forecaster/providers.py` (env-overridable via `PYTHIA_GOOGLE_SPD_TIMEOUT_FLASH_SEC` / `PYTHIA_GOOGLE_SPD_TIMEOUT_PRO_SEC`). HS triage timeout remains at 120s.
- **ACLED OAuth credential whitespace** (fixed): `_resolve_password_creds()` and `_resolve_refresh_token()` in `resolver/ingestion/acled_auth.py` read env vars without `.strip()`, so trailing whitespace/newlines from GitHub Secrets caused HTTP 415/400 on the OAuth password grant. Fixed by adding `.strip()` to all credential env var reads (`ACLED_USERNAME`, `ACLED_PASSWORD`, `ACLED_REFRESH_TOKEN`). Also added debug logging of the username in `_password_grant()`.
- **acled_political missing from default ingest** (fixed): `acled_political` was placed in `_SOURCE_ALIASES` (with a duplicate entry) instead of `_SOURCE_GROUPS` in `pythia/tools/ingest_structured_data.py`, so it never ran in default (all-sources) mode. Moved to `_SOURCE_GROUPS` and removed duplicates.
- **Debug bundle health check false statuses** (fixed): In `scripts/dump_pythia_debug_bundle.py`, HS Grounding check was reading wrong data source (now uses `hs_web_research_rows`), and Research Grounding reported FAIL when the retriever was intentionally disabled (now checks `retriever_enabled` flag).
- **CrisisWatch Cloudflare block** (fixed): crisisgroup.org is behind Cloudflare, returning 403 to programmatic fetches including Gemini grounding's internal fetcher. The old `site:crisisgroup.org` query in `crisiswatch_horizon.py` produced no results. Redesigned as `crisiswatch.py`: primary source is now the Playwright-scraped JSON file (`horizon_scanner/data/crisiswatch_latest.json`), with Gemini grounding as fallback (only attempted when JSON is missing/empty). Expanded injection from ACE-RC-only to ACE RC + triage + SPD prompts. Data persisted in the `crisiswatch_entries` DuckDB table.
- **CrisisWatch empty raw_text from fetch_via_gemini** (fixed): `_fetch_on_the_horizon()` and `_fetch_global_overview()` called `fetch_via_gemini()` and tried to extract text via `pack.get("markdown") or pack.get("raw_text")`, but `fetch_via_gemini` puts content into `pack.recent_signals`/`pack.structural_context` (not `markdown`/`raw_text`), and its internal JSON parser discards CrisisWatch's custom schema keys (`conflict_risks`, `countries`). Fixed by replacing `fetch_via_gemini` with a direct Gemini API call (`_call_gemini_grounding`) that returns raw text for CrisisWatch's own parsers.
- **Gemini 2.5 Flash Lite retired**: Replaced `gemini-2.5-flash-lite` with `gemini-2.5-flash` as the default grounding model across the system (`DEFAULT_GEMINI_CANDIDATES` in `gemini_grounding.py`, `_GEMINI_MODEL_CANDIDATES` in `crisiswatch.py`, cost entries in `model_costs.json` and `llm_logging.py`, fallback reference in `regime_change_llm.py`).
- **ACLED client silent stdout** (fixed): `acled_client.py` `main()` produced no stdout on skip, error, or empty-data paths, so `run_connectors.py` (which parses stdout for `rows=(\d+)`) reported 0 rows with no diagnostics. Fixed by adding `print(f"wrote {OUT_PATH} rows=...")` on all four exit paths in `main()`, upgrading `collect_rows()` error logging from `dbg()` to `LOG.warning()`, adding `_write_run_summary()` on the exception path, and adding a diagnostics JSON fallback in `run_connectors.py` that reads `acled_client_run.json` when the regex finds nothing. `acled_auth.py` `get_access_token()` also now emits `print("[acled_auth] ...")` at each auth stage (env token, refresh grant, password grant, exhausted) for CI log visibility regardless of `RESOLVER_DEBUG`.
- **IDMC double-run in backfill workflow** (fixed): IDMC ran twice in `resolver_update.yml` — once in the dedicated HELIX API step and again inside `run_connectors.py`. Fixed by adding `RESOLVER_SKIP_IDMC: "1"` to the "Run connectors" step env block.
- **Calibration advice dual-constraint failures** (fixed): `_upsert_advice()` in `pythia/tools/generate_calibration_advice.py` failed repeatedly due to the `calibration_advice` table having both the original 3-column PK `(as_of_month, hazard_code, metric)` and the newer 4-column unique index `ux_calibration_advice (as_of_month, hazard_code, metric, model_name)`. Root cause: the `CREATE TABLE` DDL in `schema.py` defined the 3-column PK, and **DuckDB does not support `ALTER TABLE DROP CONSTRAINT` for PRIMARY KEY constraints** — all `DROP CONSTRAINT` attempts failed silently regardless of constraint name. Fixed by **recreating the table**: `_migrate_calibration_advice_pk()` checks `information_schema.table_constraints` for a PK, and if found, uses CTAS to copy data into a temp table, `DROP TABLE` on the original (which cascades all dependencies including the PK — `ALTER TABLE RENAME` is also blocked by `DependencyException`), creates a new table without a PK, copies data back with column-safe SELECT (handling missing columns and NULL model_name), drops the temp table, and creates the 4-column unique index. The `CREATE TABLE` DDL in `schema.py` was also changed to `PRIMARY KEY (as_of_month, hazard_code, metric, model_name)` so fresh DBs never get the old PK. The upsert uses `INSERT INTO ... ON CONFLICT ... DO UPDATE SET`, and `_seed_default_advice` uses `ON CONFLICT ... DO NOTHING`.
- **Calibration advice hardcoded bucket thresholds** (fixed): `_bucket_case_sql()` in `generate_calibration_advice.py` hardcoded threshold values (e.g. `10000, 50000, 250000, 500000` for PA) that duplicated the canonical `PA_THRESHOLDS`/`FATAL_THRESHOLDS` lists from `compute_scores.py`. The imported constants `_bucket_index`, `PA_THRESHOLDS`, `FATAL_THRESHOLDS` were unused. Fixed by deriving the SQL `CASE` expression dynamically from the canonical threshold lists. Removed unused `_bucket_index` import.
- **Calibration advice silent exception swallowing** (fixed): Multiple `except Exception: pass` blocks in `generate_calibration_advice.py` (`_table_exists`, `_row_count`, `_ensure_findings_json_column`, `_migrate_calibration_advice_pk`) silently swallowed errors, making real DB connection failures or schema issues look like empty data. Added `LOGGER.warning` / `LOGGER.debug` to all critical silent exception paths.
- **Calibration workflow manual dispatch broken** (fixed): `compute_calibration_pythia.yml` used `github.event.workflow_run.id` to download the DB artifact, which is empty on `workflow_dispatch` (manual trigger). Replaced with a two-path strategy: (A) if triggered by `workflow_run`, download from the triggering run; (B) on manual dispatch or fallback, use canonical DB discovery (searches recent successful runs from Compute SPD Scores, Compute Calibration, Horizon Scanner Triage, and Ingest Structured Data workflows).
- **Vestigial "Export Facts: facts.csv rows: 0" in summary** (fixed): `summarize_connectors.py` always rendered the "## Export Facts" section because `_collect_export_summary()` was gutted in PR #610 (always returns empty dict with rows=0), but the condition `if export_info or export_error or mapping_debug_records` was always truthy. Fixed by gating on `export_error or has_export_rows` (where `has_export_rows = export_info.get("rows", 0) > 0`). The `mapping_debug_records` section renders independently.
- **Precedence engine dropping connector metadata** (fixed): `resolve_facts_frame()` in `resolver/tools/precedence_engine.py` (used by `run_pipeline.py`) built a new record dict with only key columns (`country_iso3`, `hazard_type`, `month`, `metric`, `value`, `selected_*`), discarding all metadata fields from the chosen row. This caused `publisher`, `source_type`, `source_url`, `confidence`, `definition_text`, `doc_title`, and other fields to be NULL in `facts_resolved` after pipeline writes. The CLI path (`_main_impl`) already carried these through but the programmatic `resolve_facts_frame` path did not. Fixed by adding passthrough of 13 metadata fields from the chosen row and mapping `selected_as_of`→`as_of_date`, `selected_tier`→`precedence_tier`, `selected_source`→`provenance_source` so the DB upsert can match column names. The comparison tool `tools/compare_gdacs_ifrc.py` also has a fallback classifier that identifies GDACS rows by `metric='in_need'` + DR/FL/TC hazard when publisher is NULL (for pre-fix data).
- **False-zero resolution contamination** (fixed): `compute_resolutions` defaulted all unresolved horizons to `value=0.0`, treating "no data" as "no impact". This was correct for ACLED fatalities (continuous coverage) but wrong for IFRC (PA), IDMC, FEWS NET IPC, and EM-DAT where absence means "unknown/not monitored". Fixed by implementing source-aware null handling: FATALITIES+ACE/ACO and EVENT_OCCURRENCE still default to 0; PA and PHASE3PLUS_IN_NEED horizons with no data are now left unresolved (no row written) so scoring skips them. Also added HW to UNRESOLVABLE_HAZARDS set.
- **DI, CU, HW hazards fully silenced**: DI (Displacement Influx), CU (Civil Unrest), and HW (Heatwave) are now blocked at the hazard catalog level (`BLOCKED_HAZARDS` in `db_writer.py`). They never enter RC assessment, triage, grounding, or question generation. This eliminates all wasted LLM/grounding calls on these unforecastable hazards. Historical data for these hazards is preserved in the database. HW/DI remain in `resolver/data/shocks.csv` and `UNRESOLVABLE_HAZARDS` in `compute_resolutions.py`.
- **Question overwrite bug destroying run provenance** (fixed): `scripts/create_questions_from_triage.py` used stable question_ids (e.g. `ETH_ACE_FATALITIES`) with a DELETE+INSERT pattern in `_upsert_question`. Each HS run destroyed the previous question's `hs_run_id`, `window_start_date`, and `target_month`. After the March 2026 run, all 524 questions pointed to the March HS run with `window_start_date=2026-04-01`, making December/January forecasts unresolvable while March forecasts were incorrectly scored. Additionally, `_llm_derived_window_starts` in `compute_resolutions.py` used `MIN(timestamp)` from `llm_calls` across all runs for shared question_ids, causing cross-run contamination. Fixed by: (1) making question_ids epoch-specific with `_{epoch_label}` suffix (e.g. `ETH_ACE_FATALITIES_2026-04`), (2) replacing DELETE+INSERT with INSERT-only (skip if exists), (3) fixing `target_month` to represent the 6th horizon month instead of the opening month, (4) removing the LLM-derived window override from `compute_resolutions.py` so the question's own `window_start_date` is authoritative. Recovery script: `scripts/recover_historical_questions.py`.
- **IPC API now integrated as supplementary source** (fixed): The legacy IPC connector (`pythia/ipc_phases.py`) was dead code. A new Resolver connector (`resolver/connectors/ipc_api.py`) fetches Phase 3+ data from `api.ipcinfo.org` for non-FEWS NET countries (Afghanistan, Pakistan, Guatemala, etc.). The unified module `pythia/food_security.py` tries FEWS NET first, falls back to IPC. The old `fewsnet_food_security.py` is now a backward-compat shim. The `ipc_phases.py` module is retained as dead code.
- **Gemini grounding unreliable — OpenAI primary** (fixed): Gemini Google Search grounding failed 6/6 times in test run fc_1774107846 (HTTP 200 but `grounded=False`). OpenAI web search (GPT-4.1-mini) succeeded 15/15. Swapped primary/fallback: OpenAI web search is now primary, Gemini is fallback. Helper functions `_try_openai_grounding` / `_try_gemini_grounding` extracted for reuse between RC and triage. Superseded by Brave Search as primary (see below).
- **OpenAI web search nondeterministic — Brave Search primary**: OpenAI web search (LLM-mediated) is nondeterministic and returns empty results ~40% of the time. Replaced with Brave Search API ($5/1K queries) as the primary grounding backend. Brave provides deterministic direct web search with freshness filtering. Three-tier fallback: Brave → OpenAI → Gemini. Override with `PYTHIA_GROUNDING_PRIMARY_BACKEND=openai` or `=gemini`. `BRAVE_SEARCH_API_KEY` env var required; if missing, falls through to OpenAI/Gemini gracefully. Brave is also wired as primary for adversarial check evidence searches. Backend module: `pythia/web_research/backends/brave_search.py`. Cost tracked via `cost_usd` in usage dict ($0.005 per query), logged to `llm_calls` as `model_id=brave-web-search`, `provider=brave`.
- **Triage grounding calls invisible in telemetry** (fixed): RC grounding logged to `llm_calls` via `log_hs_llm_call` with `hazard_code=grounding_{hazard}`, but triage grounding did not log at all. Added logging in `_run_triage_grounding_for_hazard` with `hazard_code=TRIAGE_GROUNDING_{hazard}` and `purpose=hs_triage_grounding`.
- **Adversarial checks gated behind deprecated flags** (fixed): Adversarial checks in `horizon_scanner.py` were gated on `PYTHIA_RETRIEVER_ENABLED=1 OR PYTHIA_HS_RESEARCH_WEB_SEARCH_ENABLED=1`, both deprecated (always 0 in workflow). The adversarial check module manages its own web search enablement internally. Simplified gate to check only `PYTHIA_ADVERSARIAL_CHECK_ENABLED` (default "1"). Added explicit `PYTHIA_ADVERSARIAL_CHECK_ENABLED: "1"` to workflow env block.
- **Grounding detail CSV case sensitivity and missing data** (fixed): `emit_grounding_detail_csv` in `dump_pythia_debug_bundle.py` used case-sensitive `LIKE '%grounding%'` which missed `GROUNDING_ACE` (uppercase). Fixed with `LOWER(hazard_code) LIKE '%grounding%'`. Also added fallback detection for markdown evidence packs (non-JSON `response_text`) by checking for "Grounded: True" / "Sources:" patterns and counting URLs.
- **Triage grounding jargon queries returning 0 sources** (fixed): `_run_triage_grounding_for_hazard` in `triage.py` used `query_label = f"{country} ({iso3}) {hazard_code} triage grounding"` as the search query — internal jargon that no web page contains. Replaced with natural-language keyword queries via `build_triage_grounding_query()` in `hs_triage_grounding_prompts.py` (e.g. "Iraq (IRQ) armed conflict violence displacement humanitarian situation 2026"). RC grounding queries were already using natural language and were not changed.
- **Adversarial check queries too verbose** (fixed): LLM-generated trigger-specific adversarial queries could be 15+ words, which search engines handle poorly. Added 10-word truncation post-processing in `_build_adversarial_queries()` in `adversarial_check.py`. Additionally, `fetch_evidence_pack()` in `web_research.py` was wrapping adversarial queries with "country context ... latest humanitarian situation" via the HS country pack retry path (because `_is_hs_country_pack()` matched the `hs_adversarial_check` purpose). This inflated the already well-formed queries, causing 7/9 adversarial searches to return 0 sources. Fixed by adding `hs_adversarial_check` to `_HS_PASSTHROUGH_PURPOSES` so it bypasses the HS retry wrapper.
- **Anthropic SPD max_tokens not wired through** (fixed): `_ANTHROPIC_SPD_MAX_OUTPUT` (from `PYTHIA_ANTHROPIC_SPD_MAX_TOKENS` env var) was defined in `providers.py` but never used — all Anthropic calls used `_ANTHROPIC_MAX_OUTPUT` regardless of purpose. Fixed by adding `purpose` parameter to `call_anthropic()` and `_call_provider_sync()`, so SPD/binary calls (`purpose in ("spd_v2", "binary_v2")`) use the higher limit (default 16384).
- **Resolver page IFRC/IDMC/ACLED showing "No data"** (fixed): Phase 1 of the backfill workflow uses `load_and_derive.py`, which set `publisher` from the adapter's `canonical_source` slug — writing lowercase values `"acled"`, `"idmc"`, `"ifrc_go"`. The API source registry filters used case-sensitive exact matches (`publisher = 'ACLED'`, `publisher = 'IDMC'`, `publisher = 'IFRC'`) which didn't match. Fixed in two places: (1) `load_and_derive.py` now maps source slugs to proper uppercase publisher names via `_SOURCE_TO_PUBLISHER` (`acled→ACLED`, `idmc→IDMC`, `ifrc_go→IFRC`); (2) API source registry filters now use `LOWER(publisher) IN (...)` for case-insensitive matching that handles both old and new data. IFRC filter also matches `ifrc_go` and `ifrc_montandon` variants.
- **Debug bundle seasonal_tc_loaded always False** (fixed): `emit_data_inject_inventory_csv` in `dump_pythia_debug_bundle.py` checked seasonal TC availability by calling `get_seasonal_tc_context_for_country()`, which internally opens its own DB connection via `pythia.db.schema.connect()`. In CI, the debug bundle opens the artifact DB via `--db`, but `schema.connect()` opens the default DB path (from `PYTHIA_DB_URL`), hitting a different/empty DB. Fixed by querying the `seasonal_tc_context_cache` table directly using the bundle's own `con` parameter and checking `COUNTRY_TO_BASINS` for TC basin exposure. Non-TC-exposed countries now show empty instead of False.
- **Debug bundle grounding stats report 100% failure** (fixed): `_load_grounding_subsystem_stats` and `_load_grounding_call_stats` in `dump_pythia_debug_bundle.py` tried `json.loads()` on `response_text` from grounding calls, but `_run_grounding_for_hazard` logged the markdown output from `_render_grounding_markdown()`, not JSON. Every `json.loads()` failed → `n_sources=0` → counted as error, making all grounding calls appear to have 0 sources (21/21 failures). Actual failure rate was 8/21. Fixed in two places: (1) `_run_grounding_for_hazard` now logs `json.dumps(pack, default=str)` instead of markdown; (2) both debug bundle functions have a markdown fallback that counts `- http` lines after `Sources:` for backward compatibility with old data.
- **RC grounding JSON truncation miscounts sources as failures** (fixed): The previous fix changed grounding logging from markdown to `json.dumps(pack, default=str)[:4000]`, but packs with 10+ sources produce JSON exceeding 4000 chars. The `[:4000]` truncation cut JSON mid-string, making `json.loads()` fail in the health report parser, miscounting 4 successful calls (IRQ_ACE, IRQ_DR, KEN_DR, SOM_DR) as failures. Fixed by logging a compact summary dict (query, grounded, n_sources, source_urls, structural_context snippet, recent_signals, backend, error) instead of the full pack — always under 3KB, always parseable.
- **RC grounding queries return 0 sources for many country-hazard pairs** (fixed): `_HAZARD_QUERY_LABELS` in `regime_change_llm.py` used internal jargon ("armed conflict escalation signals", "flood risk river levels") that no web page contains, causing OpenAI web search to return empty results. Combined with tight 14-day recency windows, the search space was too constrained. Fixed by: (1) rewriting labels to use broad humanitarian vocabulary ("armed conflict violence security situation", "flooding flood displacement humanitarian impact"); (2) appending the current year to queries for temporal anchoring; (3) widening `RECENCY_DAYS` in `rc_grounding_prompts.py` (ACE/FL/HW: 14→30 days, DR/TC: 30→60 days); (4) promoting annotation-based sources to verified in `openai_web_search.py` when `web_search_call.action.sources` is empty but URL annotations exist. Further refined: ACE label changed from "security situation" to "displacement humanitarian situation" (matches proven triage pattern), FL label dropped redundant "flood" (keeping just "flooding", matches triage pattern).
- **RC grounding Gemini fallback uses same failing query** (fixed): When OpenAI primary returned empty sources, the Gemini fallback received the identical query string, so it also failed — the fallback never rescued anything. Fixed by reformulating the query for the fallback: stripping the year and appending "humanitarian crisis update" to broaden the search space.
- **Triage grounding logging markdown truncation** (fixed): Triage grounding logged `response_text` as markdown from `_render_grounding_markdown(pack)` truncated at 2000 chars, but the Sources section never appeared because structural_context + recent_signals consumed 1300+ chars. Health report parser saw 0 sources for all triage grounding calls. Fixed by switching to the same compact JSON format (`build_compact_grounding_log()`) used by RC grounding, with pre-computed `n_sources` count.
- **Adversarial check Brave calls invisible in health report** (fixed): Adversarial checks used Brave Search as primary backend but didn't log the calls to `llm_calls`. The `fetch_evidence_pack` fallback logged to `phase='hs_web_research'`, but successful Brave calls were invisible. Fixed by adding `_log_adversarial_brave_call()` that logs to `llm_calls` with `hazard_code='ADVERSARIAL_{hazard}'` using compact JSON format. The health report now merges both `hs_web_research` and `ADVERSARIAL_%` rows for the Adversarial Checks section.
- **Grounding detail CSV missed compact JSON format** (fixed): `emit_grounding_detail_csv` in the debug bundle only parsed the old `sources` list format, missing the compact format's `n_sources` and `source_urls` fields. RC grounding showed 0 sources despite 10 actual sources. Fixed by adding compact format detection (`n_sources`, `source_urls` fields) before falling back to `sources` list counting.

## Canonical DB artifact discovery

The `pythia-resolver-db` DuckDB artifact is shared across multiple workflows. Each workflow's "Download canonical resolver DB" step searches for the most recent successful run from candidate workflow types:

1. **Horizon Scanner Triage** (`run_horizon_scanner.yml`) — DB_SOURCE=`pipeline`
2. **Resolver Update** — DB_SOURCE=`backfill` (primary ingestion: single-job workflow with 5 phases)
3. **Ingest Structured Data** (`ingest-structured-data.yml`) — DB_SOURCE=`ingest` (mid-cycle refresh for fast-changing sources)
4. **Compute SPD Scores** (`compute_scores.yml`) — upstream of calibration
5. **Compute Calibration Weights & Advice** (`compute_calibration_pythia.yml`) — calibration pipeline

Candidates are sorted by `createdAt` descending; the first one that downloads successfully and passes the DB signature check is used. If a new workflow is added that produces `pythia-resolver-db`, it must be added as a candidate source in all workflows that consume the artifact. The `compute_scores.yml` and `compute_calibration_pythia.yml` workflows use the triggering `workflow_run.id` as the primary source, falling back to canonical discovery on manual `workflow_dispatch`.

## Consolidated backfill workflow

The `resolver_update.yml` workflow is the primary data ingestion point (runs 15th monthly). It is a single-job workflow with 5 sequential phases operating on one DuckDB file:

```
Phase 1 (fatal):    Resolver connectors (IDMC, ACLED, IFRC) → facts_resolved + acled_monthly_fatalities
Phase 2 (non-fatal): Resolution sources (FEWS NET IPC, IPC API, GDACS) → facts_resolved
Phase 3 (non-fatal): Structured data (conflict forecasts, ACAPS, ReliefWeb, ACLED political, NMME) → Pythia tables
Phase 4 (non-fatal): Context sources (ENSO, Seasonal TC, HDX Signals, CrisisWatch) → Pythia tables
Phase 5:            Verify + export pythia-resolver-db artifact
```

Phase 1 failures are fatal (core resolution data). Phases 2-4 use `continue-on-error: true` so individual source failures don't block the entire backfill. Backfill depth is controlled by `FEWSNET_MONTHS` (120 on reset, 12 otherwise), `IPC_API_MONTHS` (120 on reset, 24 otherwise), and `GDACS_MONTHS` (135 on reset, 3 otherwise).

The `ingest-structured-data.yml` workflow is a mid-cycle refresh for fast-changing sources (weekly Sunday 03:00 UTC). Scheduled runs only refresh `conflict`, `gdacs`, `reliefweb`, and `acled_political`. Manual dispatch supports all sources for debugging.

## DB-first context data pattern

Context sources (ENSO, Seasonal TC, HDX Signals, CrisisWatch) follow a DB-first loading pattern:

1. During backfill (Phase 4): data is fetched from upstream sources and stored in DB tables (`enso_state`, `seasonal_tc_context_cache`, `hdx_signals`, `crisiswatch_entries`)
2. At HS/forecaster runtime: prompt loaders try DB first, fall back to live fetch/file cache if DB is empty
3. The HS workflow no longer downloads the seasonal TC artifact — TC data is in the DB artifact

This ensures the DB artifact is fully self-contained and reproducible.

## Run health diagnostics

- `hs_country_evidence` and `question_evidence` CSVs are the primary artifacts for diagnosing structured data connector health post-run.
- If a connector shows "unavailable" across all countries, it indicates an upstream data gap, a connector bug, or a missing environment secret.
- After applying fixes, a clean re-run must produce a queryable DuckDB artifact before connector health can be verified.
- The `inspect_resolver_duckdb.yml` workflow includes 7 data quality checks: conflict forecast value range validation (warns if probability values > 10), seasonal forecast country count, empty connector table warnings, IDMC/IDU hazard code consistency, conflict forecast staleness (> 45 days), HDX Signals note (cached as CSV, not in DB), and per-country conflict forecast sampling (IRN, SOM, ETH, SDN, UKR).
- **CrisisWatch diagnostics** in the debug bundle (`scripts/dump_pythia_debug_bundle.py`): (1) Traffic-light health check in `_evaluate_pipeline_health` — reports OK/WARN/FAIL with country counts, arrow breakdown, and alerts; WARN if 0 entries, <10 countries, or no arrow data; (2) Per-country `crisiswatch_arrow` column in the data inject inventory CSV showing arrow direction and alert type; (3) `crisiswatch_health` section in health report JSON with arrow counts, alert counts, notable (deteriorated/alert) entries, and countries missing CrisisWatch data.
- **Food security diagnostics** in the debug bundle: Food security data comes from two separate connectors: `fewsnet_ipc` (48 FEWS NET countries, no auth) and `ipc_api` (non-FEWS NET countries, requires `IPC_API_KEY`). The data inject inventory shows per-country row counts (`fewsnet_ipc_rows`, `ipc_api_rows`) and source attribution (`food_security_source`) for each connector. The health report flags countries with DR questions but no food security data. The `ipc_phases` table (from the legacy `pythia/ipc_phases.py` module) is dead code and is not used for diagnostics.

## LLM ensemble configuration

The forecast ensemble is defined in `pythia/config.yaml` under `llm.profiles.prod.ensemble` (7 models):

| Provider | Model | Notes |
|----------|-------|-------|
| OpenAI | gpt-5.2 | Reasoning model (`reasoning_effort=high`), temperature not supported |
| Anthropic | claude-sonnet-4-6 | Standard model, low temperature |
| Google | gemini-3.1-pro-preview | Thinking model (`thinkingLevel=medium`) |
| Google | gemini-3-flash-preview | Thinking model (`thinkingLevel=low`), also used for Track 2 |
| Kimi | kimi-k2.5 | Reasoning model, requires `temperature=1.0` |
| DeepSeek | deepseek-reasoner | Native reasoning model, temperature ignored by API |
| OpenAI | gpt-5-mini | Reasoning model (`reasoning_effort=medium`) |

Purpose-specific overrides: `hs_fallback: openai:gpt-5.2`, `scenario_writer: google:gemini-3-flash-preview`.

Model costs are in `pythia/model_costs.json` (input/output cost per 1K tokens in USD).

## GitHub Actions workflows

| Workflow | Purpose | Schedule |
|----------|---------|----------|
| `run_horizon_scanner.yml` | Full HS + forecaster pipeline | Manual / triggered |
| `resolver_update.yml` | Primary data ingestion (5-phase backfill) | 15th monthly |
| `ingest-structured-data.yml` | Mid-cycle refresh (conflict, GDACS, ReliefWeb, ACLED political) | Weekly Sun 03:00 UTC |
| `compute_resolutions.yml` | Ground truth resolution | After forecaster |
| `compute_scores.yml` | Brier/log/CRPS scoring | After resolutions |
| `compute_calibration_pythia.yml` | Calibration weights + advice | After scores |
| `refresh-enso.yml` | ENSO state/forecast refresh | Scheduled |
| `refresh-seasonal-tc.yml` | Seasonal TC forecast refresh | Scheduled |
| `refresh-crisiswatch.yml` | CrisisWatch Playwright scraper | 3rd monthly |
| `ingest-nmme.yml` | NMME seasonal forecasts (deprecated schedule) | Manual |
| `inspect_resolver_duckdb.yml` | Data quality checks | Manual |
| `forecaster-ci.yml` | SPD unit tests | On push/PR |
| `resolver-ci-fast.yml` | Fast resolver tests | On push/PR |
| `resolver-smoke.yml` | Resolver smoke tests | On push/PR |
| `web-ci.yml` | Dashboard build/lint | On push/PR |
| `ci-lint.yml` / `lint.yml` | Code linting | On push/PR |
| `manual_test.yaml` | Test mode pipeline run | Manual |
| `mark_test_runs.yml` | Retroactive test data marking | Manual |
| `build_dashboard_data.yml` | Dashboard data build | Triggered |
| `publish_latest_data.yml` | Publish latest data | Triggered |
| `publish_snapshot.yml` | Publish DB snapshot | Manual |
| `purge_hs_run.yml` | Delete an HS run | Manual |
| `resolver-initial-backfill.yml` | Full historical backfill | Manual |
| `resolver-simple-snapshot.yml` | Simple DB snapshot | Manual |
| `resolver-snapshot-from-db.yml` | Snapshot from existing DB | Manual |
| `upload-repaired-db.yml` | Upload repaired DB artifact | Manual |

## Supported hazards and metrics

Hazards allowed (from `config.yaml`): FL, DR, TC, HW, ACE, DI, CU, EC, PHE. MULTI/OT disabled; ACO retired in favor of ACE.

**Active pipeline hazards**: Only **ACE, DR, FL, TC** are processed by the pipeline. DI, CU, and HW are fully silenced — they are blocked at the hazard catalog level (`BLOCKED_HAZARDS` in `db_writer.py`) and never enter RC assessment, triage, grounding, or question generation. This saves all LLM/grounding calls previously spent on these unforecastable hazards. The pipeline processes exactly 4 hazards per country (minus seasonal screen-outs for TC).

Question generation (`scripts/create_questions_from_triage.py`) supports:
- **ACE**: FATALITIES, PA
- **DR**: PHASE3PLUS_IN_NEED, EVENT_OCCURRENCE (PA remapped to PHASE3PLUS_IN_NEED for FEWS NET countries)
- **FL**: PA, EVENT_OCCURRENCE
- **TC**: PA, EVENT_OCCURRENCE

## Structured data bulk ingest

`pythia/tools/ingest_structured_data.py` orchestrates bulk fetch/store for all structured data connectors. The `_SOURCE_GROUPS` dict maps group names to table names. Currently registered groups: `views`, `conflictforecast`, `acledcast`, `acaps_inform_severity`, `acaps_risk_radar`, `acaps_daily_monitoring`, `acaps_humanitarian_access`, `ipc`, `reliefweb`, `acled_political`, `nmme`, `gdacs`, `fewsnet_ipc`, `ipc_api`. Each group has a `_bulk_fetch_*` function that parallelizes API calls across countries via `ThreadPoolExecutor`. The `acled_political` group fetches event-level political data (protests, riots, strategic developments) from the ACLED API via `pythia.acled_political.fetch_acled_political_events` and stores via `store_acled_political_events`. The `gdacs` group delegates to `resolver.tools.run_pipeline` with `--connectors gdacs`, which writes to `facts_resolved` and `facts_deltas` via the standard Resolver pipeline. GDACS, FEWS NET IPC, and IPC API are self-storing sources (no per-country store function). The `ipc_api` group delegates to `resolver.tools.run_pipeline` with `--connectors ipc_api`, fetching IPC Phase 3+ population estimates from api.ipcinfo.org for countries not covered by FEWS NET; requires `IPC_API_KEY`; backfill depth controlled via `IPC_API_MONTHS` (default 24). The `fewsnet_ipc` group delegates to `resolver.tools.run_pipeline` with `--connectors fewsnet_ipc`, fetching IPC Phase 3+ population estimates from the FEWS NET Data Warehouse; backfill depth controlled via `FEWSNET_MONTHS` (default 12; 120 for backfill to 2016). The `ingest-structured-data.yml` workflow runs weekly (Sunday 03:00 UTC) as a mid-cycle refresh for fast-changing sources (`conflict`, `gdacs`, `reliefweb`, `acled_political`). The primary ingestion of all sources happens in the `resolver_update.yml` backfill workflow (15th monthly). GDACS backfill depth is controlled via the `gdacs_months` workflow input (default 3; set to 135 for full backfill to 2015).
