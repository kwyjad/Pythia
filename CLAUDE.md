# CLAUDE.md

This file provides guidance for Claude Code when working with the Pythia codebase.

## Project overview

Pythia is an end-to-end AI forecasting system for humanitarian crises. It scans countries for emerging hazards, produces triage signals, generates research briefs, runs LLM ensembles for probabilistic forecasts (SPDs), scores and calibrates them, and serves outputs via a FastAPI API and Next.js dashboard.

**Core pipeline:**
```
Resolver facts/base rates -> Horizon Scanner per-hazard pipeline (RC grounding → RC → triage grounding → triage)
  -> Structured data connectors (ReliefWeb, ACAPS, IPC, ACLED political, ENSO, seasonal TC, HDX Signals, ACLED CAST)
  -> Adversarial checks (RC L2+) -> Forecaster SPD ensemble
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
    connectors/        #   Connector protocol + registry (ACLED, IDMC) + forecast registry (VIEWS, CF.org, ACLED CAST)
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
    adversarial_check.py # Counter-evidence checks for RC Level 2+ cases
    ipc_phases.py      # IPC food security phase connector
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
- `pythia/config.yaml` — Central configuration (LLM profiles, hazards, bucket specs)
- `horizon_scanner/horizon_scanner.py` — HS main entrypoint (~1700 LOC)
- `horizon_scanner/regime_change.py` — RC scoring (score = likelihood x magnitude, 4 levels)
- `horizon_scanner/regime_change_llm.py` — Per-hazard RC LLM pipeline (2-pass: Gemini Flash + GPT-5-mini)
- `horizon_scanner/triage.py` — Per-hazard triage LLM pipeline (2-pass, ACLED low-activity filter)
- `horizon_scanner/rc_prompts.py` — Per-hazard RC prompt builders (ACE, DR, FL, HW, TC with calibration anchors)
- `horizon_scanner/hs_triage_prompts.py` — Per-hazard triage prompt builders (scoring anchors, RC context injection)
- `horizon_scanner/rc_grounding_prompts.py` — RC-specific grounding queries (TRIGGER/DAMPENER/BASELINE signals)
- `horizon_scanner/hs_triage_grounding_prompts.py` — Triage-specific grounding queries (SITUATION/RESPONSE/FORECAST/VULNERABILITY)
- `horizon_scanner/enso/enso_module.py` — ENSO state/forecast scraper (IRI/CPC, 7-day cache)
- `horizon_scanner/seasonal_tc/seasonal_tc_runner.py` — Seasonal TC forecast orchestrator (TSR + NOAA CPC + BoM)
- `horizon_scanner/seasonal_tc/__init__.py` — Country-to-basin mapping + cached TC context reader
- `horizon_scanner/hdx_signals.py` — HDX Signals connector (OCHA automated crisis monitoring, indicator-to-hazard mapping)
- `horizon_scanner/conflict_forecasts.py` — Unified conflict forecast loader (VIEWS + conflictforecast.org + ACLED CAST)
- `forecaster/cli.py` — Forecaster main runner (~7100 LOC)
- `pythia/api/app.py` — FastAPI application (~3200 LOC)
- `pythia/tools/compute_resolutions.py` — Resolves forecasts against Resolver ground truth
- `pythia/tools/compute_scores.py` — Brier/log/CRPS scoring per horizon
- `pythia/tools/compute_calibration_pythia.py` — Calibration weights + LLM advice
- `resolver/connectors/protocol.py` — Connector protocol (21-column canonical schema contract)
- `resolver/connectors/__init__.py` — Connector registry (discover_connectors) + FORECAST_REGISTRY (VIEWS, CF.org, ACLED CAST)
- `resolver/connectors/acled_cast.py` — ACLED CAST connector (event-count forecasts by type: total/battles/ERV/VAC)
- `resolver/tools/run_pipeline.py` — Pipeline orchestrator (fetch -> validate -> enrich -> precedence -> deltas -> DuckDB)
- `resolver/tools/enrich.py` — Enrichment (registry lookups, ym derivation, defaults)
- `resolver/tools/precedence_config.yml` — Precedence tier policy
- `pythia/prediction_markets/retriever.py` — Prediction market signal retriever (runs at research time, queries Metaculus/Polymarket/Manifold)
- `pythia/acaps.py` — ACAPS unified connector (4 datasets: INFORM Severity, Risk Radar, Daily Monitoring, Humanitarian Access)
- `pythia/ipc_phases.py` — IPC food security phase classification connector
- `pythia/acled_political.py` — ACLED event-level political data (protests, riots, strategic developments)
- `pythia/adversarial_check.py` — Counter-evidence searches for RC Level 2+ (devil's advocate)
- `horizon_scanner/reliefweb.py` — ReliefWeb humanitarian reports connector
- `forecaster/hazard_prompts.py` — Hazard-specific reasoning guidance for SPD prompts
- `scripts/ci/snapshot_prompt_artifact.py` — Prompt version snapshot script
- `pythia/tools/generate_calibration_advice.py` — Per-hazard/metric calibration advice generation

## Databases

Two DuckDB databases:

**Pythia DB** (`PYTHIA_DB_URL`): system of record
- `hs_runs`, `hs_triage` — Horizon Scanner outputs (triage scores, RC fields)
- `hs_hazard_tail_packs` — RC-triggered hazard evidence packs
- `hs_adversarial_checks` — Counter-evidence for RC Level 2+ (devil's advocate)
- `seasonal_forecasts` — NMME country-level temp/precip anomalies (monthly from CPC)
- `conflict_forecasts` — VIEWS + conflictforecast.org + ACLED CAST conflict predictions (PK: source, iso3, hazard_code, metric, lead_months, forecast_issue_date)
- `questions`, `question_research` — Seeded questions + research briefs
- `forecasts_raw`, `forecasts_ensemble` — Per-model + aggregated SPDs
- `resolutions` — Ground truth values per (question_id, horizon_m)
- `scores` — Brier/log/CRPS per (question, horizon, model)
- `calibration_weights`, `calibration_advice` — Per hazard/metric weights + LLM advice
- `reliefweb_reports` — ReliefWeb humanitarian situation reports
- `acled_political_events` — ACLED event-level political data
- `ipc_phases` — IPC food security phase populations
- `acaps_inform_severity`, `acaps_inform_severity_trend` — ACAPS INFORM severity scores
- `acaps_risk_radar` — ACAPS forward-looking risk assessments
- `acaps_daily_monitoring` — ACAPS analyst-curated daily updates
- `acaps_humanitarian_access` — ACAPS humanitarian access scores
- `llm_calls` — Full telemetry (cost, tokens, latency, errors)

**Resolver DB** (`resolver/db/schema.sql`): fact ingestion
- `facts_resolved` — Precedence-resolved facts (unique on ym, iso3, hazard_code, metric, series_semantics)
- `facts_deltas` — Monthly flow changes
- `snapshots`, `manifests`, `meta_runs` — Pipeline metadata

## Resolver architecture

The Resolver was refactored in PR #610 to a connector-based architecture. Defunct legacy connectors (DTM, EM-DAT ingestion, GDACS, HDX, IPC, ODP, ReliefWeb, UNHCR, WFP, WHO, WorldPop) were removed.

**Connector protocol** (`resolver/connectors/protocol.py`): Every data source implements a `Connector` protocol with a `name` attribute and a `fetch_and_normalize()` method that returns a DataFrame with exactly 21 canonical columns (event_id, iso3, hazard_code, metric, value, as_of_date, publisher, etc.).

**Active connectors** (`resolver/connectors/__init__.py` REGISTRY):
- `acled` — ACLED conflict/fatalities data (wraps `resolver/ingestion/acled_client`)
- `idmc` — IDMC internal displacement data (wraps `resolver/ingestion/idmc/`)
- `ifrc_montandon` — IFRC Go connector (stubbed, not yet active)

**Pipeline orchestrator** (`resolver/tools/run_pipeline.py`):
```
discover_connectors() -> fetch_and_normalize() per connector
  -> validate_canonical() -> enrich() + derive_ym()
  -> precedence_engine (tiered source resolution) -> make_deltas()
  -> write to DuckDB (facts_resolved + facts_deltas)
```

**Precedence policy** (`resolver/tools/precedence_config.yml`):
- Tier 0: IFRC Montandon/ACLED (highest priority; IFRC Montandon is the active source for natural hazard PA: FL, DR, TC, HW)
- Tier 1: IDMC
- Tier 2: EM-DAT (historical read-only, no active connector; replaced by IFRC Montandon)

**Transform adapters** (`resolver/transform/adapters/`): ACLED and IDMC adapters normalize source-specific schemas to the common format used by the precedence engine.

**NMME seasonal forecasts** (`resolver/ingestion/nmme.py` + `resolver/tools/ingest_nmme.py`):
Separate from the Connector pipeline. Fetches NMME ensemble mean anomalies from CPC FTP, computes area-weighted country averages using xarray + regionmask, and writes to the `seasonal_forecasts` table in Pythia DB. Injected into HS triage/RC prompts via `horizon_scanner/seasonal_context.py` (`climate_data` kwarg) and into forecaster prompts via `research_json["nmme_seasonal_outlook"]`. Run: `python -m resolver.tools.ingest_nmme`.

**Conflict forecast connectors** (`resolver/connectors/views.py`, `resolver/connectors/conflictforecast.py`, `resolver/connectors/acled_cast.py`):
Separate from the Connector pipeline (use `FORECAST_REGISTRY`, not `REGISTRY`). VIEWS connector fetches ML-based fatality predictions from the VIEWS API (`views_predicted_fatalities`, `views_p_gte25_brd`, leads 1–6). conflictforecast.org connector fetches news-based risk scores from Backendless API (`cf_armed_conflict_risk_3m`, `cf_armed_conflict_risk_12m`, `cf_violence_intensity_3m`). ACLED CAST connector fetches event-count forecasts via OAuth2 API (`cast_total_events`, `cast_battles_events`, `cast_erv_events`, `cast_vac_events`, 6-month lead), aggregated from admin1 to country level. All three write to `conflict_forecasts` table. Loaded into ACE prompts via `horizon_scanner/conflict_forecasts.py`. Run: `python -m resolver.tools.fetch_conflict_forecasts`.

**ENSO state and forecast** (`horizon_scanner/enso/enso_module.py`):
Scrapes IRI/CPC ENSO Quick Look page for current ENSO state, Niño 3.4 anomaly, 9-season probabilistic forecast, multi-model plume averages, and IOD state. Cached as JSON with 7-day expiry. Injected into RC and triage prompts for DR, FL, HW, TC hazards via `get_enso_prompt_context()`. Refreshed by `.github/workflows/refresh-enso.yml`.

**Seasonal TC forecasts** (`horizon_scanner/seasonal_tc/`):
Aggregates basin-level seasonal TC forecasts from TSR (PDF extraction), NOAA CPC (press release scraping), and BoM (outlook scraping) across 8 basins. Country-to-basin mapping in `__init__.py`. Cached as JSON; `get_seasonal_tc_context_for_country(iso3)` returns prompt-ready text. Refreshed by `.github/workflows/refresh-seasonal-tc.yml`.

**HDX Signals** (`horizon_scanner/hdx_signals.py`):
Downloads OCHA's HDX Signals CSV from CKAN API. Indicator-to-hazard mapping (acled_conflict→ACE, ipc_food_insecurity→DR, etc.). Filtered by country, hazard, recency (180 days). Injected into RC and triage prompts for all hazards via `format_hdx_signals_for_prompt()`.

**ICG CrisisWatch "On the Horizon"** (`horizon_scanner/crisiswatch_horizon.py`):
Monthly fetch of ICG's forward-looking conflict risk/resolution flags via Gemini grounding search. Called once per HS run, cached in-memory, injected into RC prompts for flagged countries.

Run the pipeline: `python -m resolver.tools.run_pipeline [--connectors acled idmc] [--db path/to/resolver.duckdb]`

## Per-horizon architecture

Forecasts cover a 6-month window. Each question has `window_start_date` and `target_month` (= month 6).

- **Resolutions**: `compute_resolutions` resolves each horizon independently against Resolver's `facts_resolved.created_at` for ordering.
- **Scoring**: `compute_scores` scores each (question, horizon_m) pair separately using Brier, log loss, and CRPS.
- **Calibration**: `compute_calibration_pythia` aggregates scores across horizons to produce per-model weights.

## Regime Change (RC) scoring

RC detects departures from historical base rates (distinct from triage_score which measures overall risk).

- `score = likelihood x magnitude`, clamped [0, 1]
- **Levels** (env-overridable): L0 (likelihood < 0.45 or score < 0.25), L1, L2 (likelihood >= 0.60, magnitude >= 0.50), L3 (likelihood >= 0.75, magnitude >= 0.60)
- L2+ forces `need_full_spd = TRUE` and triggers hazard tail pack generation
- Distribution check warns when too many assessments exceed expected proportions
- See `docs/hs_regime_change.md` for full details

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
| `PYTHIA_LLM_PROFILE` | LLM profile: `test` or `prod` |
| `PYTHIA_LLM_CONCURRENCY` | Max concurrent LLM calls |
| `PYTHIA_API_TOKEN` | Admin API auth token |
| `PYTHIA_WEB_RESEARCH_ENABLED` | Enable shared retriever (0/1) |
| `HS_MAX_WORKERS` | HS concurrent country workers |
| `FORECASTER_RESEARCH_MAX_WORKERS` | Research phase concurrency |
| `FORECASTER_SPD_MAX_WORKERS` | SPD phase concurrency |
| `PYTHIA_HS_RC_LEVEL*_*` | RC threshold overrides |
| `PYTHIA_HS_RC_DIST_WARN_*` | RC distribution warning thresholds |
| `PYTHIA_PREDICTION_MARKETS_ENABLED` | Enable prediction market retriever (0/1) |
| `ACAPS_EMAIL` / `ACAPS_PASSWORD` | ACAPS API credentials |

Provider API keys: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `KIMI_API_KEY`, `DEEPSEEK_API_KEY`.

## Build and run

```bash
# Install dependencies
pip install -r python_library_requirements.txt

# Initialize DB schema
python3 -c "from pythia.db.schema import ensure_schema; ensure_schema()"

# Run Resolver pipeline (fetch facts from ACLED + IDMC -> DuckDB)
python3 -m resolver.tools.run_pipeline
# Or specific connectors: python3 -m resolver.tools.run_pipeline --connectors acled idmc

# Ingest NMME seasonal forecasts (temp/precip anomalies from CPC FTP)
python3 -m resolver.tools.ingest_nmme
# Or specific month: python3 -m resolver.tools.ingest_nmme --year-month 202603

# Fetch conflict forecasts (VIEWS + conflictforecast.org + ACLED CAST -> conflict_forecasts table)
python3 -m resolver.tools.fetch_conflict_forecasts
# Or specific sources: python3 -m resolver.tools.fetch_conflict_forecasts --sources views conflictforecast_org acled_cast
# Dry run: python3 -m resolver.tools.fetch_conflict_forecasts --dry-run

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
- `pythia/web_research/backends/gemini_grounding.py` — Gemini grounding backend

## Code conventions

- Copyright header on every file: `# Pythia / Copyright (c) 2025 Kevin Wyjad`
- License: Pythia Non-Commercial Public License v1.0
- DuckDB is the only database backend (no Postgres/SQLite)
- `pythia/db/schema.py` is authoritative for Pythia tables; `resolver/db/schema.sql` for Resolver tables
- Config loaded via `pythia.config.load()` which reads `pythia/config.yaml`
- LLM providers abstracted through `forecaster/providers.py` (OpenAI, Google, Anthropic, XAI, Kimi, DeepSeek)
- All LLM calls logged to `llm_calls` table with cost, tokens, latency, error tracking
- Env vars override config defaults; threshold env vars use `_env_float()` pattern
- Structured data connectors follow a standard pattern: `fetch_*()` → `store_*()` → `load_*()` → `format_*_for_prompt()` / `format_*_for_spd()`
- Research LLM stage is deprecated; evidence now flows from structured data connectors (ReliefWeb, ACAPS, IPC, ACLED political, NMME, ENSO, seasonal TC, HDX Signals, VIEWS, conflictforecast.org, ACLED CAST, ICG CrisisWatch) into HS and SPD prompts
- HS pipeline is per-hazard: each hazard gets its own RC grounding, RC call, triage grounding, and triage call (2-pass each: Gemini Flash + GPT-5-mini)
- RC and triage grounding use different signal categories and recency windows (RC: TRIGGER/DAMPENER/BASELINE; triage: SITUATION/RESPONSE/FORECAST/VULNERABILITY)

## Post-Edit Documentation Requirements

After making any code changes, evaluate whether the following files need updating and update them if so:

- **README.md** – Update if you've changed setup steps, dependencies, usage instructions, file structure, or how to run the project.
- **docs/fred_overview.md** – This is a plain-English description of the system for non-technical readers. Update it if you've changed what the system does, how it behaves, its inputs/outputs, or its overall logic. Avoid technical jargon; explain changes in terms of what the system now does differently from a user perspective. Assume readers understand forecasting and humanitarian data, but not code.

Do not update these files for trivial changes (e.g. formatting, minor refactors with no behavioral change). Use your judgment.
