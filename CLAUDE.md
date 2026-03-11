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
  -> Structured data connectors (ReliefWeb, ACAPS, IPC, ACLED political, ENSO, seasonal TC, HDX Signals, ACLED CAST)
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
    adversarial_check.py # Counter-evidence checks for RC Level 1+ cases
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
- `pythia/prediction_markets/retriever.py` — Prediction market signal retriever (currently disabled by default via `PYTHIA_PM_RETRIEVER_ENABLED=0`; Metaculus returns 403, Polymarket returns 422, each times out at 30s)
- `scripts/print_forecaster_ensemble.py` — Ensemble diagnostic script (must be invoked as `python -m scripts.print_forecaster_ensemble`, not directly)
- `pythia/acaps.py` — ACAPS unified connector (4 datasets: INFORM Severity, Risk Radar, Daily Monitoring, Humanitarian Access)
- `pythia/ipc_phases.py` — IPC food security phase classification connector
- `pythia/acled_political.py` — ACLED event-level political data (protests, riots, strategic developments)
- `pythia/adversarial_check.py` — Counter-evidence searches for RC Level 1+ (devil's advocate)
- `horizon_scanner/reliefweb.py` — ReliefWeb humanitarian reports connector
- `forecaster/hazard_prompts.py` — Hazard-specific reasoning guidance for SPD prompts
- `scripts/ci/snapshot_prompt_artifact.py` — Prompt version snapshot script
- `pythia/tools/generate_calibration_advice.py` — Per-hazard/metric calibration advice generation

## Databases

Two DuckDB databases:

**Pythia DB** (`PYTHIA_DB_URL`): system of record
- `hs_runs`, `hs_triage` — Horizon Scanner outputs (triage scores, RC fields)
- `hs_hazard_tail_packs` — RC-triggered hazard evidence packs
- `hs_adversarial_checks` — Counter-evidence for RC Level 1+ (devil's advocate)
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
Separate from the Connector pipeline. Fetches NMME ensemble mean anomalies from CPC FTP, computes area-weighted country averages using xarray + regionmask, and writes to the `seasonal_forecasts` table in Pythia DB. Injected into HS triage/RC prompts via `horizon_scanner/seasonal_context.py` (`climate_data` kwarg) and into forecaster prompts via `research_json["nmme_seasonal_outlook"]`. Run: `python -m resolver.tools.ingest_nmme`. **Now runs as a step in the Ingest Structured Data workflow** (`ingest-structured-data.yml`) with `continue-on-error: true` since NMME FTP files are published ~9th-10th of each month. The standalone `ingest-nmme.yml` workflow is deprecated (schedule disabled, manual dispatch retained).

**Conflict forecast connectors** (`resolver/connectors/views.py`, `resolver/connectors/conflictforecast.py`, `resolver/connectors/acled_cast.py`):
Separate from the Connector pipeline (use `FORECAST_REGISTRY`, not `REGISTRY`). VIEWS connector fetches ML-based fatality predictions from the VIEWS API (`views_predicted_fatalities`, `views_p_gte25_brd`, leads 1–6). conflictforecast.org connector fetches news-based risk scores from Backendless API (`cf_armed_conflict_risk_3m`, `cf_armed_conflict_risk_12m`, `cf_violence_intensity_3m`). ACLED CAST connector fetches event-count forecasts via OAuth2 API (`cast_total_events`, `cast_battles_events`, `cast_erv_events`, `cast_vac_events`, 6-month lead), aggregated from admin1 to country level. All three write to `conflict_forecasts` table. `fetch_and_store` deduplicates before writing (keeps only the latest `forecast_issue_date` per source), and `_write_to_db` prunes old vintages (keeps only the 2 most recent issue dates per source). Loaded into ACE prompts via `horizon_scanner/conflict_forecasts.py`. Run: `python -m resolver.tools.fetch_conflict_forecasts`.

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
- **Levels** (env-overridable, likelihood-only thresholds): L0 (likelihood < 0.15), L1 (likelihood >= 0.15), L2 (likelihood >= 0.35), L3 (likelihood >= 0.55). Env vars: `PYTHIA_HS_RC_LEVEL1_LIKELIHOOD`, `PYTHIA_HS_RC_LEVEL2_LIKELIHOOD`, `PYTHIA_HS_RC_LEVEL3_LIKELIHOOD`.
- **Track assignment**: RC level > 0 → Track 1 (full ensemble), RC level 0 + priority tier → Track 2 (single Gemini Flash model), otherwise no SPD.
- L1+ triggers hazard tail pack generation and adversarial evidence checks. L2+ additionally forces `need_full_spd = TRUE`.
- **CRITICAL sync constraint**: The RC level threshold in `_select_tail_pack_hazards` (horizon_scanner.py) and the re-check threshold inside `adversarial_check.py` must match. If they drift, candidates will be passed to adversarial checks but silently rejected.
- Tail packs are enabled by default (`HS_TAIL_PACKS_ENABLED` defaults to `"1"`). Override via `PYTHIA_HS_HAZARD_TAIL_PACKS_ENABLED=0`.
- Distribution check warns when too many assessments exceed expected proportions
- RC prompt templates include a softened distribution anchor paragraph for small-country runs (where few comparative countries are included)
- **RC model**: Both passes default to `gemini-3-flash-preview` (overridable via `PYTHIA_RC_MODEL_PASS1` / `PYTHIA_RC_MODEL_PASS2`)
- **RC data sources for ACE**: Conflict forecasts (VIEWS, conflictforecast.org, ACLED CAST) and ICG CrisisWatch "On the Horizon" flags are injected into ACE RC prompts. CrisisWatch is fetched once per run in `main()` and explicitly threaded through `_run_hs_for_country` → `run_rc_for_country` → `_run_rc_for_single_hazard` (falls back to in-memory cache if not passed).
- **RC grounding queries**: Use hazard-specific search labels (e.g. "armed conflict escalation signals" for ACE, "flood risk river levels" for FL) instead of generic terms, to improve Gemini Google Search result relevance
- See `docs/hs_regime_change.md` for full details

## Track 1 / Track 2 forecast system

The forecaster routes questions into two tracks based on RC level:

- **Track 1** (full ensemble): Multi-model ensemble producing both `ensemble_bayesmc_v2` and `ensemble_mean_v2` aggregation rows. Used for RC level > 0 (higher-RC or higher-complexity questions). Higher cost.
- **Track 2** (single model): Single `track2_flash` model. Used for RC level 0 questions with priority tier. No ensemble aggregation — produces a single forecast row per question. Lower cost.

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
| `PYTHIA_LLM_PROFILE` | LLM profile: `test` or `prod` |
| `PYTHIA_LLM_CONCURRENCY` | Max concurrent LLM calls |
| `PYTHIA_API_TOKEN` | Admin API auth token |
| `PYTHIA_WEB_RESEARCH_ENABLED` | Enable shared retriever (0/1) |
| `HS_MAX_WORKERS` | HS concurrent country workers |
| `FORECASTER_RESEARCH_MAX_WORKERS` | Research phase concurrency |
| `FORECASTER_SPD_MAX_WORKERS` | SPD phase concurrency |
| `PYTHIA_HS_RC_LEVEL*_*` | RC threshold overrides |
| `PYTHIA_HS_RC_DIST_WARN_*` | RC distribution warning thresholds |
| `PYTHIA_HS_HAZARD_TAIL_PACKS_ENABLED` | Enable hazard tail packs (0/1, default 1) |
| `PYTHIA_HS_RESEARCH_WEB_SEARCH_ENABLED` | Enable web search for RC evidence packs (0/1, default 1) |
| `PYTHIA_PM_RETRIEVER_ENABLED` | Enable prediction market retriever (0/1, default 0 — see known failure modes) |
| `PYTHIA_PREDICTION_MARKETS_ENABLED` | Legacy alias; prefer `PYTHIA_PM_RETRIEVER_ENABLED` |
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
- **Question-level web research pipeline is deprecated**: The `fetch_evidence_pack` / `_build_question_evidence_queries` / `_merge_question_evidence_packs` flow is bypassed. SPD prompts now receive structured data directly via `_load_structured_data()`: conflict forecasts, ReliefWeb, HDX Signals, HS grounding evidence, ACAPS, IPC, ACLED political, NMME, ENSO, seasonal TC, adversarial checks. The `question_research` table is no longer populated by the pipeline (only placeholder rows). Env vars `PYTHIA_RETRIEVER_ENABLED`, `PYTHIA_WEB_RESEARCH_ENABLED`, `PYTHIA_SPD_WEB_SEARCH_ENABLED`, `PYTHIA_FORECASTER_SELF_SEARCH` are set to "0" in the workflow.
- HS pipeline is per-hazard: each hazard gets its own RC grounding, RC call, triage grounding, and triage call (2-pass each: Gemini Flash + GPT-5-mini)
- RC and triage grounding use different signal categories and recency windows (RC: TRIGGER/DAMPENER/BASELINE; triage: SITUATION/RESPONSE/FORECAST/VULNERABILITY)
- **Grounding recency windows**: RC grounding uses tight windows (ACE=14d, DR=30d, FL=14d, HW=14d, TC=30d) because it hunts for change signals. Triage grounding uses wider windows (ACE=60d, DR=90d, FL=60d, HW=60d, TC=60d) for the operational picture. Both are defined in their respective `RECENCY_DAYS` dicts.
- **Grounding source steering**: All 10 grounding prompts (5 RC + 5 triage) include a `PRIORITIZE THESE SOURCES` section with hazard-specific source priority lists and a `RECENCY FILTER` instruction. RC prompts prioritize wire services and specialist sources for novelty; triage prompts elevate OCHA/humanitarian sources for the operational picture.

## Known failure modes

- **RC degradation without data sources**: RC assessments degrade severely when both `PYTHIA_HS_RESEARCH_WEB_SEARCH_ENABLED=0` and structured data connectors (ACLED, facts_deltas) are unavailable simultaneously. All country-hazard pairs may return RC Level 0 (baseline), even for active crisis countries. Always ensure at least one of web search or structured data is available for RC grounding.
- **Prediction market retriever**: Currently disabled (`PYTHIA_PM_RETRIEVER_ENABLED=0`). Metaculus returns 403, Polymarket returns HTTP 422, and each call times out at 30s — adding ~6 minutes of wasted wall time per run. Do not re-enable until upstream APIs are fixed.
- **DuckDB cache key mismatch** (fixed): The DuckDB connection pool was using raw URL strings rather than resolved paths as cache keys, causing 100+ `cache_event=miss` entries and a new connection per DB access. Fixed by normalizing to resolved paths.
- **ACAPS iso3 list values** (fixed): The ACAPS API sometimes returns `iso3` as a list instead of a string. All bulk-fetch functions in `pythia/tools/ingest_structured_data.py` (`_bulk_fetch_inform_severity`, `_bulk_fetch_risk_radar`, `_bulk_fetch_daily_monitoring`, `_bulk_fetch_humanitarian_access`, `_bulk_fetch_reliefweb`) now defensively coerce list values to strings before calling `.upper()`.
- **Adversarial checks / tail packs silenced by PYTHIA_WEB_RESEARCH_ENABLED=0** (fixed): The deprecated `PYTHIA_WEB_RESEARCH_ENABLED=0` flag (which disables the old question-level web research pipeline) was also blocking `fetch_evidence_pack` calls in adversarial checks and hazard tail packs. Both now temporarily override the env var to `"1"` around their `fetch_evidence_pack` calls, so they work independently of the deprecated flag.
- **Kimi kimi-k2.5 temperature constraint** (fixed): The kimi-k2.5 model only accepts `temperature=1`. The `_call_openai_compatible` function in `forecaster/providers.py` now clamps temperature to 1.0 for models in `_KIMI_FIXED_TEMPERATURE_MODELS`.
- **ACLED CAST expired token passthrough** (fixed): `get_access_token()` in `resolver/ingestion/acled_auth.py` now validates JWT expiry via `_jwt_is_valid()` before using environment-provided tokens. Expired tokens fall through to the refresh/password grant flow instead of being returned as-is.
- **ACAPS Humanitarian Access field names** (fixed): The ACAPS Humanitarian Access API does not return an `iso3` field in its response records (unlike INFORM Severity, Risk Radar, etc.). The bulk fetcher in `pythia/tools/ingest_structured_data.py` tries multiple field names (`iso3`, `iso`, `country_iso3`, `country_code`, `country_iso`) and falls back to a nested `country.iso3` dict lookup. A sample-record-keys log line is emitted on first record to aid debugging if the schema changes again.

## Canonical DB artifact discovery

The `pythia-resolver-db` DuckDB artifact is shared across three workflows. Each workflow's "Download canonical resolver DB" step searches for the most recent successful run from **all three** workflow types:

1. **Horizon Scanner Triage** (`run_horizon_scanner.yml`) — DB_SOURCE=`pipeline`
2. **Resolver — Initial Backfill** — DB_SOURCE=`backfill`
3. **Ingest Structured Data** (`ingest-structured-data.yml`) — DB_SOURCE=`ingest`

Candidates are sorted by `createdAt` descending; the first one that downloads successfully and passes the DB signature check is used. If a new workflow is added that produces `pythia-resolver-db`, it must be added as a candidate source in all workflows that consume the artifact.

## Run health diagnostics

- `hs_country_evidence` and `question_evidence` CSVs are the primary artifacts for diagnosing structured data connector health post-run.
- If a connector shows "unavailable" across all countries, it indicates an upstream data gap, a connector bug, or a missing environment secret.
- After applying fixes, a clean re-run must produce a queryable DuckDB artifact before connector health can be verified.
