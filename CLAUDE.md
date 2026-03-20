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
  -> Structured data connectors (ReliefWeb, ACAPS, IPC, ACLED political, ENSO, seasonal TC, HDX Signals, ACLED CAST, GDACS)
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
- `resolver/connectors/__init__.py` — Connector registry (discover_connectors: ACLED, IDMC, IFRC Montandon, GDACS) + FORECAST_REGISTRY (VIEWS, CF.org, ACLED CAST)
- `resolver/connectors/acled_cast.py` — ACLED CAST connector (event-count forecasts by type: total/battles/ERV/VAC)
- `resolver/tools/run_pipeline.py` — Pipeline orchestrator (fetch -> validate -> enrich -> precedence -> deltas -> DuckDB)
- `resolver/tools/enrich.py` — Enrichment (registry lookups, ym derivation, defaults)
- `resolver/tools/precedence_config.yml` — Precedence tier policy
- `pythia/prediction_markets/retriever.py` — Prediction market signal retriever (currently disabled by default via `PYTHIA_PM_RETRIEVER_ENABLED=0`; Metaculus returns 403, Polymarket returns 422, each times out at 30s)
- `scripts/print_forecaster_ensemble.py` — Ensemble diagnostic script (must be invoked as `python -m scripts.print_forecaster_ensemble`, not directly)
- `pythia/acaps.py` — ACAPS unified connector (4 datasets: INFORM Severity, Risk Radar, Daily Monitoring, Humanitarian Access)
- `pythia/ipc_phases.py` — IPC food security phase classification connector
- `pythia/acled_political.py` — ACLED event-level political data (protests, riots, strategic developments); wired into bulk ingest via `_bulk_fetch_acled_political` in `ingest_structured_data.py`
- `pythia/adversarial_check.py` — Counter-evidence searches for RC Level 1+ (devil's advocate)
- `horizon_scanner/reliefweb.py` — ReliefWeb humanitarian reports connector
- `forecaster/hazard_prompts.py` — Hazard-specific reasoning guidance for SPD prompts
- `scripts/ci/snapshot_prompt_artifact.py` — Prompt version snapshot script
- `scripts/refresh_crisiswatch.py` — Playwright-based CrisisWatch scraper (monthly, writes `crisiswatch_latest.json`)
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
- `crisiswatch_entries` — ICG CrisisWatch monthly arrows + alerts (PK: iso3, year, month)
- `llm_calls` — Full telemetry (cost, tokens, latency, errors)

**Resolver DB** (`resolver/db/schema.sql`): fact ingestion
- `facts_resolved` — Precedence-resolved facts (unique on ym, iso3, hazard_code, metric, series_semantics)
- `facts_deltas` — Monthly flow changes
- `snapshots`, `manifests`, `meta_runs` — Pipeline metadata

## Resolver architecture

The Resolver was refactored in PR #610 to a connector-based architecture. Defunct legacy connectors (DTM, EM-DAT ingestion, HDX, IPC, ODP, ReliefWeb, UNHCR, WFP, WHO, WorldPop) were removed. The GDACS connector was re-implemented as a new Connector protocol source.

**Connector protocol** (`resolver/connectors/protocol.py`): Every data source implements a `Connector` protocol with a `name` attribute and a `fetch_and_normalize()` method that returns a DataFrame with exactly 21 canonical columns (event_id, iso3, hazard_code, metric, value, as_of_date, publisher, etc.).

**Active connectors** (`resolver/connectors/__init__.py` REGISTRY):
- `acled` — ACLED conflict/fatalities data (wraps `resolver/ingestion/acled_client`)
- `idmc` — IDMC internal displacement data (wraps `resolver/ingestion/idmc/`)
- `ifrc_montandon` — IFRC Go connector (stubbed, not yet active)
- `gdacs` — GDACS disaster population exposure (FL, DR, TC). No auth required. **Two data sources**: (1) static RSS feeds (`xml/rss_fl_3m.xml`, `xml/rss_tc_3m.xml`) for ≤3-month window (fast, has population data); (2) JSON search API (`gdacsapi/api/events/geteventlist/SEARCH`) + per-event RSS enrichment for >3-month backfill. The original `rss.aspx?profile=ARCHIVE` endpoint is broken (returns only a generic info item). DR RSS feed (`rss_dr_3m.xml`) returns 404. Depth controlled by `GDACS_MONTHS` (default 3; use 135 for full backfill to 2015). Multi-country events use population-weighted allocation. TC zero-fills no-event months; FL/DR do not. Entry point: `resolver/ingestion/gdacs.py` (for `run_connectors.py`); also integrated into `pythia/tools/ingest_structured_data.py`. Env vars: `GDACS_MONTHS` (default 3), `GDACS_REQUEST_DELAY` (default 1.0s), `GDACS_FORCE_RSS`/`GDACS_FORCE_JSON` (override auto-detection).

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
Separate from the Connector pipeline. Fetches NMME ensemble mean anomalies from CPC FTP, computes area-weighted country averages using xarray + regionmask (`countries_10` resolution for 170+ countries), and writes to the `seasonal_forecasts` table in Pythia DB. Injected into HS triage/RC prompts via `horizon_scanner/seasonal_context.py` (`climate_data` kwarg) and into forecaster prompts via `research_json["nmme_seasonal_outlook"]`. **Now integrated into `pythia/tools/ingest_structured_data.py`** as the `nmme` source (NMME failures are caught as warnings, non-fatal, since FTP files are published ~9th-10th of each month). The standalone `python -m resolver.tools.ingest_nmme` entry point still works independently. The standalone `ingest-nmme.yml` workflow is deprecated (schedule disabled, manual dispatch retained). Uses `decode_times=False` in `xr.open_dataset()` calls because the new CPC multi-lead files use `months since 1960-01-02 21:00:00` as time units, which xarray cannot decode.

**Conflict forecast connectors** (`resolver/connectors/views.py`, `resolver/connectors/conflictforecast.py`, `resolver/connectors/acled_cast.py`):
Separate from the Connector pipeline (use `FORECAST_REGISTRY`, not `REGISTRY`). VIEWS connector fetches ML-based fatality predictions from the VIEWS API (`views_predicted_fatalities`, `views_p_gte25_brd`, leads 1–6). conflictforecast.org connector fetches news-based risk scores from Backendless API (`cf_armed_conflict_risk_3m`, `cf_armed_conflict_risk_12m`, `cf_violence_intensity_3m`). ACLED CAST connector fetches event-count forecasts via OAuth2 API (`cast_total_events`, `cast_battles_events`, `cast_erv_events`, `cast_vac_events`, 6-month lead), aggregated from admin1 to country level. All three write to `conflict_forecasts` table. `fetch_and_store` deduplicates before writing (keeps only the latest `forecast_issue_date` per source), and `_write_to_db` prunes old vintages (keeps only the 2 most recent issue dates per source). Loaded into ACE prompts via `horizon_scanner/conflict_forecasts.py`. The conflictforecast.org connector uses suffix-priority column selection: when multiple columns match a metric pattern (e.g. `ons_armedconf_03`), it prefers `_all` (combined forecast) over `_text`/`_hist` sub-models, and explicitly skips `_target` (ground truth, NaN for future) and `_naive` (baseline). Also includes a Backendless metadata skip set and a median-value sanity check (onset metrics only; intensity metrics use log-scale values). **Now integrated into `pythia/tools/ingest_structured_data.py`** as sources `views`, `conflictforecast`, `acledcast` (with `conflict` as a convenience alias for all three). The standalone `python -m resolver.tools.fetch_conflict_forecasts` entry point still works independently.

**ENSO state and forecast** (`horizon_scanner/enso/enso_module.py`):
Scrapes IRI/CPC ENSO Quick Look page for current ENSO state, Niño 3.4 anomaly, 9-season probabilistic forecast, multi-model plume averages, and IOD state. Cached as JSON with 7-day expiry. Injected into RC and triage prompts for DR, FL, HW, TC hazards via `get_enso_prompt_context()`. Refreshed by `.github/workflows/refresh-enso.yml`.

**Seasonal TC forecasts** (`horizon_scanner/seasonal_tc/`):
Aggregates basin-level seasonal TC forecasts from TSR (PDF extraction), NOAA CPC (press release scraping), and BoM (outlook scraping) across 8 basins. Country-to-basin mapping in `__init__.py`. Cached as JSON; `get_seasonal_tc_context_for_country(iso3)` returns prompt-ready text. Refreshed by `.github/workflows/refresh-seasonal-tc.yml`.

**HDX Signals** (`horizon_scanner/hdx_signals.py`):
Downloads OCHA's HDX Signals CSV from CKAN API. Indicator-to-hazard mapping (acled_conflict→ACE, ipc_food_insecurity→DR, etc.). Filtered by country, hazard, recency (180 days). Injected into RC and triage prompts for all hazards via `format_hdx_signals_for_prompt()`.

**ICG CrisisWatch** (`horizon_scanner/crisiswatch.py` + `scripts/refresh_crisiswatch.py`):
Monthly fetch of ICG CrisisWatch data. **Primary source**: local JSON file (`horizon_scanner/data/crisiswatch_latest.json`) produced by the Playwright scraper (`scripts/refresh_crisiswatch.py`), which runs monthly via `refresh-crisiswatch.yml` workflow, loading `https://www.crisisgroup.org/crisiswatch` in headless Chromium, parsing with BeautifulSoup, and committing structured JSON. The `/crisiswatch/print` endpoint is broken (stale Oct 2019 data); the main page renders correctly. **Fallback**: Gemini grounding calls (`_call_gemini_grounding` in `crisiswatch.py`) — only attempted when the JSON file is missing or empty. Called once per HS run, cached in-memory, persisted to `crisiswatch_entries` DuckDB table. Injected into ACE RC prompts (via `crisiswatch_context` + deprecated `icg_on_the_horizon`), ACE triage prompts, and ACE SPD prompts. Country name → ISO3 mapping via hardcoded `_ICG_COUNTRY_ISO3` dict. JSON staleness: if `fetched_at` is >45 days old, a warning is logged but data is still used.

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
| `PYTHIA_GOOGLE_SPD_TIMEOUT_FLASH_SEC` | Gemini Flash SPD timeout (default 300s) |
| `PYTHIA_GOOGLE_SPD_TIMEOUT_PRO_SEC` | Gemini Pro SPD timeout (default 300s) |
| `ACAPS_EMAIL` / `ACAPS_PASSWORD` | ACAPS API credentials |
| `GDACS_MONTHS` | Number of months of GDACS history to fetch (default 3; 135 for full backfill to 2015) |
| `GDACS_REQUEST_DELAY` | Seconds between GDACS API requests (default 1.0) |
| `GDACS_FORCE_RSS` | Force static RSS feed path ("1" to enable; overrides auto-detection) |
| `GDACS_FORCE_JSON` | Force JSON search API path ("1" to enable; overrides auto-detection) |

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

# Ingest ALL structured data (conflict forecasts, ACAPS, IPC, ReliefWeb, NMME)
python3 -m pythia.tools.ingest_structured_data
# Filter by source: python3 -m pythia.tools.ingest_structured_data --sources conflict acaps ipc
# Valid sources: views, conflictforecast, acledcast, acaps_inform_severity,
#   acaps_risk_radar, acaps_daily_monitoring, acaps_humanitarian_access, ipc, reliefweb, acled_political, nmme
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
- **IDMC double-run in backfill workflow** (fixed): IDMC ran twice in `resolver-initial-backfill.yml` — once in the dedicated HELIX API step and again inside `run_connectors.py`. Fixed by adding `RESOLVER_SKIP_IDMC: "1"` to the "Run connectors" step env block.
- **Calibration advice dual-constraint failures** (fixed): `_upsert_advice()` in `pythia/tools/generate_calibration_advice.py` failed repeatedly due to the `calibration_advice` table having both the original 3-column PK `(as_of_month, hazard_code, metric)` and the newer 4-column unique index `ux_calibration_advice (as_of_month, hazard_code, metric, model_name)`. Root cause: the `CREATE TABLE` DDL in `schema.py` defined the 3-column PK, and **DuckDB does not support `ALTER TABLE DROP CONSTRAINT` for PRIMARY KEY constraints** — all `DROP CONSTRAINT` attempts failed silently regardless of constraint name. Fixed by **recreating the table**: `_migrate_calibration_advice_pk()` checks `information_schema.table_constraints` for a PK, and if found, uses CTAS to copy data into a temp table, `DROP TABLE` on the original (which cascades all dependencies including the PK — `ALTER TABLE RENAME` is also blocked by `DependencyException`), creates a new table without a PK, copies data back with column-safe SELECT (handling missing columns and NULL model_name), drops the temp table, and creates the 4-column unique index. The `CREATE TABLE` DDL in `schema.py` was also changed to `PRIMARY KEY (as_of_month, hazard_code, metric, model_name)` so fresh DBs never get the old PK. The upsert uses `INSERT INTO ... ON CONFLICT ... DO UPDATE SET`, and `_seed_default_advice` uses `ON CONFLICT ... DO NOTHING`.
- **Calibration advice hardcoded bucket thresholds** (fixed): `_bucket_case_sql()` in `generate_calibration_advice.py` hardcoded threshold values (e.g. `10000, 50000, 250000, 500000` for PA) that duplicated the canonical `PA_THRESHOLDS`/`FATAL_THRESHOLDS` lists from `compute_scores.py`. The imported constants `_bucket_index`, `PA_THRESHOLDS`, `FATAL_THRESHOLDS` were unused. Fixed by deriving the SQL `CASE` expression dynamically from the canonical threshold lists. Removed unused `_bucket_index` import.
- **Calibration advice silent exception swallowing** (fixed): Multiple `except Exception: pass` blocks in `generate_calibration_advice.py` (`_table_exists`, `_row_count`, `_ensure_findings_json_column`, `_migrate_calibration_advice_pk`) silently swallowed errors, making real DB connection failures or schema issues look like empty data. Added `LOGGER.warning` / `LOGGER.debug` to all critical silent exception paths.
- **Calibration workflow manual dispatch broken** (fixed): `compute_calibration_pythia.yml` used `github.event.workflow_run.id` to download the DB artifact, which is empty on `workflow_dispatch` (manual trigger). Replaced with a two-path strategy: (A) if triggered by `workflow_run`, download from the triggering run; (B) on manual dispatch or fallback, use canonical DB discovery (searches recent successful runs from Compute SPD Scores, Compute Calibration, Horizon Scanner Triage, and Ingest Structured Data workflows).
- **Vestigial "Export Facts: facts.csv rows: 0" in summary** (fixed): `summarize_connectors.py` always rendered the "## Export Facts" section because `_collect_export_summary()` was gutted in PR #610 (always returns empty dict with rows=0), but the condition `if export_info or export_error or mapping_debug_records` was always truthy. Fixed by gating on `export_error or has_export_rows` (where `has_export_rows = export_info.get("rows", 0) > 0`). The `mapping_debug_records` section renders independently.
- **Question overwrite bug destroying run provenance** (fixed): `scripts/create_questions_from_triage.py` used stable question_ids (e.g. `ETH_ACE_FATALITIES`) with a DELETE+INSERT pattern in `_upsert_question`. Each HS run destroyed the previous question's `hs_run_id`, `window_start_date`, and `target_month`. After the March 2026 run, all 524 questions pointed to the March HS run with `window_start_date=2026-04-01`, making December/January forecasts unresolvable while March forecasts were incorrectly scored. Additionally, `_llm_derived_window_starts` in `compute_resolutions.py` used `MIN(timestamp)` from `llm_calls` across all runs for shared question_ids, causing cross-run contamination. Fixed by: (1) making question_ids epoch-specific with `_{epoch_label}` suffix (e.g. `ETH_ACE_FATALITIES_2026-04`), (2) replacing DELETE+INSERT with INSERT-only (skip if exists), (3) fixing `target_month` to represent the 6th horizon month instead of the opening month, (4) removing the LLM-derived window override from `compute_resolutions.py` so the question's own `window_start_date` is authoritative. Recovery script: `scripts/recover_historical_questions.py`.

## Canonical DB artifact discovery

The `pythia-resolver-db` DuckDB artifact is shared across multiple workflows. Each workflow's "Download canonical resolver DB" step searches for the most recent successful run from candidate workflow types:

1. **Horizon Scanner Triage** (`run_horizon_scanner.yml`) — DB_SOURCE=`pipeline`
2. **Resolver — Initial Backfill** — DB_SOURCE=`backfill`
3. **Ingest Structured Data** (`ingest-structured-data.yml`) — DB_SOURCE=`ingest`
4. **Compute SPD Scores** (`compute_scores.yml`) — upstream of calibration
5. **Compute Calibration Weights & Advice** (`compute_calibration_pythia.yml`) — calibration pipeline

Candidates are sorted by `createdAt` descending; the first one that downloads successfully and passes the DB signature check is used. If a new workflow is added that produces `pythia-resolver-db`, it must be added as a candidate source in all workflows that consume the artifact. The `compute_scores.yml` and `compute_calibration_pythia.yml` workflows use the triggering `workflow_run.id` as the primary source, falling back to canonical discovery on manual `workflow_dispatch`.

## Run health diagnostics

- `hs_country_evidence` and `question_evidence` CSVs are the primary artifacts for diagnosing structured data connector health post-run.
- If a connector shows "unavailable" across all countries, it indicates an upstream data gap, a connector bug, or a missing environment secret.
- After applying fixes, a clean re-run must produce a queryable DuckDB artifact before connector health can be verified.
- The `inspect_resolver_duckdb.yml` workflow includes 7 data quality checks: conflict forecast value range validation (warns if probability values > 10), seasonal forecast country count, empty connector table warnings, IDMC/IDU hazard code consistency, conflict forecast staleness (> 45 days), HDX Signals note (cached as CSV, not in DB), and per-country conflict forecast sampling (IRN, SOM, ETH, SDN, UKR).
- **CrisisWatch diagnostics** in the debug bundle (`scripts/dump_pythia_debug_bundle.py`): (1) Traffic-light health check in `_evaluate_pipeline_health` — reports OK/WARN/FAIL with country counts, arrow breakdown, and alerts; WARN if 0 entries, <10 countries, or no arrow data; (2) Per-country `crisiswatch_arrow` column in the data inject inventory CSV showing arrow direction and alert type; (3) `crisiswatch_health` section in health report JSON with arrow counts, alert counts, notable (deteriorated/alert) entries, and countries missing CrisisWatch data.

## Structured data bulk ingest

`pythia/tools/ingest_structured_data.py` orchestrates bulk fetch/store for all structured data connectors. The `_SOURCE_GROUPS` dict maps group names to table names. Currently registered groups: `views`, `conflictforecast`, `acledcast`, `acaps_inform_severity`, `acaps_risk_radar`, `acaps_daily_monitoring`, `acaps_humanitarian_access`, `ipc`, `reliefweb`, `acled_political`, `nmme`, `gdacs`. Each group has a `_bulk_fetch_*` function that parallelizes API calls across countries via `ThreadPoolExecutor`. The `acled_political` group fetches event-level political data (protests, riots, strategic developments) from the ACLED API via `pythia.acled_political.fetch_acled_political_events` and stores via `store_acled_political_events`. The `gdacs` group delegates to `resolver.tools.run_pipeline` with `--connectors gdacs`, which writes to `facts_resolved` and `facts_deltas` via the standard Resolver pipeline. GDACS is a self-storing source (no per-country store function). The `ingest-structured-data.yml` workflow runs weekly (Sunday 03:00 UTC) in incremental mode (last 3 months) and monthly (28th) with all sources; GDACS backfill depth is controlled via the `gdacs_months` workflow input (default 3; set to 135 for full backfill to 2015).
