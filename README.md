# Pythia

Pythia is an end-to-end AI forecasting system for humanitarian questions. It scans countries for emerging hazards, turns triage into forecastable questions, runs an LLM ensemble to produce Subjective Probability Distributions (SPD), and stores forecasts, research, and diagnostics in a DuckDB-backed system of record.

## Table of contents
- [What is Pythia?](#what-is-pythia)
- [System at a glance](#system-at-a-glance)
- [Core components](#core-components)
- [Regime Change (RC): out-of-pattern detection](#regime-change-rc-out-of-pattern-detection)
- [Hazard Tail Packs (HTP): hazard-specific trigger evidence](#hazard-tail-packs-htp-hazard-specific-trigger-evidence)
- [Retriever web research (shared evidence packs)](#retriever-web-research-shared-evidence-packs)
- [Data model / DuckDB tables](#data-model--duckdb-tables)
- [Model management](#model-management)
- [Quickstart (local)](#quickstart-local)
- [Running in GitHub Actions](#running-in-github-actions)
- [Dashboard](#dashboard)
- [Downloads / exports](#downloads--exports)
- [Troubleshooting](#troubleshooting)
- [Cross-links](#cross-links)

## What is Pythia?
Pythia connects historical hazard data, web research, and LLM reasoning into a repeatable pipeline for early-warning forecasts. It produces triage signals, structured research briefs, and probabilistic forecasts across hazards, then serves those outputs through a DuckDB system of record, a FastAPI service, and a Next.js dashboard.

## System at a glance

```
Resolver facts/base rates → Horizon Scanner per-hazard pipeline:
  Per hazard: RC grounding → RC assessment (2-pass) → triage grounding → triage (2-pass)
  + Structured data: ReliefWeb, ACAPS, IPC, ACLED political, ENSO, seasonal TC, HDX Signals
  + Conflict forecasts: VIEWS, conflictforecast.org, ACLED CAST
  + Adversarial checks (RC L1+) → Calibration advice
  → Track 1: full ensemble SPD + scenarios
  → Track 2: single-model SPD
  → DuckDB → API → Dashboard + downloads
```

- **System of record**: DuckDB (`PYTHIA_DB_URL` / `app.db_url`) holds HS triage, questions, structured data, forecasts, and diagnostics.
- **End-to-end flow**: Resolver facts and base rates inform the HS per-hazard pipeline. Each hazard gets its own RC assessment (with dedicated grounding) and triage assessment (with separate grounding), both running 2-pass (Gemini Flash + GPT-5-mini). Structured data connectors pull evidence from humanitarian, climate, and conflict-forecast APIs including ENSO state, seasonal TC forecasts, HDX Signals, and ACLED CAST. Adversarial checks run for RC L2+ cases. Forecaster routes questions by track — Track 1 (RC-elevated) through the full ensemble, Track 2 (priority, no RC) through a single model — and writes results back to DuckDB.

## Core components
- **Resolver (facts + base rates)**: Resolver tables in DuckDB (`facts_resolved`, `facts_deltas`, `snapshots`) provide historical context. Schema is defined in [`pythia/db/schema.py`](pythia/db/schema.py).
- **Horizon Scanner (HS)**: `python -m horizon_scanner.horizon_scanner` triages countries and hazards, writes `hs_runs`/`hs_triage`, seeds `questions`/`question_research`, and generates country evidence packs. HS runs a fully per-hazard pipeline: each hazard (ACE, DR, FL, HW, TC) gets its own RC grounding call, RC assessment, triage grounding call, and triage assessment. RC and triage each run 2-pass for stability (RC: both Gemini Flash; triage: Gemini Pro + Gemini Flash). RC grounding uses TRIGGER/DAMPENER/BASELINE signal categories; triage grounding uses SITUATION/RESPONSE/FORECAST/VULNERABILITY categories with longer recency windows. An ACLED low-activity filter skips ACE triage for countries with minimal recent conflict (0 fatalities in 2 months AND <25 in 12 months).
- **Retriever web research (shared evidence packs)**: When enabled, the retriever builds evidence packs reused across HS, research, and SPD prompts. The shared retriever defaults to `gemini-3-flash-preview` when `PYTHIA_RETRIEVER_ENABLED=1` and `PYTHIA_RETRIEVER_MODEL_ID` is unset; see [`pythia/web_research/web_research.py`](pythia/web_research/web_research.py).
- **Structured data connectors**: Authoritative humanitarian, climate, and conflict-forecast data pulled from specialist APIs and stored in DuckDB for direct prompt injection. Replaced the former Research LLM stage. See [`pythia/acaps.py`](pythia/acaps.py), [`pythia/fewsnet_food_security.py`](pythia/fewsnet_food_security.py), [`horizon_scanner/reliefweb.py`](horizon_scanner/reliefweb.py), [`pythia/acled_political.py`](pythia/acled_political.py), [`horizon_scanner/enso/`](horizon_scanner/enso/), [`horizon_scanner/seasonal_tc/`](horizon_scanner/seasonal_tc/), [`horizon_scanner/hdx_signals.py`](horizon_scanner/hdx_signals.py), [`resolver/connectors/acled_cast.py`](resolver/connectors/acled_cast.py).
- **Adversarial evidence checks**: Counter-evidence searches for RC Level 1+ cases, stored in `hs_adversarial_checks`. See [`pythia/adversarial_check.py`](pythia/adversarial_check.py).
- **Hazard-specific reasoning**: Per-hazard forecasting instructions injected into SPD prompts. See [`forecaster/hazard_prompts.py`](forecaster/hazard_prompts.py).
- **Calibration advice**: Per-hazard/metric calibration guidance generated from historical performance, injected into forecasting prompts. See [`pythia/tools/generate_calibration_advice.py`](pythia/tools/generate_calibration_advice.py).
- **Forecaster SPD v2 ensemble**: `python -m forecaster.cli --mode pythia` runs SPD v2 prompts across the active ensemble and writes per-model outputs (`forecasts_raw`) and aggregated results (`forecasts_ensemble`). Questions are routed by track: **Track 1** (RC-elevated, level > 0) uses the full multi-model ensemble; **Track 2** (priority, no RC) uses a lightweight single-model path (Gemini Flash).
- **Scenarios (Track 1 only)**: When an ensemble SPD is available, scenarios can be generated for Track 1 questions and written back to DuckDB.
- **Dashboard + API**: FastAPI serves `/v1/*` endpoints for the dashboard, downloads, and diagnostics. The Next.js UI lives in `web/`.

## Regime Change (RC): out-of-pattern detection

**Definition:** Regime Change (RC) is the likelihood of a generating-process shift within the next 1–6 months that makes historical base rates less reliable. RC is assessed per-hazard in a dedicated LLM pipeline step that runs before triage. Each hazard type gets its own RC prompt with calibration anchors (default likelihood 0.05, expected ~80% at ≤0.10), dedicated grounding call, and hazard-specific data injection (ENSO context for climate hazards, conflict forecasts for ACE, seasonal TC for TC). RC runs 2-pass (both Gemini Flash by default) and writes to `hs_triage`.

**Stored fields in `hs_triage`:**
- `regime_change_likelihood` (probability)
- `regime_change_direction` (`up`, `down`, `mixed`, `unclear`)
- `regime_change_magnitude`
- `regime_change_score` (computed)
- `regime_change_level` (severity)
- `regime_change_window`
- `regime_change_json` (full normalized object)

**Scoring + levels (defaults; env-overridable via `PYTHIA_HS_RC_LEVEL*_LIKELIHOOD`):**
- `score = likelihood × magnitude`
- Level 0: `likelihood < 0.15`
- Level 1: `likelihood ≥ 0.15`
- Level 2: `likelihood ≥ 0.35`
- Level 3: `likelihood ≥ 0.55`

**Behavioral implications:**
- **Track routing**: Questions with RC level > 0 are assigned Track 1 (full multi-model ensemble); priority questions without RC are Track 2 (lightweight single-model).
- **`need_full_spd` override**: HS forces `need_full_spd = TRUE` when RC is elevated (Level ≥2) even if the triage tier is quiet.
- **Research v2**: RC is surfaced in the research prompt, and RC-elevated hazards require at least one `regime_shift_signals` entry (or a rebuttal).
- **SPD v2**: RC guidance is embedded in the SPD prompt, and the model must include a sentence starting with `RC:` in `human_explanation` describing how RC affected the SPD.

See [docs/hs_regime_change.md](docs/hs_regime_change.md) for full details and env overrides.

## Hazard Tail Packs (HTP): hazard-specific trigger evidence

Hazard Tail Packs are hazard-scoped evidence bundles generated by HS when RC is elevated. They provide targeted trigger/counter-trigger evidence for downstream Research and SPD.

**Trigger rules:**
- Generated only for **RC Level ≥1** hazards.
- Limited to **max 2 hazards per country** per HS run (`PYTHIA_HS_HAZARD_TAIL_PACKS_MAX_PER_COUNTRY`, default `2`).
- Tail packs are **off by default** unless `PYTHIA_HS_HAZARD_TAIL_PACKS_ENABLED=1`.

**Storage table:** `hs_hazard_tail_packs` (one row per `hs_run_id × iso3 × hazard_code`), including:
- `rc_level`, `rc_score`, `rc_direction`, `rc_window`
- `query`, `report_markdown`, `sources_json`, `grounded`, `grounding_debug_json`
- `structural_context`, `recent_signals_json`, `created_at`

**Output format:**
- `recent_signals` bullets are emitted as `TRIGGER | <window> | <direction> | <signal>`, `DAMPENER | ...`, `BASELINE | ...`.

**Caching behavior:**
- Tail packs are reused per `(hs_run_id, iso3, hazard_code)`; HS skips regeneration if the row already exists.

**Phase A / Phase B behavior:**
- **Phase A (Research)**: When present, the tail pack is injected into Research v2 evidence (`hs_hazard_tail_pack`).
- **Phase B (SPD v2)**: For RC-elevated hazards, the tail pack is injected into the SPD prompt. Signals are capped at **12 bullets** (`PYTHIA_SPD_TAIL_PACKS_MAX_SIGNALS`, default `12`).

**Relevant env vars (defaults):**
- `PYTHIA_HS_HAZARD_TAIL_PACKS_ENABLED` (`0`)
- `PYTHIA_HS_HAZARD_TAIL_PACKS_MAX_PER_COUNTRY` (`2`)
- `PYTHIA_HS_HAZARD_TAIL_PACKS_MAX_SOURCES` (defaults to `PYTHIA_HS_EVIDENCE_MAX_SOURCES`)
- `PYTHIA_HS_HAZARD_TAIL_PACKS_MAX_SIGNALS` (defaults to `PYTHIA_HS_EVIDENCE_MAX_SIGNALS`)
- `PYTHIA_SPD_TAIL_PACKS_MAX_SIGNALS` (`12`)

See [docs/hs_hazard_tail_packs.md](docs/hs_hazard_tail_packs.md) for full details.

## Retriever web research (deprecated)
The question-level web research pipeline (`fetch_evidence_pack` / `_build_question_evidence_queries`) is deprecated. SPD prompts now receive structured data directly via `_load_structured_data()`, including conflict forecasts, ReliefWeb, HDX Signals, ACAPS, ACLED political, NMME, ENSO, seasonal TC, GDACS event history, and adversarial checks. The `question_research` table is no longer populated by the pipeline (only placeholder rows). Env vars `PYTHIA_RETRIEVER_ENABLED`, `PYTHIA_WEB_RESEARCH_ENABLED`, `PYTHIA_SPD_WEB_SEARCH_ENABLED`, `PYTHIA_FORECASTER_SELF_SEARCH` are set to "0" in the workflow.

Grounding for RC and triage uses OpenAI GPT-4.1-mini web search (primary) with Gemini Google Search as fallback. Override with `PYTHIA_GROUNDING_PRIMARY_BACKEND=gemini`.

## NMME seasonal climate forecasts

Pythia ingests NMME (North American Multi-Model Ensemble) seasonal temperature and precipitation anomaly forecasts from the CPC FTP server. These provide structured climate context for drought, flood, and tropical cyclone assessments.

- **Source**: `ftp://ftp.cpc.ncep.noaa.gov/NMME/realtime_anom/ENSMEAN/` — ensemble mean anomalies at 1° resolution, updated ~9th of each month with 7 lead months.
- **Processing**: Country-level area-weighted averages using `xarray` + `regionmask` (Natural Earth admin-0 boundaries). Anomalies expressed in σ (standard deviations from climatology) with derived tercile categories (above/below/near normal).
- **Storage**: `seasonal_forecasts` table in Pythia DuckDB (~2,700 rows per monthly update: 195 countries × 2 variables × 7 leads).
- **Injection**: Automatically loaded into HS triage and RC prompts via the existing `climate_data` parameter for DR, FL, TC hazards. Also injected into forecaster research and SPD prompts via `research_json`.
- **Run manually**: `python -m resolver.tools.ingest_nmme` (or `--year-month YYYYMM` for a specific month, `--dry-run` to preview).
- **Automation**: `.github/workflows/ingest-nmme.yml` runs on the 10th of each month.

## Conflict forecasts

Pythia ingests external conflict forecasts from three independent quantitative sources and incorporates qualitative expert assessments from a fourth, providing forward-looking signals for the Armed Conflict (ACE) hazard.

### Quantitative sources (stored in DuckDB)

- **VIEWS (Uppsala/PRIO)**: ML-based country-month state-based conflict forecasts from the [VIEWS Forecasting](https://viewsforecasting.org/) project. Provides predicted fatalities (`views_predicted_fatalities`) and probability of ≥25 battle-related deaths (`views_p_gte25_brd`) at 1–6 month lead times. Good at trends and baseline levels; weaker at sudden onset.
- **conflictforecast.org (Mueller/Rauh)**: News-based armed conflict risk scores from the [Conflict Forecast](https://conflictforecast.org/) project. Provides armed conflict risk at 3-month and 12-month horizons (`cf_armed_conflict_risk_3m`, `cf_armed_conflict_risk_12m`) and violence intensity outlook (`cf_violence_intensity_3m`). Better at detecting shifts and escalation signals from media coverage patterns.
- **ACLED CAST (Conflict Alert System Tool)**: Event-count forecasts from ACLED's CAST API, disaggregated by event type: total events (`cast_total_events`), battles (`cast_battles_events`), explosions/remote violence (`cast_erv_events`), and violence against civilians (`cast_vac_events`) at 6-month lead. CAST forecasts event counts (not fatalities) and the event-type breakdown reveals shifts in the character of violence that aggregate measures would miss. See [`resolver/connectors/acled_cast.py`](resolver/connectors/acled_cast.py).

### Qualitative source (prompt-time web research)

- **ICG CrisisWatch**: International Crisis Group's monthly [CrisisWatch](https://www.crisisgroup.org/crisiswatch) bulletin. Per-country directional assessments (Deteriorated/Improved/Unchanged) and forward-looking "On the Horizon" flags (~3 conflict risks + ~1 resolution opportunity per month). Primary data source: Playwright-based scraper (`scripts/refresh_crisiswatch.py`) runs monthly via `refresh-crisiswatch.yml`, loading the main CrisisWatch page in headless Chromium and parsing with BeautifulSoup. Secondary: Gemini grounding calls during HS runs. Data injected into RC and triage evidence for ACE hazards.

### Storage and injection

- **Table**: `conflict_forecasts` in Pythia DuckDB (keyed on source, iso3, hazard_code, metric, lead_months, forecast_issue_date).
- **HS integration**: Automatically loaded into RC and triage prompts for ACE hazards via `horizon_scanner/conflict_forecasts.py`. Includes staleness warnings for data >45 days old. CAST section notes that it measures event counts rather than fatalities.
- **Forecaster integration**: Injected into research prompts for ACE questions as structured quantitative anchors.
- **Run manually**: `python -m resolver.tools.fetch_conflict_forecasts` (or `--sources views conflictforecast_org acled_cast`, `--dry-run` to preview).

## Structured data connectors

Pythia pulls structured humanitarian, climate, and conflict-forecast data from authoritative APIs, stores it in DuckDB, and injects it into pipeline prompts (HS triage, RC assessment, and/or forecaster SPD). This replaced the former Research LLM stage with deterministic, reproducible evidence injection.

### Data sources

| Source | Module | What it provides | DuckDB table(s) |
| --- | --- | --- | --- |
| **ReliefWeb** | `horizon_scanner/reliefweb.py` | Humanitarian situation reports (45-day window, up to 15/country) | `reliefweb_reports` |
| **ACAPS INFORM Severity** | `pythia/acaps.py` | Crisis severity scores + trend | `acaps_inform_severity`, `acaps_inform_severity_trend` |
| **ACAPS Risk Radar** | `pythia/acaps.py` | Forward-looking risk with triggers | `acaps_risk_radar` |
| **ACAPS Daily Monitoring** | `pythia/acaps.py` | Analyst-curated daily updates | `acaps_daily_monitoring` |
| **ACAPS Humanitarian Access** | `pythia/acaps.py` | Access constraint scores (HS triage only) | `acaps_humanitarian_access` |
| **IPC Phases** | `pythia/ipc_phases.py` | Food security phase populations (Phase 3+ = Crisis) | `ipc_phases` |
| **ACLED Political Events** | `pythia/acled_political.py` | Event-level political data (ACE/DI hazards only) | `acled_political_events` |
| **NMME Seasonal Forecasts** | `resolver/tools/ingest_nmme.py` | Temp/precip anomalies, 7-month lead (DR/FL/TC) | `seasonal_forecasts` |
| **ENSO State/Forecast** | `horizon_scanner/enso/enso_module.py` | ENSO state, Niño 3.4 anomaly, 9-season probabilistic outlook, IOD (DR/FL/TC) | `enso_state` (DB-first, falls back to live scrape) |
| **Seasonal TC Forecasts** | `horizon_scanner/seasonal_tc/` | Basin-level TC activity forecasts from TSR, NOAA CPC, BoM across 8 basins (TC only) | `seasonal_tc_outlooks`, `seasonal_tc_context_cache` (DB-first) |
| **HDX Signals** | `horizon_scanner/hdx_signals.py` | OCHA automated crisis alerts: conflict, displacement, food insecurity, agricultural stress (all hazards) | `hdx_signals` (DB-first, falls back to CSV) |
| **VIEWS** | `resolver/connectors/views.py` | ML-based conflict fatality predictions (ACE, 1–6 month leads) | `conflict_forecasts` |
| **conflictforecast.org** | `resolver/connectors/conflictforecast.py` | News-based conflict risk scores (ACE, 3m/12m) | `conflict_forecasts` |
| **ACLED CAST** | `resolver/connectors/acled_cast.py` | Event-count forecasts by type: total/battles/ERV/VAC (ACE, 6-month lead) | `conflict_forecasts` |
| **FEWS NET IPC** | `resolver/connectors/fewsnet_ipc.py` | Phase 3+ population estimates (DR hazard; Current Situation + Most Likely) | `facts_resolved` (via Resolver pipeline) |
| **GDACS** | `resolver/connectors/gdacs.py` | Disaster population exposure + event occurrence (FL/DR/TC) | `facts_resolved` (via Resolver pipeline) |
| **ICG CrisisWatch** | `horizon_scanner/crisiswatch.py` + `scripts/refresh_crisiswatch.py` | Expert conflict arrows + "On the Horizon" flags (ACE RC + triage + SPD). Playwright scraper (monthly) + Gemini grounding (runtime fallback) | `crisiswatch_entries` |

### Adversarial evidence checks

For RC Level 1+ cases, `pythia/adversarial_check.py` runs counter-evidence web searches and synthesizes results into structured output (counter-evidence, historical analogs, stabilizing factors, net assessment). Stored in `hs_adversarial_checks`.

### Calibration advice

`pythia/tools/generate_calibration_advice.py` generates per-hazard/metric calibration guidance from historical scores. Stored in `calibration_advice` and injected into forecasting prompts.

### Env vars

- `ACAPS_EMAIL`, `ACAPS_PASSWORD`: ACAPS API credentials (required for ACAPS feeds).
- `PYTHIA_DB_URL`: DuckDB path (shared with all connectors).

## Data model / DuckDB tables

Key tables (see [`pythia/db/schema.py`](pythia/db/schema.py) and [`SCHEMAS.md`](SCHEMAS.md) for canonical fields):

- **HS**: `hs_runs`, `hs_triage` (includes RC columns), `hs_country_reports`, `hs_hazard_tail_packs`, `hs_adversarial_checks`
- **Seasonal climate**: `seasonal_forecasts` (NMME country-level temp/precip anomalies)
- **Conflict forecasts**: `conflict_forecasts` (VIEWS + conflictforecast.org + ACLED CAST predicted fatalities, risk scores, and event counts)
- **Structured data**: `reliefweb_reports`, `acled_political_events`, `ipc_phases`, `acaps_inform_severity`, `acaps_inform_severity_trend`, `acaps_risk_radar`, `acaps_daily_monitoring`, `acaps_humanitarian_access`
- **Context sources**: `enso_state`, `seasonal_tc_outlooks`, `seasonal_tc_context_cache`, `hdx_signals`, `crisiswatch_entries`
- **Questions + research**: `questions`, `question_research`
- **Forecasts**: `forecasts_raw`, `forecasts_ensemble` (both include `reasoning_trace_json`)
- **Resolutions + scoring**: `resolutions`, `scores`, `eiv_scores`
- **Calibration**: `calibration_weights`, `calibration_advice`, `bucket_centroids`, `bucket_definitions`
- **Diagnostics**: `llm_calls`, `question_run_metrics`
- **Resolver**: `facts_resolved`, `facts_deltas`, `snapshots`, `manifests`, `meta_runs`

## Model management

The forecast ensemble and purpose-specific models are configured in [`pythia/config.yaml`](pythia/config.yaml) under `llm.profiles`. Each profile contains an `ensemble` list supporting both string format (`provider:model_id`) and dict format with per-model parameters (`thinking`, `temperature`). To add, remove, or swap a model, edit this list.

```yaml
llm:
  profile: "prod"        # or "test"; override with PYTHIA_LLM_PROFILE

  profiles:
    test:
      ensemble:
        - google:gemini-3-flash-preview

    prod:
      ensemble:
        - openai:gpt-5.2
        - anthropic:claude-sonnet-4-6
        - google:gemini-3.1-pro-preview
        - google:gemini-3-flash-preview
        - kimi:kimi-k2.5
        - deepseek:deepseek-reasoner
        - openai:gpt-5-mini

      # Purpose-specific overrides (optional)
      hs_fallback: openai:gpt-5.2
      scenario_writer: google:gemini-3-flash-preview
```

### Changing models

- **Swap a model**: change the `provider:model_id` line in the ensemble list.
- **Add a model**: add a new line. Multiple models from the same provider are supported (e.g. two Google models).
- **Remove a model**: delete the line.
- **Add a new provider**: requires adding a `call_<provider>()` function and dispatch branch in `forecaster/providers.py`, plus an entry in `_PROVIDER_ENV_KEYS`.

### Model costs

Per-model cost rates are stored in [`pythia/model_costs.json`](pythia/model_costs.json) as `[input, output]` cost per 1,000 tokens in USD. When switching to a new model ID, add its cost entry to this file.

### Purpose-specific overrides

The `hs_fallback` and `scenario_writer` keys under a profile set the default model for HS triage fallback and scenario generation respectively. These can still be overridden at runtime via `PYTHIA_HS_FALLBACK_MODEL_SPECS` and `PYTHIA_SCENARIO_MODEL_ID` env vars.

### Env var overrides

- `PYTHIA_SPD_ENSEMBLE_SPECS`: overrides the entire SPD ensemble at runtime (comma-separated `provider:model_id` pairs).
- `PYTHIA_BLOCK_PROVIDERS`: comma-separated provider names to exclude (e.g. `xai`).
- `PYTHIA_SPD_GOOGLE_MODEL_ID`: overrides all Google model IDs in the SPD ensemble.

## Quickstart (local)

### 1) Clone + install
```bash
git clone <YOUR_REPO_URL>
cd Pythia
python -m venv .venv
source .venv/bin/activate
pip install -r python_library_requirements.txt
```

### 2) Configure the DB and profile
```bash
export PYTHIA_DB_URL="duckdb:///data/resolver.duckdb"
export PYTHIA_LLM_PROFILE="test"  # test or prod
```

### 3) Ensure the schema exists
```bash
python - <<'PY'
from pythia.db.schema import ensure_schema
ensure_schema()
print("schema ready")
PY
```

### 4) (Optional) Enable tail packs
```bash
export PYTHIA_HS_HAZARD_TAIL_PACKS_ENABLED=1  # tail packs are off by default
# Note: The question-level web research pipeline is deprecated.
# SPD prompts receive structured data directly via _load_structured_data().
# Grounding uses OpenAI web search (primary) + Gemini (fallback).
```

### 5) Run Horizon Scanner
- Default list: [`horizon_scanner/hs_country_list.txt`](horizon_scanner/hs_country_list.txt)
```bash
PYTHIA_LLM_PROFILE=test python -m horizon_scanner.horizon_scanner
```

### 6) Run the forecaster (bounded)
```bash
PYTHIA_LLM_PROFILE=test python -m forecaster.cli --mode pythia --limit 20 --purpose local_smoke
```

### 7) Inspect outputs
- **DuckDB**: open the database at `PYTHIA_DB_URL` (`app.db_url` default is `duckdb:///data/resolver.duckdb`).
- **Debug bundle** (optional):
```bash
python -m scripts.dump_pythia_debug_bundle \
  --db "${PYTHIA_DB_URL}" \
  --hs-run-id "<HS_RUN_ID>" \
  --forecaster-run-id "<FORECASTER_RUN_ID>"
```
- Bundles default to `debug/pytia_debug_bundle__<run_id>.md` (note: filename uses `pytia_`).

## Running in GitHub Actions

### Workflows
- **Production pipeline**: [`run_horizon_scanner.yml`](.github/workflows/run_horizon_scanner.yml) runs HS + forecaster end-to-end.
- **Resolver Update (backfill)**: [`resolver_update.yml`](.github/workflows/resolver_update.yml) — primary data ingestion (15th monthly). Single-job workflow with 5 sequential phases: (1) Resolver connectors (fatal), (2) Resolution sources, (3) Structured data, (4) Context sources, (5) Verify + export artifact.
- **Structured data refresh**: [`ingest-structured-data.yml`](.github/workflows/ingest-structured-data.yml) — mid-cycle refresh for fast-changing sources (weekly Sunday 03:00 UTC): conflict forecasts, GDACS, ReliefWeb, ACLED political.
- **Post-forecast pipeline**: [`compute_resolutions.yml`](.github/workflows/compute_resolutions.yml), [`compute_scores.yml`](.github/workflows/compute_scores.yml), [`compute_calibration_pythia.yml`](.github/workflows/compute_calibration_pythia.yml).
- **Forecaster CI**: [`forecaster-ci.yml`](.github/workflows/forecaster-ci.yml) covers SPD unit tests and optional compare artifacts.
- **ENSO refresh**: [`refresh-enso.yml`](.github/workflows/refresh-enso.yml) refreshes the ENSO state/forecast cache (scrapes IRI/CPC).
- **Seasonal TC refresh**: [`refresh-seasonal-tc.yml`](.github/workflows/refresh-seasonal-tc.yml) refreshes the seasonal TC forecast cache (TSR, NOAA CPC, BoM scrapers).
- **CrisisWatch refresh**: [`refresh-crisiswatch.yml`](.github/workflows/refresh-crisiswatch.yml) fetches ICG CrisisWatch data via Playwright (headless Chromium) on the 3rd of each month, parses with BeautifulSoup, and commits updated JSON to `horizon_scanner/data/crisiswatch_latest.json`.
- **DuckDB inspection**: [`inspect_resolver_duckdb.yml`](.github/workflows/inspect_resolver_duckdb.yml) — 7 data quality checks on the DB artifact.

### Canonical DB artifacts + signature guardrails
- Workflow downloads the canonical DB artifact (`pythia-resolver-db` by default), appends new HS + forecaster rows, and uploads the updated DB as a new artifact.
- It validates a **signature** (`scripts/ci/db_signature.py`) and writes:
  - `diagnostics/db_signature_before.json`
  - `diagnostics/db_signature_after.json`
- If signature validation fails, the workflow rejects the candidate DB and tries the next run. A missing/failed signature is a hard-stop; confirm required tables exist or re-run with a reset DB artifact for bootstrap runs.

### Artifact outputs
- **Debug bundle**: `debug/pytia_debug_bundle__<run_id>.md` (artifact)
- **SPD compare JSON**: `debug/spd_compare_smoke` or `debug/spd_compare_tests`
- **Diagnostics**: `diagnostics/` (DB signatures, compare JSON, latency summaries)
- **HS triage coverage**: `diagnostics/hs_triage_coverage__<HS_RUN_ID>.csv` and `diagnostics/hs_triage_failures__<HS_RUN_ID>.json`

### DB migrations
```bash
python scripts/migrate_llm_calls_telemetry.py --db duckdb:///path/to/resolver.duckdb
```

### HS triage reruns
- After the HS run completes, stdout includes `HS_TRIAGE_RERUN_ISO3S=...`. To re-run just those countries:
```bash
PYTHIA_HS_ONLY_COUNTRIES="<ISO3S>" python -m horizon_scanner.horizon_scanner
```

## Dashboard

### Run the API locally
```bash
PYTHIA_DB_URL="duckdb:///data/resolver.duckdb" uvicorn pythia.api.app:app --reload --port 8000
```

### Run the web UI locally
```bash
cd web
npm install
NEXT_PUBLIC_PYTHIA_API_BASE=http://localhost:8000/v1 npm run dev
```

### Pages and RC visibility
- **Forecast Index (Overview)**: shows the Humanitarian Impact Forecast Index, with RC KPI counts and RC map markers (highest RC level per country).
- **Forecasts** (`/questions`): latest forecasts table includes RC score, triage fields, and track assignment when `latest_only=true`.
- **HS Triage** (`/hs-triage`): displays per-run triage with RC likelihood/direction/magnitude/score and tier (quiet/priority).
- **Question detail** (`/questions/[questionId]`): RC fields appear alongside research and SPD outputs.
- **Countries** (`/countries`): includes highest RC level/score per country from the latest HS run.
- **Performance** (`/performance`): forecast evaluation with KPI cards, ensemble selector dropdown (compare ensemble_mean vs. ensemble_bayesmc), median and average scores (Brier, Log, CRPS), and views by Total, Hazard, Run, and Model.
- **Downloads** (`/downloads`): forecast/triage exports with RC fields, plus per-question score CSVs, model-level summary CSVs, and rationale exports.
- **About** (`/about`): versioned prompt snapshots and system overview history.

## Downloads / exports
- **Forecast SPD & EIV export**: `/v1/downloads/forecasts.csv` and `/v1/downloads/forecasts.xlsx` include RC probability/direction/magnitude/score columns and track assignment per row.
- **Score exports**: per-question CSV for each named ensemble (full 6-month x 5-bin grid, expected impact values, resolutions, and Brier/Log/CRPS scores), model-level summary CSV (avg/median/min/max per model and hazard), and rationale export (human-readable LLM explanations by hazard).
- **Countries endpoint**: `/v1/countries` includes `highest_rc_level`/`highest_rc_score` (latest HS run).
- **Questions endpoint**: `/v1/questions?latest_only=true` includes RC fields (`regime_change_*`) and `track`.

See [PUBLIC_APIS.md](PUBLIC_APIS.md) for canonical API contracts.

## Configuration

### Canonical config
- [`pythia/config.yaml`](pythia/config.yaml) is the authoritative configuration.
- There is no `pythia/config.py`; all runtime defaults live in the YAML + env vars.
- `app.db_url` defines DuckDB; `PYTHIA_DB_URL` overrides at runtime.
- `llm.profile` selects the default model bundle; override with `PYTHIA_LLM_PROFILE`.
- `llm.profiles.<name>.ensemble` defines the forecast ensemble; see [Model management](#model-management).
- [`pythia/model_costs.json`](pythia/model_costs.json) contains per-model cost rates.

### Key env vars
- **Provider keys**: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `KIMI_API_KEY`, `DEEPSEEK_API_KEY`, `EXA_API_KEY`, `PERPLEXITY_API_KEY`.
- **Structured data**: `ACAPS_EMAIL`, `ACAPS_PASSWORD` (ACAPS API auth); `ACLED_ACCESS_KEY`, `ACLED_EMAIL` (ACLED CAST API auth).
- **Concurrency**:
  - `PYTHIA_LLM_CONCURRENCY` (global LLM call cap)
  - `HS_MAX_WORKERS`
  - `FORECASTER_RESEARCH_MAX_WORKERS`
  - `FORECASTER_SPD_MAX_WORKERS`
- **Grounding**:
  - `PYTHIA_GROUNDING_PRIMARY_BACKEND`: `openai` (default) or `gemini` — controls which backend is tried first for RC/triage grounding.
  - `PYTHIA_ADVERSARIAL_CHECK_ENABLED`: Enable adversarial checks for RC L1+ (default `1`).
  - Note: The legacy `PYTHIA_WEB_RESEARCH_ENABLED`, `PYTHIA_RETRIEVER_ENABLED`, and `PYTHIA_HS_RESEARCH_WEB_SEARCH_ENABLED` are deprecated (set to `0` in workflow).
- **SPD tuning / timeouts (Gemini 3)**:
  - `PYTHIA_GOOGLE_SPD_THINKING_LEVEL_FLASH`, `PYTHIA_GOOGLE_SPD_THINKING_LEVEL_PRO`
  - `PYTHIA_GOOGLE_SPD_TIMEOUT_FLASH_SEC`, `PYTHIA_GOOGLE_SPD_TIMEOUT_PRO_SEC`
  - `PYTHIA_GOOGLE_SPD_RETRIES`
- **SPD ensemble override**: `PYTHIA_SPD_ENSEMBLE_SPECS` (e.g. `openai:gpt-5.2,google:gemini-3-flash-preview`).
- **HS triage resilience**:
  - `PYTHIA_HS_FALLBACK_MODEL_SPECS` (defaults to `hs_fallback` from the active profile, then `openai:gpt-5.2`; keeps HS triage running when Gemini fails).
  - `PYTHIA_HS_ONLY_COUNTRIES` (comma-separated ISO3s/names to rerun HS triage for a subset).
  - `PYTHIA_PROVIDER_FAILURE_THRESHOLD`, `PYTHIA_PROVIDER_COOLDOWN_SECONDS`, `PYTHIA_PROVIDER_RESET_ON_SUCCESS`
  - `PYTHIA_LLM_RETRY_TIMEOUTS` (set `0` to opt out of timeout retries outside HS triage)
  - `PYTHIA_HS_LLM_MAX_ATTEMPTS` (defaults to 3 for HS triage retries)
  - `PYTHIA_HS_GEMINI_TIMEOUT_SEC` (defaults to 120s for HS triage)

## GitHub Secrets setup

| Secret | Used by | Required for |
| --- | --- | --- |
| `OPENAI_API_KEY` | Forecaster SPD ensemble | OpenAI models in ensemble |
| `GEMINI_API_KEY` | HS, retriever, forecaster | Gemini models (HS + SPD + retriever) |
| `ANTHROPIC_API_KEY` | Forecaster SPD ensemble | Anthropic models |
| `XAI_API_KEY` | Forecaster SPD ensemble | XAI models |
| `KIMI_API_KEY` | Forecaster SPD ensemble | Kimi models (Moonshot) |
| `DEEPSEEK_API_KEY` | Forecaster SPD ensemble | DeepSeek models |
| `EXA_API_KEY` | Web research | Exa backend (optional) |
| `PERPLEXITY_API_KEY` | Web research | Perplexity backend (optional) |
| `ACAPS_EMAIL` | Structured data | ACAPS API auth (INFORM, Risk Radar, etc.) |
| `ACAPS_PASSWORD` | Structured data | ACAPS API auth |
| `ACLED_ACCESS_KEY` | Structured data | ACLED CAST API auth |
| `ACLED_EMAIL` | Structured data | ACLED CAST API auth |
| `GITHUB_TOKEN` | Actions | Artifact download + summary updates |

**Minimum set for full pipeline**: `GEMINI_API_KEY` + at least one of (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, `KIMI_API_KEY`, `DEEPSEEK_API_KEY`). Missing keys disable their providers, and the run proceeds with a partial ensemble.

## Operational notes
- **Partial ensembles are expected**: providers can timeout; the ensemble uses available members and records `ensemble_meta`.
- **Timeouts cap tail risk**: per-provider timeouts are enforced; increase them only if needed.
- **Shared retriever**: evidence packs are cached and reused to stabilize sources and reduce cost.
- **HS triage always writes**: HS stores triage outputs even if no questions are produced.

## Known limitations and challenges
- **Prediction market retriever disabled**: Metaculus returns 403, Polymarket returns 422. Do not re-enable until upstream APIs are fixed.
- **IPC API unavailable**: The IPC connector requires `IPC_API_KEY` which is not available; FEWS NET Phase 3+ data is used instead.
- **IFRC Montandon sparsity**: natural hazard PA data may be sparse for some hazards/countries, reducing base-rate strength.
- **DI, CU, HW fully silenced**: Displacement Influx, Civil Unrest, and Heatwave hazards are blocked at the hazard catalog level and never enter RC, triage, grounding, or question generation.
- **Occasional “200 but ungrounded”**: some web search calls can return success without verified sources. OpenAI is the primary grounding backend; Gemini is the fallback.

## Troubleshooting
- **No questions generated**: HS may mark all tiers quiet. Check `hs_triage` in DuckDB and confirm your country list (`horizon_scanner/hs_country_list.txt`) and `hazards_allowed` in config.
- **No hazard tail packs**: confirm `PYTHIA_HS_HAZARD_TAIL_PACKS_ENABLED=1` and RC Level ≥1 for the hazard. Tail packs are limited to 2 hazards per country.
- **RC fields missing in API**: ensure `hs_triage` has `regime_change_*` columns (`pythia/db/schema.py`) and that `latest_only=true` is set on `/v1/questions`.
- **No active models**: verify `PYTHIA_LLM_PROFILE`, `llm.profiles` ensemble config in `pythia/config.yaml`, and provider API keys.
- **Debug bundle too large for step summary**: artifacts still exist under `debug/` even if GitHub Step Summary truncates.
- **Slow runs**: Gemini tails can dominate latency. Tune `PYTHIA_LLM_CONCURRENCY`, `FORECASTER_*_MAX_WORKERS`, and SPD timeouts.
- **Interpreting question_run_metrics**: `question_run_metrics` (if present) records wall-clock vs compute vs queue time per question; see `scripts/dump_pythia_debug_bundle.py`.

## Cross-links
- Non-technical system overview: [`docs/fred_overview.md`](docs/fred_overview.md)
- Config: [`pythia/config.yaml`](pythia/config.yaml)
- Model costs: [`pythia/model_costs.json`](pythia/model_costs.json)
- Horizon Scanner: [`horizon_scanner/horizon_scanner.py`](horizon_scanner/horizon_scanner.py)
- Regime Change docs: [`docs/hs_regime_change.md`](docs/hs_regime_change.md)
- Hazard Tail Packs docs: [`docs/hs_hazard_tail_packs.md`](docs/hs_hazard_tail_packs.md)
- HS country list: [`horizon_scanner/hs_country_list.txt`](horizon_scanner/hs_country_list.txt)
- Forecaster CLI: [`forecaster/cli.py`](forecaster/cli.py)
- Hazard-specific prompts: [`forecaster/hazard_prompts.py`](forecaster/hazard_prompts.py)
- Per-hazard RC prompts: [`horizon_scanner/rc_prompts.py`](horizon_scanner/rc_prompts.py)
- Per-hazard triage prompts: [`horizon_scanner/hs_triage_prompts.py`](horizon_scanner/hs_triage_prompts.py)
- RC grounding prompts: [`horizon_scanner/rc_grounding_prompts.py`](horizon_scanner/rc_grounding_prompts.py)
- Triage grounding prompts: [`horizon_scanner/hs_triage_grounding_prompts.py`](horizon_scanner/hs_triage_grounding_prompts.py)
- ENSO module: [`horizon_scanner/enso/`](horizon_scanner/enso/)
- Seasonal TC module: [`horizon_scanner/seasonal_tc/`](horizon_scanner/seasonal_tc/)
- HDX Signals: [`horizon_scanner/hdx_signals.py`](horizon_scanner/hdx_signals.py)
- ACLED CAST connector: [`resolver/connectors/acled_cast.py`](resolver/connectors/acled_cast.py)
- Conflict forecasts loader: [`horizon_scanner/conflict_forecasts.py`](horizon_scanner/conflict_forecasts.py)
- CrisisWatch: [`horizon_scanner/crisiswatch.py`](horizon_scanner/crisiswatch.py), [`scripts/refresh_crisiswatch.py`](scripts/refresh_crisiswatch.py)
- Structured data: [`pythia/acaps.py`](pythia/acaps.py), [`pythia/ipc_phases.py`](pythia/ipc_phases.py), [`horizon_scanner/reliefweb.py`](horizon_scanner/reliefweb.py), [`pythia/acled_political.py`](pythia/acled_political.py)
- Adversarial checks: [`pythia/adversarial_check.py`](pythia/adversarial_check.py)
- Calibration advice: [`pythia/tools/generate_calibration_advice.py`](pythia/tools/generate_calibration_advice.py)
- Schema: [`pythia/db/schema.py`](pythia/db/schema.py)
- Debug bundle script: [`scripts/dump_pythia_debug_bundle.py`](scripts/dump_pythia_debug_bundle.py)
- Workflows: [`run_horizon_scanner.yml`](.github/workflows/run_horizon_scanner.yml), [`forecaster-ci.yml`](.github/workflows/forecaster-ci.yml), [`refresh-enso.yml`](.github/workflows/refresh-enso.yml), [`refresh-seasonal-tc.yml`](.github/workflows/refresh-seasonal-tc.yml), [`refresh-crisiswatch.yml`](.github/workflows/refresh-crisiswatch.yml)
- Public API contracts: [`PUBLIC_APIS.md`](PUBLIC_APIS.md)

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards and workflow notes. Archived READMEs live in [docs/archive/README_INDEX.md](docs/archive/README_INDEX.md).

## License
This repository follows the licensing terms bundled with the codebase (see `LICENSE` if present or repository metadata).
